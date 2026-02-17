"""
Pydantic AI Agent — multi-provider question generation with FallbackModel.

Replaces the custom ProviderPool with pydantic-ai's built-in model abstractions.
Providers: Groq → Gemini → HuggingFace (in priority order, no Anthropic).
"""

import json
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .config import Config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Structured Output Models
# ═══════════════════════════════════════════════════════════════════════════

class GeneratedQuestion(BaseModel):
    """A single generated question from the LLM."""
    question: str = Field(description="The generated question text in English")
    expected_intents: List[int] = Field(default_factory=list, description="List of expected intent IDs")
    confusion_points: List[str] = Field(default_factory=list, description="Why this is confusing for classifiers")


class GenerationOutput(BaseModel):
    """Batch of generated questions."""
    questions: List[GeneratedQuestion]


# ═══════════════════════════════════════════════════════════════════════════
# Dependencies dataclass (passed to Agent tools via RunContext)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineDeps:
    """Shared dependencies injected into agent tools via RunContext."""
    intent_manager: object       # IntentManager instance
    similarity_checker: object   # SimilarityChecker instance
    prompt_builder: object       # PromptBuilder instance
    config: Config
    mongo_store: Optional[object] = None  # MongoStore instance (optional)


# ═══════════════════════════════════════════════════════════════════════════
# Build the FallbackModel from config
# ═══════════════════════════════════════════════════════════════════════════

def _build_fallback_model(config: Config):
    """
    Build a pydantic-ai FallbackModel from configured provider keys.

    Priority: Groq → Gemini → HuggingFace (no Anthropic).
    For providers with multiple keys, creates multiple model instances.
    """
    from pydantic_ai.models.fallback import FallbackModel

    models = []

    # ── Groq models ──────────────────────────────────────────────────────
    groq_keys = getattr(config, "GROQ_API_KEYS", [])
    groq_model_name = getattr(config, "GROQ_MODEL", "llama-3.1-70b-versatile")
    if groq_keys:
        from pydantic_ai.models.groq import GroqModel
        for key in groq_keys:
            # GroqModel doesn't accept api_key param, so set env var
            os.environ["GROQ_API_KEY"] = key
            models.append(GroqModel(groq_model_name))
            logger.info("Added Groq model: %s", groq_model_name)

    # ── Gemini models ────────────────────────────────────────────────────
    gemini_keys = getattr(config, "GEMINI_API_KEYS", [])
    gemini_model_name = getattr(config, "GEMINI_MODEL", "gemini-2.0-flash")
    if gemini_keys:
        from pydantic_ai.models.google import GoogleModel
        for key in gemini_keys:
            # GoogleModel expects GOOGLE_API_KEY env var
            os.environ["GOOGLE_API_KEY"] = key
            models.append(GoogleModel(gemini_model_name))
            logger.info("Added Gemini model: %s", gemini_model_name)

    # ── HuggingFace models ───────────────────────────────────────────────
    hf_keys = getattr(config, "HF_API_KEYS", [])
    hf_model_name = getattr(config, "HF_MODEL", "Qwen/Qwen3-32Bclear")
    if hf_keys:
        from pydantic_ai.models.huggingface import HuggingFaceModel
        for key in hf_keys:
            # HuggingFaceModel expects HF_TOKEN env var
            os.environ["HF_TOKEN"] = key
            models.append(HuggingFaceModel(hf_model_name))
            logger.info("Added HuggingFace model: %s", hf_model_name)

    # ── OpenRouter models ───────────────────────────────────────────────
    or_keys = getattr(config, "OPENROUTER_API_KEYS", [])
    or_model_name = getattr(config, "OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
    if or_keys:
        from pydantic_ai.models.openai import OpenAIModel
        for key in or_keys:
            # OpenRouter uses OpenAI client interface
            # Swap env var for safety/consistency with other providers
            os.environ["OPENAI_API_KEY"] = key
            os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
            
            # We must pass base_url explicitly or set OPENAI_BASE_URL
            models.append(OpenAIModel(or_model_name))
            logger.info("Added OpenRouter model: %s", or_model_name)

    if not models:
        raise RuntimeError(
            "No API keys configured for any provider. "
            "Set GROQ_API_KEYS, GEMINI_API_KEYS, or HF_API_KEYS in your .env file."
        )

    if len(models) == 1:
        return models[0]

    return FallbackModel(*models)


# ═══════════════════════════════════════════════════════════════════════════
# Create the Agent
# ═══════════════════════════════════════════════════════════════════════════

def create_agent(config: Config) -> Agent:
    """
    Create and configure the pydantic-ai Agent with FallbackModel and tools.

    The Agent uses tools to:
      - Sample intent mixes from the IntentManager
      - Find similar questions for deduplication context
      - Check for duplicate questions
    """
    model = _build_fallback_model(config)

    agent = Agent(
        model,
        deps_type=PipelineDeps,
        instructions=(
            "You are an expert question designer for evaluating agricultural chatbots. "
            "Your goal is to generate realistic, confusing, multi-intent questions that "
            "a real farmer might ask. These questions should be challenging for intent "
            "classification systems to categorize correctly.\n\n"
            "IMPORTANT: Generate all questions in ENGLISH only.\n\n"
            "When asked to generate questions, respond with a JSON array. Each element "
            "must be an object with these fields:\n"
            '- "question": The generated question text in English\n'
            '- "expected_intents": Array of intent IDs\n'
            '- "confusion_points": Array of brief explanations of why this is confusing\n\n'
            "Return ONLY the JSON array, no other text."
        ),
        retries=config.MAX_RETRIES,
    )

    # ── Register tools ───────────────────────────────────────────────────

    @agent.tool
    def sample_intents(ctx: RunContext[PipelineDeps], n_intents: int) -> str:
        """Sample a weighted mix of intents for question generation.

        Args:
            n_intents: Number of intents to mix (typically 2-3).

        Returns:
            JSON string with intent IDs, weights, names, descriptions, and key signals.
        """
        deps = ctx.deps
        intent_mix = deps.intent_manager.sample_intent_mix(n_intents=n_intents)
        intent_ids = [iid for iid, _ in intent_mix]
        details = deps.intent_manager.get_intent_details(intent_ids)

        result = []
        for (iid, weight), detail in zip(intent_mix, details):
            result.append({
                "intent_id": iid,
                "weight": weight,
                "name": detail["name"],
                "description": detail.get("description", ""),
                "primary_intent": detail.get("primary_intent", ""),
                "key_signals": detail.get("key_signals", []),
            })
        return json.dumps(result, ensure_ascii=False)

    @agent.tool
    def find_similar_questions(ctx: RunContext[PipelineDeps], query: str, top_k: int = 5) -> str:
        """Find existing questions similar to a query for reference/anti-duplication.

        Args:
            query: A representative query string (e.g. intent names + key signals).
            top_k: Maximum number of similar questions to return.

        Returns:
            JSON string with similar questions and their similarity scores.
        """
        deps = ctx.deps
        results = deps.similarity_checker.find_similar_questions(query, top_k=top_k)
        return json.dumps(
            [{"question": q, "similarity": round(s, 3)} for q, s in results],
            ensure_ascii=False,
        )

    @agent.tool
    def check_duplicate(ctx: RunContext[PipelineDeps], question: str) -> str:
        """Check if a question is too similar to existing ones.

        Args:
            question: The question text to check.

        Returns:
            JSON string with is_duplicate boolean and max_similarity score.
        """
        deps = ctx.deps
        is_dup, max_sim = deps.similarity_checker.is_duplicate(question)
        return json.dumps({"is_duplicate": is_dup, "max_similarity": round(max_sim, 4)})

    @agent.tool
    def save_to_mongo(
        ctx: RunContext[PipelineDeps],
        questions_json: str,
        batch_id: str,
    ) -> str:
        """Save generated questions to MongoDB.

        Args:
            questions_json: JSON string of question dicts to save.
            batch_id: Identifier for this batch.

        Returns:
            JSON string with the count of inserted documents.
        """
        deps = ctx.deps
        if deps.mongo_store is None:
            return json.dumps({"inserted": 0, "reason": "MongoDB not configured"})
        questions = json.loads(questions_json)
        count = deps.mongo_store.insert_questions(questions, batch_id=batch_id)
        return json.dumps({"inserted": count})

    logger.info("Agent created with %d tools", len(agent._toolsets) if hasattr(agent, '_toolsets') else 4)
    return agent


# ═══════════════════════════════════════════════════════════════════════════
# Direct generation helper (bypasses agent's tool-calling loop)
# ═══════════════════════════════════════════════════════════════════════════

def generate_with_agent(
    agent: Agent,
    deps: PipelineDeps,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Run the agent with a generation prompt and return raw text.

    This is a direct LLM call — the prompt is fully constructed by the
    QuestionGenerator, so we just need the LLM to produce JSON output.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    result = agent.run_sync(prompt, deps=deps)
    return result.output
