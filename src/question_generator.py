"""
Question Generator — core generation logic using pydantic-ai Agent.
"""

import json
import time
import random
import logging
from typing import List, Dict, Tuple, Optional

from pydantic_ai import Agent

from .intent_manager import IntentManager
from .similarity_checker import SimilarityChecker
from .prompt_builder import PromptBuilder
from .agent import PipelineDeps
from .config import Config

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generates multi-intent confusing questions using a pydantic-ai Agent.

    Orchestrates:
      - Intent sampling from IntentManager
      - Similar-question retrieval from SimilarityChecker
      - Prompt construction via PromptBuilder
      - LLM calls via pydantic-ai Agent (FallbackModel with rotation)
      - Post-generation validation and deduplication
    """

    def __init__(
        self,
        intent_manager: IntentManager,
        similarity_checker: SimilarityChecker,
        prompt_builder: PromptBuilder,
        agent: Agent,
        deps: PipelineDeps,
        config: Config,
    ):
        self.intent_manager = intent_manager
        self.similarity_checker = similarity_checker
        self.prompt_builder = prompt_builder
        self.agent = agent
        self.deps = deps
        self.config = config

        self._total_generated = 0
        self._total_rejected_duplicates = 0
        self._last_provider = None
        self._last_model = None

    # ── Batch Generation ─────────────────────────────────────────────────

    def generate_batch(
        self,
        batch_size: int = 10,
        difficulty: str = "hard",
        intent_mix_size: int = 3,
        persona: Optional[object] = None,
        target_intents: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Generate a batch of multi-intent confusing questions.

        Returns a list of question dicts with provider metadata.
        """
        logger.info(
            "Generating batch: size=%d, difficulty=%s, mix_size=%d%s",
            batch_size, difficulty, intent_mix_size,
            f", fixed_intents={target_intents}" if target_intents else "",
        )

        # 1. Determine intent mix
        if target_intents:
            # Use fixed intents if provided (override sampling)
            intent_ids = target_intents
            # Distribute weight evenly
            weight = 1.0 / len(intent_ids)
            intent_mix = [(iid, weight) for iid in intent_ids]
        else:
            # Sample random mix
            intent_mix = self.intent_manager.sample_intent_mix(n_intents=intent_mix_size)
            intent_ids = [iid for iid, _ in intent_mix]
        
        intent_details = self.intent_manager.get_intent_details(intent_ids)

        # 2. Build a representative query for similarity search
        representative_query = self._build_representative_query(intent_details)
        similar_questions = self.similarity_checker.find_similar_questions(
            representative_query, top_k=5
        )

        # 3. Build prompt
        generation_prompt = self.prompt_builder.build_generation_prompt(
            intent_mix=intent_mix,
            intent_details=intent_details,
            similar_questions=similar_questions,
            difficulty=difficulty,
            generation_count=self._total_generated,
            batch_size=batch_size,
        )

        # 4. Call LLM via pydantic-ai Agent (handles fallback across providers)
        raw_questions = self._call_llm(generation_prompt, persona=persona)

        # 5. Validate and deduplicate
        validated = self._validate_and_deduplicate(raw_questions, intent_mix, difficulty)

        # 6. Record intents used
        for q in validated:
            self.intent_manager.record_generation(q["intents"])

        self._total_generated += len(validated)
        logger.info(
            "Batch complete: %d/%d accepted (total: %d, dupes rejected: %d) via %s/%s",
            len(validated), len(raw_questions), self._total_generated,
            self._total_rejected_duplicates,
            self._last_provider or "?", self._last_model or "?",
        )

        return validated

    # ── LLM Call ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, persona: Optional[object] = None) -> List[Dict]:
        """Call LLM via pydantic-ai Agent and parse the JSON response."""
        system_prompt = self.prompt_builder.build_system_prompt(persona=persona)

        for attempt in range(1, self.config.MAX_RETRIES + 1):
            try:
                # Use the agent for the LLM call
                result = self.agent.run_sync(
                    f"{system_prompt}\n\n{prompt}",
                    deps=self.deps,
                )

                raw_text = result.output
                if not isinstance(raw_text, str):
                    raw_text = str(raw_text)

                # Extract provider/model info from the result
                if hasattr(result, 'all_messages') and result.all_messages():
                    for msg in result.all_messages():
                        if hasattr(msg, 'model_name') and msg.model_name:
                            parts = msg.model_name.split("/", 1)
                            self._last_provider = parts[0] if len(parts) > 1 else "unknown"
                            self._last_model = parts[-1]
                            break

                content = self._extract_json(raw_text)
                questions = json.loads(content)

                if isinstance(questions, list):
                    logger.info(
                        "LLM returned %d questions via %s/%s (attempt %d)",
                        len(questions), self._last_provider, self._last_model,
                        attempt,
                    )
                    return questions
                else:
                    logger.warning(
                        "LLM response is not a list (attempt %d, provider=%s)",
                        attempt, self._last_provider,
                    )

            except json.JSONDecodeError as e:
                logger.warning("JSON parse error on attempt %d: %s", attempt, e)
            except RuntimeError as e:
                logger.error("Agent error on attempt %d: %s", attempt, e, exc_info=True)
                if attempt < self.config.MAX_RETRIES:
                    time.sleep(5)
            except Exception as e:
                logger.error("Unexpected error on attempt %d: %s", attempt, e, exc_info=True)
                if attempt < self.config.MAX_RETRIES:
                    time.sleep(2 ** attempt)

        logger.error("All %d attempts failed. Returning empty batch.", self.config.MAX_RETRIES)
        return []

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from potential markdown code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # Remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove closing fence
            text = "\n".join(lines)
        return text.strip()

    # ── Validation ───────────────────────────────────────────────────────

    def _validate_and_deduplicate(
        self,
        raw_questions: List[Dict],
        intent_mix: List[Tuple[int, float]],
        difficulty: str,
    ) -> List[Dict]:
        """Validate structure and check for duplicates."""
        validated = []

        for q in raw_questions:
            if not isinstance(q, dict) or "question" not in q:
                logger.debug("Skipping malformed question: %s", q)
                continue

            question_text = q.get("question", "").strip()
            if not question_text or len(question_text) < 10:
                logger.debug("Skipping too-short question: %s", question_text)
                continue

            # Duplicate check
            is_dup, max_sim = self.similarity_checker.is_duplicate(question_text)
            if is_dup:
                self._total_rejected_duplicates += 1
                logger.debug("Rejected duplicate (sim=%.3f): %s", max_sim, question_text[:60])
                continue

            # Track the accepted question
            self.similarity_checker.add_generated_question(question_text)

            validated.append({
                "question": question_text,
                "intents": intent_mix,
                "difficulty": difficulty,
                "confusion_points": q.get("confusion_points", []),
                "expected_intents": q.get("expected_intents", [iid for iid, _ in intent_mix]),
                "similarity_score": max_sim,
                "provider": self._last_provider,
                "model": self._last_model,
            })

        return validated

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_representative_query(intent_details: List[Dict]) -> str:
        """Build a representative query string for similarity search."""
        parts = []
        for d in intent_details:
            parts.append(d["name"])
            parts.extend(d.get("key_signals", [])[:3])
        return " ".join(parts)

    @property
    def stats(self) -> Dict:
        """Return generation statistics."""
        return {
            "total_generated": self._total_generated,
            "total_rejected_duplicates": self._total_rejected_duplicates,
            "acceptance_rate": (
                self._total_generated / max(1, self._total_generated + self._total_rejected_duplicates)
            ),
            "last_provider": self._last_provider,
            "last_model": self._last_model,
        }
