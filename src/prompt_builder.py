"""
Prompt Builder — constructs dynamic, evolving prompts for Claude.
"""

import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds generation prompts that evolve over time.

    Constructs prompts with:
      1. Agent context (system prompt)
      2. Intent descriptions with key signals
      3. Similar question examples (for reference / anti-duplication)
      4. Confusion requirements and techniques
      5. Difficulty constraints
    """

    CONFUSION_TECHNIQUES = {
        "medium": [
            "Combine two related but distinct domains in one question",
            "Use phrasing that could apply to multiple agricultural contexts",
            "Ask about a topic that spans two intent categories",
        ],
        "hard": [
            "Use ambiguous phrasing that makes intent classification unclear",
            "Mix location-specific context with unrelated topics",
            "Combine temporal queries with procedural questions",
            "Ask compound questions spanning different agricultural domains",
            "Use conditional phrasing (if/then) that involves multiple intents",
        ],
        "expert": [
            "Create a question where 3+ intents are genuinely interleaved",
            "Use hypothetical scenarios that blend weather, pricing, and farming advice",
            "Embed a subsidy/scheme question inside a cultivation query",
            "Mix livestock and crop concerns in a single natural question",
            "Ask about cascading effects that cross multiple intent domains",
            "Create a question with implicit intent that requires deep parsing",
        ],
    }

    def __init__(self, agent_system_prompt: str, config=None):
        self.agent_prompt = agent_system_prompt
        self.config = config
        self.version = 1

    def build_system_prompt(self, persona: Optional[object] = None) -> str:
        """Build the system-level prompt for the LLM."""
        base_prompt = (
            "You are an expert question designer for evaluating agricultural chatbots. "
            "Your goal is to generate realistic, confusing, multi-intent questions that "
            "a real farmer might ask. These questions should be challenging for intent "
            "classification systems to categorize correctly.\n\n"
        )
        
        if persona:
            base_prompt += (
                f"ADOPT THE FOLLOWING PERSONA:\n"
                f"{persona.to_xml()}\n\n"
                "Speak, think, and ask questions exactly as this farmer would. "
                "Use their vocabulary, concerns, and perspective.\n\n"
            )
            
        return base_prompt + (
            "The chatbot you are testing has the following capabilities:\n"
            f"{self.agent_prompt}\n\n"
            "IMPORTANT: Generate all questions in ENGLISH only."
        )

    def build_generation_prompt(
        self,
        intent_mix: List[Tuple[int, float]],
        intent_details: List[Dict],
        similar_questions: List[Tuple[str, float]],
        difficulty: str,
        generation_count: int,
        batch_size: int = 5,
    ) -> str:
        """
        Construct the full generation prompt.

        Args:
            intent_mix: List of (intent_id, weight) tuples
            intent_details: Full intent dicts for the selected intents
            similar_questions: Reference questions to avoid duplicating
            difficulty: 'medium', 'hard', or 'expert'
            generation_count: Total questions generated so far
            batch_size: Number of questions to generate in this call
        """
        sections = []

        # ── Section 1: Intent Mix Target ─────────────────────────────────
        sections.append("## TARGET INTENT MIX\n")
        sections.append(
            "Generate questions that blend the following intents. "
            "Each question should genuinely confuse an intent classifier "
            "about which category it belongs to.\n"
        )
        for (intent_id, weight), details in zip(intent_mix, intent_details):
            sections.append(
                f"- **Intent {intent_id}: {details['name']}** (weight: {weight})\n"
                f"  Primary intent: {details['primary_intent']}\n"
                f"  Key signals: {', '.join(details['key_signals'])}\n"
                f"  Description: {details['description']}\n"
            )

        # ── Section 2: Reference Questions ───────────────────────────────
        if similar_questions:
            sections.append("\n## REFERENCE QUESTIONS (DO NOT DUPLICATE)\n")
            sections.append(
                "These are existing questions in the database. Use them as stylistic "
                "reference but DO NOT copy or closely paraphrase them:\n"
            )
            for i, (q, score) in enumerate(similar_questions[:8], 1):
                sections.append(f"  {i}. {q} (similarity: {score:.2f})")

        # ── Section 3: Generation Requirements ───────────────────────────
        intent_names = [d["name"] for d in intent_details]
        n_intents = len(intent_mix)
        weight_desc = ", ".join(
            f"{d['name']}={w}" for (_, w), d in zip(intent_mix, intent_details)
        )

        sections.append(f"\n## GENERATION REQUIREMENTS\n")
        sections.append(f"- Generate exactly **{batch_size}** questions")
        sections.append(f"- Each question must blend {n_intents} intents: {', '.join(intent_names)}")
        sections.append(f"- Intent weight distribution: {weight_desc}")
        sections.append(f"- Difficulty level: **{difficulty.upper()}**")
        sections.append("- All questions must be in **English**")
        sections.append("- Questions must be realistic — something a farmer would actually ask")
        sections.append("- Questions must be answerable by an agricultural chatbot")
        sections.append("- Questions should test edge cases of intent classification")

        # ── Section 4: Confusion Techniques ──────────────────────────────
        techniques = self.CONFUSION_TECHNIQUES.get(difficulty, self.CONFUSION_TECHNIQUES["hard"])
        sections.append(f"\n## CONFUSION TECHNIQUES TO USE (Difficulty: {difficulty})\n")
        for t in techniques:
            sections.append(f"- {t}")

        # ── Section 5: Diversity Guidance ────────────────────────────────
        if generation_count > 20:
            sections.append(
                f"\n## DIVERSITY NOTE\n"
                f"You have already generated {generation_count} questions. "
                f"Ensure these questions explore NEW angles, crop types, locations, "
                f"and phrasings. Avoid repeating patterns from earlier generations."
            )

        # ── Section 6: Output Format ─────────────────────────────────────
        sections.append(
            '\n## OUTPUT FORMAT\n'
            'Respond with a JSON array. Each element must be an object with these fields:\n'
            '```json\n'
            '[\n'
            '  {\n'
            '    "question": "The generated question text in English",\n'
            '    "expected_intents": [<intent_id_1>, <intent_id_2>],\n'
            '    "confusion_points": [\n'
            '      "Brief explanation of why this is confusing for classifiers"\n'
            '    ]\n'
            '  }\n'
            ']\n'
            '```\n'
            'Return ONLY the JSON array, no other text.'
        )

        return "\n".join(sections)

    def evolve_prompt_template(self, feedback_metrics: Optional[Dict] = None):
        """
        Update prompt strategies based on quality metrics.

        If diversity is low, emphasize variety. If duplication is high,
        strengthen anti-duplication instructions.
        """
        self.version += 1
        if feedback_metrics:
            if feedback_metrics.get("diversity", 1.0) < 0.5:
                logger.info("Low diversity detected — adding variety emphasis (v%d)", self.version)
                self.CONFUSION_TECHNIQUES["hard"].append(
                    "Use completely different crops, locations, and scenarios than previous questions"
                )
            if feedback_metrics.get("duplication_rate", 0) > 0.1:
                logger.info("High duplication — strengthening uniqueness constraints (v%d)", self.version)
