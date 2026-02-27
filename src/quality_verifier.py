"""
Quality Verifier — post-generation agent that reviews and filters questions.

Reviews a batch of generated questions and removes any that feel artificial,
unnatural, or use overly technical/scientific language. Ensures questions
sound like they come from a real farmer.
"""

import json
import logging
from typing import List, Dict, Optional

from pydantic_ai import Agent

from .config import Config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Verification Prompt
# ═══════════════════════════════════════════════════════════════════════════

VERIFIER_SYSTEM_PROMPT = """You are a quality reviewer for agricultural chatbot test questions.

Your job is to review a batch of generated questions and REMOVE any that do NOT meet 
the quality standards below. You are acting as a gatekeeper — only natural, farmer-like 
questions should pass through.

## REJECTION CRITERIA — Remove a question if it has ANY of these problems:

1. **ARTIFICIAL / CONSTRUCTED**: The question feels like it was written by an AI or 
   a researcher, not by a real farmer. It sounds too structured, too polished, or 
   deliberately designed to be confusing.

2. **SCIENTIFIC / TECHNICAL LANGUAGE**: The question uses scientific names 
   (e.g. "Spodoptera frugiperda", "Xanthomonas oryzae"), complex jargon 
   (e.g. "integrated pest management", "phytosanitary", "fertigation", 
   "yield parameters"), or textbook terminology that a farmer would never use.

3. **NOT GROUNDED**: The question is hypothetical, academic, or doesn't come from 
   a real farming situation. A farmer asks because they have a real problem — 
   their crop is dying, prices are bad, they need scheme help, etc.

4. **UNNATURALLY COMPLEX**: The question tries too hard to combine multiple topics 
   in a way no real person would phrase. Real farmers ask messy, run-on questions,
   not carefully crafted multi-intent sentences.

5. **OVERLY FORMAL**: Uses language like "Kindly advise on...", "What strategies 
   exist for...", "Please elaborate on the impact of..." — real farmers speak 
   casually and directly.

## ACCEPTANCE CRITERIA — Keep a question if:

- It sounds like something a real farmer would say out loud
- It uses simple, everyday words to describe real problems
- The concerns are practical (crop damage, money, weather, pests, schemes, water)
- It's messy, informal, and genuinely confusing (not artificially so)

## OUTPUT FORMAT

Respond with a JSON object:
{
  "accepted": [
    {"question": "...", "expected_intents": [...], "confusion_points": [...]}
  ],
  "rejected": [
    {"question": "...", "reason": "Brief explanation of why rejected"}
  ]
}

Return ONLY the JSON object, no other text."""


class QualityVerifier:
    """
    Post-generation quality gate that uses LLM to review and filter questions.

    Reviews each batch of generated questions and removes any that:
    - Sound artificial or constructed
    - Use scientific/technical language
    - Are not grounded in real farming scenarios
    - Feel unnatural or overly formal
    """

    def __init__(self, agent: Agent, config: Config):
        """
        Args:
            agent: The pydantic-ai Agent instance (shared with generation).
            config: Config object with settings.
        """
        self.agent = agent
        self.config = config
        self._total_reviewed = 0
        self._total_rejected = 0

    def verify_batch(
        self,
        questions: List[Dict],
        deps: object,
    ) -> List[Dict]:
        """
        Review a batch of questions and return only those that pass quality checks.

        Args:
            questions: List of question dicts with 'question', 'expected_intents',
                       'confusion_points', and other metadata.
            deps: PipelineDeps instance for the agent.

        Returns:
            Filtered list of question dicts that passed verification.
        """
        if not questions:
            return []

        # Build the review prompt with the questions to verify
        questions_for_review = []
        for q in questions:
            questions_for_review.append({
                "question": q["question"],
                "expected_intents": q.get("expected_intents", []),
                "confusion_points": q.get("confusion_points", []),
            })

        review_prompt = (
            f"{VERIFIER_SYSTEM_PROMPT}\n\n"
            f"## QUESTIONS TO REVIEW\n\n"
            f"Review the following {len(questions_for_review)} questions and "
            f"categorize each as accepted or rejected:\n\n"
            f"```json\n{json.dumps(questions_for_review, indent=2, ensure_ascii=False)}\n```"
        )

        try:
            result = self.agent.run_sync(review_prompt, deps=deps)
            raw_text = result.output
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)

            # Extract JSON from potential markdown fences
            content = self._extract_json(raw_text)
            review_result = json.loads(content)

            # Extract accepted questions
            accepted_texts = set()
            if isinstance(review_result, dict) and "accepted" in review_result:
                for item in review_result["accepted"]:
                    if isinstance(item, dict) and "question" in item:
                        accepted_texts.add(item["question"].strip())

                # Log rejected questions
                rejected = review_result.get("rejected", [])
                for item in rejected:
                    if isinstance(item, dict):
                        logger.info(
                            "Quality rejected: '%s' — Reason: %s",
                            item.get("question", "?")[:80],
                            item.get("reason", "unknown"),
                        )
                self._total_rejected += len(rejected)
            else:
                # If the response doesn't have expected structure, accept all
                logger.warning(
                    "Verifier response didn't have expected structure, accepting all questions"
                )
                self._total_reviewed += len(questions)
                return questions

            # Filter original questions to preserve full metadata
            verified = []
            for q in questions:
                if q["question"].strip() in accepted_texts:
                    verified.append(q)

            self._total_reviewed += len(questions)
            logger.info(
                "Quality verification: %d/%d accepted (%d rejected total so far)",
                len(verified), len(questions), self._total_rejected,
            )

            return verified

        except json.JSONDecodeError as e:
            logger.warning("Verifier JSON parse error: %s — accepting all questions", e)
            self._total_reviewed += len(questions)
            return questions
        except Exception as e:
            logger.error("Verifier error: %s — accepting all questions", e, exc_info=True)
            self._total_reviewed += len(questions)
            return questions

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

    @property
    def stats(self) -> Dict:
        """Return verification statistics."""
        return {
            "total_reviewed": self._total_reviewed,
            "total_rejected": self._total_rejected,
            "rejection_rate": (
                self._total_rejected / max(1, self._total_reviewed)
            ),
        }
