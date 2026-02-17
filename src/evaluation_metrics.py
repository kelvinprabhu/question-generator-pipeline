"""
Evaluation Metrics — quality scoring for generated questions.
"""

import logging
import numpy as np
from collections import Counter
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Calculates quality metrics for a set of generated questions.

    Metrics:
      - semantic_diversity: Avg pairwise distance between generated questions
      - intent_coverage: Fraction of all intents used at least once
      - duplication_rate: Fraction rejected as duplicates
      - difficulty_distribution: Counts by difficulty level
      - avg_intents_per_question: Mean number of intents per question
    """

    def __init__(self, total_intents: int = 26, embedding_model=None):
        """
        Args:
            total_intents: Number of active intents (excluding non-informational).
            embedding_model: SentenceTransformer model for diversity calculations.
        """
        self.total_intents = total_intents
        self.model = embedding_model

    def calculate_metrics(
        self,
        generated_questions: List[Dict],
        rejected_duplicates: int = 0,
    ) -> Dict:
        """
        Calculate all quality metrics.

        Args:
            generated_questions: List of question dicts with 'question', 'intents',
                                 'difficulty', etc.
            rejected_duplicates: Number of questions rejected as duplicates.
        """
        if not generated_questions:
            return {
                "total_generated": 0,
                "diversity": 0.0,
                "intent_coverage": 0.0,
                "duplication_rate": 0.0,
                "difficulty_distribution": {},
                "avg_intents_per_question": 0.0,
                "intent_distribution": {},
            }

        total = len(generated_questions)

        return {
            "total_generated": total,
            "diversity": self._semantic_diversity(generated_questions),
            "intent_coverage": self._intent_coverage(generated_questions),
            "duplication_rate": rejected_duplicates / max(1, total + rejected_duplicates),
            "difficulty_distribution": self._difficulty_distribution(generated_questions),
            "avg_intents_per_question": self._avg_intents(generated_questions),
            "intent_distribution": self._intent_distribution(generated_questions),
        }

    # ── Individual Metrics ───────────────────────────────────────────────

    def _semantic_diversity(self, questions: List[Dict]) -> float:
        """
        Calculate semantic diversity as average pairwise cosine distance.

        Returns a value between 0 (identical) and 1 (maximally diverse).
        """
        if self.model is None or len(questions) < 2:
            return 0.0

        texts = [q["question"] for q in questions]
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype=np.float32)

            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normalized = embeddings / norms

            # Pairwise cosine similarity matrix
            sim_matrix = np.dot(normalized, normalized.T)

            # Average of upper triangle (excluding diagonal)
            n = len(texts)
            upper_tri_indices = np.triu_indices(n, k=1)
            avg_similarity = float(np.mean(sim_matrix[upper_tri_indices]))

            # Diversity = 1 - similarity
            return round(1.0 - avg_similarity, 4)
        except Exception as e:
            logger.warning("Error computing diversity: %s", e)
            return 0.0

    def _intent_coverage(self, questions: List[Dict]) -> float:
        """Fraction of all active intents that appear at least once."""
        used_intents = set()
        for q in questions:
            for intent_id, _ in q.get("intents", []):
                used_intents.add(intent_id)
        coverage = len(used_intents) / max(1, self.total_intents)
        return round(coverage, 4)

    def _difficulty_distribution(self, questions: List[Dict]) -> Dict[str, int]:
        """Count questions by difficulty level."""
        counts = Counter(q.get("difficulty", "unknown") for q in questions)
        return dict(counts)

    def _avg_intents(self, questions: List[Dict]) -> float:
        """Average number of intents per question."""
        intent_counts = [len(q.get("intents", [])) for q in questions]
        return round(np.mean(intent_counts), 2) if intent_counts else 0.0

    def _intent_distribution(self, questions: List[Dict]) -> Dict[int, int]:
        """Count of how many times each intent has been used."""
        counts = Counter()
        for q in questions:
            for intent_id, _ in q.get("intents", []):
                counts[intent_id] += 1
        return dict(sorted(counts.items()))

    # ── Report ───────────────────────────────────────────────────────────

    def print_report(self, metrics: Dict):
        """Print a formatted quality report."""
        print("\n" + "=" * 60)
        print("  GENERATION QUALITY REPORT")
        print("=" * 60)
        print(f"  Total questions generated:  {metrics['total_generated']}")
        print(f"  Semantic diversity:         {metrics['diversity']:.4f}")
        print(f"  Intent coverage:            {metrics['intent_coverage']:.2%}")
        print(f"  Duplication rate:           {metrics['duplication_rate']:.2%}")
        print(f"  Avg intents per question:   {metrics['avg_intents_per_question']:.1f}")
        print(f"\n  Difficulty Distribution:")
        for level, count in sorted(metrics["difficulty_distribution"].items()):
            print(f"    {level:>8s}: {count}")
        print(f"\n  Intent Usage (top 10):")
        intent_dist = metrics.get("intent_distribution", {})
        sorted_intents = sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        for intent_id, count in sorted_intents:
            print(f"    Intent {intent_id:>2d}: {count} times")
        print("=" * 60 + "\n")
