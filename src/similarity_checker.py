"""
Similarity Checker — prevents duplicate questions using cosine similarity.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SimilarityChecker:
    """
    Detects semantic similarity to prevent duplicate question generation.

    Uses cosine similarity between sentence-transformer embeddings.
    Tracks both existing questions and newly generated ones.
    """

    def __init__(self, config, embedding_model=None):
        """
        Args:
            config: Config object with paths and thresholds.
            embedding_model: SentenceTransformer model instance (loaded externally).
        """
        self.config = config
        self.model = embedding_model

        # Load existing data
        self.questions_df = self._load_questions(config.QUESTIONS_CSV_PATH)
        self.existing_embeddings = self._load_embeddings(config.EMBEDDINGS_CSV_PATH)

        # Track generated questions
        self.generated_questions: List[str] = []
        self.generated_embeddings: List[np.ndarray] = []

        logger.info(
            "SimilarityChecker ready: %d existing questions, embedding dim=%d",
            len(self.questions_df),
            self.existing_embeddings.shape[1] if self.existing_embeddings.size > 0 else 0,
        )

    # ── Data Loading ─────────────────────────────────────────────────────

    @staticmethod
    def _load_questions(path: Path) -> pd.DataFrame:
        """Load questions CSV."""
        df = pd.read_csv(path)
        logger.info("Loaded %d questions from %s", len(df), path)
        return df

    @staticmethod
    def _load_embeddings(path: Path) -> np.ndarray:
        """Load pre-computed embeddings as a numpy array."""
        df = pd.read_csv(path)
        embeddings = df.values.astype(np.float32)
        logger.info("Loaded embeddings: shape=%s", embeddings.shape)
        return embeddings

    # ── Embedding ────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Cannot encode text.")
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].astype(np.float32)

    # ── Similarity Computation ───────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _cosine_similarity_batch(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and a matrix of vectors."""
        norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1.0, norms)
        safe_query_norm = max(query_norm, 1e-10)
        similarities = np.dot(matrix, query_vec) / (safe_norms * safe_query_norm)
        return similarities

    # ── Duplicate Detection ──────────────────────────────────────────────

    def is_duplicate(
        self,
        new_question: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Check if the question is too similar to any existing or generated question.

        Returns:
            (is_duplicate: bool, max_similarity: float)
        """
        threshold = threshold or self.config.DUPLICATE_THRESHOLD
        query_emb = self.encode(new_question)

        max_sim = 0.0

        # Check against existing questions
        if self.existing_embeddings.size > 0:
            sims = self._cosine_similarity_batch(query_emb, self.existing_embeddings)
            max_sim = max(max_sim, float(np.max(sims)))

        # Check against previously generated questions
        if self.generated_embeddings:
            gen_matrix = np.array(self.generated_embeddings, dtype=np.float32)
            sims = self._cosine_similarity_batch(query_emb, gen_matrix)
            max_sim = max(max_sim, float(np.max(sims)))

        is_dup = max_sim >= threshold
        if is_dup:
            logger.debug("Duplicate detected (sim=%.3f >= %.3f): %s",
                         max_sim, threshold, new_question[:80])
        return is_dup, max_sim

    # ── Tracking ─────────────────────────────────────────────────────────

    def add_generated_question(self, question: str, embedding: Optional[np.ndarray] = None):
        """Add a generated question to the internal tracker."""
        self.generated_questions.append(question)
        if embedding is None:
            embedding = self.encode(question)
        self.generated_embeddings.append(embedding)

    # ── Reference Retrieval ──────────────────────────────────────────────

    def find_similar_questions(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find top-k most similar existing questions for context/reference.

        Returns:
            List of (question_text, similarity_score) tuples, sorted descending.
        """
        min_sim = min_similarity or self.config.SIMILAR_REFERENCE_THRESHOLD
        query_emb = self.encode(query)

        if self.existing_embeddings.size == 0:
            return []

        sims = self._cosine_similarity_batch(query_emb, self.existing_embeddings)
        top_indices = np.argsort(sims)[::-1][:top_k * 2]  # get extra, then filter

        results = []
        question_col = self.questions_df.columns[0]  # 'question'
        for idx in top_indices:
            score = float(sims[idx])
            if score >= min_sim:
                text = str(self.questions_df.iloc[idx][question_col])
                results.append((text, score))
            if len(results) >= top_k:
                break

        return results

    @property
    def total_tracked(self) -> int:
        """Total number of questions tracked (existing + generated)."""
        return len(self.questions_df) + len(self.generated_questions)
