"""
Intent Manager — manages the 28-intent taxonomy and dynamic weight evolution.
"""

import json
import random
import copy
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class IntentManager:
    """
    Manages intent taxonomy and dynamic weight evolution.

    Responsibilities:
      - Load 28 intents from JSON taxonomy
      - Weighted random sampling of intent mixes (2-3 intents per question)
      - Three weight-evolution strategies: adaptive, random_walk, coverage_based
      - Track weight history for logging / analysis
    """

    # Intents that are non-informational and should be excluded from generation
    EXCLUDED_INTENTS = {18, 25}  # Cluster 18 = Acknowledgements, 25 = Fragmented

    def __init__(
        self,
        intent_taxonomy_path: str,
        initial_weights: Optional[Dict[int, float]] = None,
        config=None,
    ):
        self.intents = self._load_intents(intent_taxonomy_path)
        self.active_intent_ids = [
            i["id"] for i in self.intents if i["id"] not in self.EXCLUDED_INTENTS
        ]
        self.current_weights = initial_weights or self._initialize_weights()
        self.weight_history: List[Dict[int, float]] = []
        self.generation_count = 0
        self.generated_intent_log: List[List[Tuple[int, float]]] = []
        self.config = config

    # ── Loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_intents(path: str) -> List[Dict]:
        """Load intents from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            intents = json.load(f)
        logger.info("Loaded %d intents from %s", len(intents), path)
        return intents

    def _initialize_weights(self) -> Dict[int, float]:
        """Uniform weights across active intents."""
        n = len(self.active_intent_ids)
        weight = 1.0 / n
        return {iid: weight for iid in self.active_intent_ids}

    # ── Sampling ─────────────────────────────────────────────────────────

    def sample_intent_mix(self, n_intents: int = 2) -> List[Tuple[int, float]]:
        """
        Sample n_intents based on current weights.

        Returns a list of (intent_id, normalized_weight) tuples.
        The weights within the mix are re-normalized to sum to 1.0.
        """
        ids = list(self.current_weights.keys())
        weights = [self.current_weights[i] for i in ids]

        chosen_ids = random.choices(ids, weights=weights, k=n_intents)
        # Avoid duplicate intents in a single mix
        chosen_ids = list(dict.fromkeys(chosen_ids))  # preserve order, remove dupes
        while len(chosen_ids) < n_intents:
            extra = random.choices(ids, weights=weights, k=1)[0]
            if extra not in chosen_ids:
                chosen_ids.append(extra)

        # Assign proportional weights within the mix
        raw = [self.current_weights[i] for i in chosen_ids]
        total = sum(raw)
        mix = [(iid, round(w / total, 3)) for iid, w in zip(chosen_ids, raw)]
        return mix

    # ── Intent Templates ─────────────────────────────────────────────────

    def get_intent_details(self, intent_ids: List[int]) -> List[Dict]:
        """Return full intent details for the given IDs."""
        id_set = set(intent_ids)
        return [i for i in self.intents if i["id"] in id_set]

    def get_intent_by_id(self, intent_id: int) -> Optional[Dict]:
        """Return a single intent by ID."""
        for i in self.intents:
            if i["id"] == intent_id:
                return i
        return None

    # ── Weight Evolution ─────────────────────────────────────────────────

    def record_generation(self, intent_mix: List[Tuple[int, float]]):
        """Record which intents were used in a generation."""
        self.generated_intent_log.append(intent_mix)
        self.generation_count += 1

    def evolve_weights(self, strategy: Optional[str] = None):
        """
        Update intent weights after a batch.

        Strategies:
          - 'adaptive':       Increase weight of underrepresented intents
          - 'random_walk':    Random perturbation
          - 'coverage_based': Ensure all intents get coverage
        """
        strategy = strategy or (self.config.EVOLUTION_STRATEGY if self.config else "adaptive")
        self.weight_history.append(copy.deepcopy(self.current_weights))

        if strategy == "adaptive":
            self._adaptive_evolution()
        elif strategy == "random_walk":
            self._random_walk_evolution()
        elif strategy == "coverage_based":
            self._coverage_based_evolution()
        else:
            raise ValueError(f"Unknown evolution strategy: {strategy}")

        self._normalize_weights()
        self._clamp_weights()
        self._normalize_weights()

        logger.info("Weights evolved (strategy=%s). Top 5: %s",
                     strategy, self._top_k_weights(5))

    def _adaptive_evolution(self):
        """Increase weights for under-represented intents, decrease over-represented."""
        usage = self._intent_usage_counts()
        total = max(sum(usage.values()), 1)

        for iid in self.active_intent_ids:
            usage_ratio = usage.get(iid, 0) / total
            expected_ratio = self.current_weights[iid]

            if usage_ratio < expected_ratio * 0.8:
                self.current_weights[iid] *= 1.1
            elif usage_ratio > expected_ratio * 1.2:
                self.current_weights[iid] *= 0.95

    def _random_walk_evolution(self):
        """Apply small random perturbations."""
        for iid in self.active_intent_ids:
            perturbation = random.gauss(0, 0.02)
            self.current_weights[iid] = max(0.01, self.current_weights[iid] + perturbation)

    def _coverage_based_evolution(self):
        """Heavily boost intents that have never been used."""
        usage = self._intent_usage_counts()
        for iid in self.active_intent_ids:
            if usage.get(iid, 0) == 0:
                self.current_weights[iid] *= 2.0
            elif usage.get(iid, 0) < 3:
                self.current_weights[iid] *= 1.3

    def _intent_usage_counts(self) -> Counter:
        """Count how many times each intent has been used."""
        counts = Counter()
        for mix in self.generated_intent_log:
            for intent_id, _ in mix:
                counts[intent_id] += 1
        return counts

    def _normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        total = sum(self.current_weights.values())
        if total > 0:
            self.current_weights = {
                k: v / total for k, v in self.current_weights.items()
            }

    def _clamp_weights(self):
        """Clamp weights to [MIN_WEIGHT, MAX_WEIGHT]."""
        min_w = self.config.MIN_WEIGHT if self.config else 0.05
        max_w = self.config.MAX_WEIGHT if self.config else 0.30
        for iid in self.active_intent_ids:
            self.current_weights[iid] = max(min_w, min(max_w, self.current_weights[iid]))

    def _top_k_weights(self, k: int = 5) -> List[Tuple[int, float]]:
        """Return top-k intents by weight."""
        sorted_w = sorted(self.current_weights.items(), key=lambda x: x[1], reverse=True)
        return [(iid, round(w, 4)) for iid, w in sorted_w[:k]]

    # ── Serialization ────────────────────────────────────────────────────

    def get_evolution_log(self) -> Dict:
        """Return a serializable evolution log."""
        return {
            "generation_count": self.generation_count,
            "current_weights": {str(k): round(v, 6) for k, v in self.current_weights.items()},
            "weight_history_length": len(self.weight_history),
            "intent_usage": dict(self._intent_usage_counts()),
        }
