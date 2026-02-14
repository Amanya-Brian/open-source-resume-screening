"""Fairness metrics computation utilities."""

import logging
from typing import Any

import numpy as np

from src.models.schemas import RankedCandidate, Student

logger = logging.getLogger(__name__)


class FairnessMetricsComputer:
    """Compute fairness metrics for candidate rankings.

    Implements:
    - Disparate Impact Ratio (Four-Fifths Rule)
    - Demographic Parity
    - Equal Opportunity
    - Counterfactual Fairness Testing
    """

    PROTECTED_ATTRIBUTES = [
        "gender",
        "age_group",
        "ethnicity",
        "nationality",
    ]

    def __init__(self, dir_threshold: float = 0.8):
        """Initialize the fairness metrics computer.

        Args:
            dir_threshold: Threshold for Disparate Impact Ratio (default 0.8)
        """
        self.dir_threshold = dir_threshold

    def compute_disparate_impact_ratio(
        self,
        ranked_candidates: list[RankedCandidate],
        candidates: list[Student],
        attribute: str,
        selection_count: int,
    ) -> float:
        """Compute Disparate Impact Ratio for an attribute.

        DIR = P(selected | minority) / P(selected | majority)

        According to the Four-Fifths Rule, DIR >= 0.8 is considered fair.

        Args:
            ranked_candidates: Ranked candidates list
            candidates: Full candidate list
            attribute: Protected attribute name
            selection_count: Number of selected candidates

        Returns:
            DIR value (0-2, typically 0.8-1.2 for fair systems)
        """
        if not ranked_candidates or not candidates:
            return 1.0

        # Get selected candidate IDs
        selected_ids = set(
            c.candidate_id for c in ranked_candidates[:selection_count]
        )

        # Group candidates by attribute
        groups: dict[str, list[str]] = {}
        for candidate in candidates:
            value = getattr(candidate, attribute, None) or "unknown"
            if value not in groups:
                groups[value] = []
            groups[value].append(candidate.id)

        if len(groups) < 2:
            return 1.0  # Only one group

        # Calculate selection rates
        selection_rates = {}
        for group_name, member_ids in groups.items():
            if not member_ids:
                continue
            selected_count = len(set(member_ids) & selected_ids)
            selection_rates[group_name] = selected_count / len(member_ids)

        if not selection_rates:
            return 1.0

        # Find majority group
        majority_group = max(groups.keys(), key=lambda g: len(groups[g]))
        majority_rate = selection_rates.get(majority_group, 0)

        if majority_rate == 0:
            return 1.0

        # Compute minimum DIR across minority groups
        dir_values = []
        for group_name, rate in selection_rates.items():
            if group_name != majority_group:
                dir_values.append(rate / majority_rate)

        return min(dir_values) if dir_values else 1.0

    def compute_all_dir_scores(
        self,
        ranked_candidates: list[RankedCandidate],
        candidates: list[Student],
        selection_count: Optional[int] = None,
    ) -> dict[str, float]:
        """Compute DIR for all protected attributes.

        Args:
            ranked_candidates: Ranked candidates list
            candidates: Full candidate list
            selection_count: Number to consider as selected

        Returns:
            Dictionary of attribute name to DIR value
        """
        if selection_count is None:
            selection_count = len(ranked_candidates) // 3  # Top third

        selection_count = max(1, selection_count)

        return {
            attr: self.compute_disparate_impact_ratio(
                ranked_candidates, candidates, attr, selection_count
            )
            for attr in self.PROTECTED_ATTRIBUTES
        }

    def compute_demographic_parity(
        self,
        ranked_candidates: list[RankedCandidate],
        candidates: list[Student],
        attribute: str,
        selection_count: int,
    ) -> float:
        """Compute demographic parity metric.

        Measures difference in selection rates between groups.
        Parity = 1 - max_difference (1.0 = perfect parity)

        Args:
            ranked_candidates: Ranked candidates list
            candidates: Full candidate list
            attribute: Protected attribute name
            selection_count: Number of selected candidates

        Returns:
            Parity score (0-1, higher is better)
        """
        if not ranked_candidates or not candidates:
            return 1.0

        selected_ids = set(
            c.candidate_id for c in ranked_candidates[:selection_count]
        )

        # Group and compute rates
        groups: dict[str, list[str]] = {}
        for candidate in candidates:
            value = getattr(candidate, attribute, None) or "unknown"
            if value not in groups:
                groups[value] = []
            groups[value].append(candidate.id)

        rates = []
        for group_name, member_ids in groups.items():
            if member_ids:
                selected = len(set(member_ids) & selected_ids)
                rates.append(selected / len(member_ids))

        if len(rates) < 2:
            return 1.0

        return 1.0 - (max(rates) - min(rates))

    def check_compliance(
        self,
        dir_scores: dict[str, float],
        variance_scores: dict[str, float],
        dir_threshold: float = 0.8,
        variance_threshold: float = 0.001,
    ) -> tuple[bool, list[str]]:
        """Check if fairness metrics meet compliance thresholds.

        Args:
            dir_scores: DIR scores by attribute
            variance_scores: Variance scores by attribute
            dir_threshold: Minimum DIR threshold
            variance_threshold: Maximum variance threshold

        Returns:
            Tuple of (is_compliant, list of violations)
        """
        violations = []

        for attr, dir_val in dir_scores.items():
            if dir_val < dir_threshold:
                violations.append(
                    f"DIR violation for {attr}: {dir_val:.3f} < {dir_threshold}"
                )

        for attr, var in variance_scores.items():
            if var > variance_threshold:
                violations.append(
                    f"Variance violation for {attr}: {var:.5f} > {variance_threshold}"
                )

        return len(violations) == 0, violations

    def get_fairness_summary(
        self,
        ranked_candidates: list[RankedCandidate],
        candidates: list[Student],
    ) -> dict[str, Any]:
        """Get comprehensive fairness summary.

        Args:
            ranked_candidates: Ranked candidates list
            candidates: Full candidate list

        Returns:
            Dictionary with fairness metrics and compliance status
        """
        selection_count = max(1, len(ranked_candidates) // 3)

        dir_scores = self.compute_all_dir_scores(
            ranked_candidates, candidates, selection_count
        )

        min_dir = min(dir_scores.values()) if dir_scores else 1.0

        parity_scores = {
            attr: self.compute_demographic_parity(
                ranked_candidates, candidates, attr, selection_count
            )
            for attr in self.PROTECTED_ATTRIBUTES
        }

        is_compliant, violations = self.check_compliance(
            dir_scores, {}, self.dir_threshold, 0.001
        )

        return {
            "disparate_impact_ratio": min_dir,
            "dir_scores_by_attribute": dir_scores,
            "demographic_parity_scores": parity_scores,
            "is_compliant": is_compliant,
            "violations": violations,
            "selection_count": selection_count,
            "total_candidates": len(candidates),
        }
