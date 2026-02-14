"""Ranking Agent for ordering candidates based on scores."""

import logging
from typing import Optional

from src.agents.base import AgentConfig, AgentContext, AgentResult, BaseAgent
from src.models.schemas import (
    RankedCandidate,
    RankingResult,
    Recommendation,
    ScreeningScore,
)

logger = logging.getLogger(__name__)


class RankingInput:
    """Input for Ranking Agent."""

    def __init__(
        self,
        scores: list[ScreeningScore],
        top_k: Optional[int] = None,
    ):
        self.scores = scores
        self.top_k = top_k


class RankingOutput:
    """Output from Ranking Agent."""

    def __init__(
        self,
        ranking_result: RankingResult,
    ):
        self.ranking_result = ranking_result

    @property
    def top_candidates(self) -> list[RankedCandidate]:
        return self.ranking_result.ranked_candidates


class RankingAgent(BaseAgent[RankingInput, RankingOutput]):
    """Agent responsible for ranking candidates based on screening scores.

    This agent:
    - Normalizes scores across candidates
    - Applies weighted scoring based on configuration
    - Ranks candidates by overall score
    - Assigns recommendation levels
    - Handles tiebreakers
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
    ):
        """Initialize the Ranking Agent.

        Args:
            config: Agent configuration
        """
        super().__init__(config or AgentConfig(name="ranking"))

    async def execute(
        self,
        input_data: RankingInput,
        context: AgentContext,
    ) -> AgentResult[RankingOutput]:
        """Execute candidate ranking.

        Args:
            input_data: Screening scores to rank
            context: Pipeline context

        Returns:
            AgentResult with ranking results
        """
        try:
            if not input_data.scores:
                logger.warning("No scores provided for ranking")
                return AgentResult.success_result(
                    RankingOutput(RankingResult(
                        job_id=context.job_id,
                        total_candidates=0,
                        ranked_candidates=[],
                    )),
                    self.name,
                )

            # Sort by overall score (descending)
            sorted_scores = sorted(
                input_data.scores,
                key=lambda s: (s.overall_score, s.experience_match, s.education_match),
                reverse=True,
            )

            # Calculate percentiles
            total = len(sorted_scores)
            ranked_candidates = []

            for rank, score in enumerate(sorted_scores, start=1):
                percentile = ((total - rank + 1) / total) * 100

                # Determine recommendation
                recommendation = self._get_recommendation(score.overall_score, percentile)

                ranked_candidate = RankedCandidate(
                    candidate_id=score.candidate_id,
                    rank=rank,
                    score=score.overall_score,
                    percentile=percentile,
                    screening_score=score,
                    recommendation=recommendation,
                )

                ranked_candidates.append(ranked_candidate)

            # Apply top_k filter if specified
            if input_data.top_k and input_data.top_k < len(ranked_candidates):
                ranked_candidates = ranked_candidates[:input_data.top_k]

            # Store in context
            context.rankings = ranked_candidates

            ranking_result = RankingResult(
                job_id=context.job_id,
                total_candidates=total,
                ranked_candidates=ranked_candidates,
            )

            logger.info(f"Ranked {total} candidates, top score: {sorted_scores[0].overall_score:.3f}")

            return AgentResult.success_result(
                RankingOutput(ranking_result),
                self.name,
            )

        except Exception as e:
            logger.error(f"Ranking error: {e}")
            return AgentResult.failure_result([str(e)], self.name)

    def _get_recommendation(
        self,
        score: float,
        percentile: float,
    ) -> Recommendation:
        """Determine recommendation level based on score and percentile.

        Args:
            score: Overall screening score (0-1)
            percentile: Candidate's percentile in the pool

        Returns:
            Recommendation enum value
        """
        # Primary: based on absolute score
        if score >= 0.85:
            return Recommendation.HIGHLY_RECOMMENDED
        elif score >= 0.70:
            return Recommendation.RECOMMENDED
        elif score >= 0.50:
            return Recommendation.CONSIDER
        else:
            return Recommendation.NOT_RECOMMENDED

    def _apply_tiebreaker(
        self,
        candidates: list[tuple[ScreeningScore, int]],
    ) -> list[tuple[ScreeningScore, int]]:
        """Apply tiebreaker rules for candidates with same score.

        Args:
            candidates: List of (score, original_index) tuples with same overall score

        Returns:
            Sorted list with tiebreakers applied
        """
        # Tiebreaker priority:
        # 1. Experience match
        # 2. Skills match (number of matching skills)
        # 3. Education match
        return sorted(
            candidates,
            key=lambda x: (
                x[0].experience_match,
                len(x[0].matching_skills),
                x[0].education_match,
            ),
            reverse=True,
        )

    def validate_input(self, data: RankingInput) -> bool:
        """Validate ranking input."""
        if data.scores is None:
            logger.error("Scores list is None")
            return False
        return True

    def get_rank_distribution(
        self,
        ranked_candidates: list[RankedCandidate],
    ) -> dict[str, int]:
        """Get distribution of recommendations.

        Args:
            ranked_candidates: List of ranked candidates

        Returns:
            Dictionary with recommendation counts
        """
        distribution = {
            Recommendation.HIGHLY_RECOMMENDED.value: 0,
            Recommendation.RECOMMENDED.value: 0,
            Recommendation.CONSIDER.value: 0,
            Recommendation.NOT_RECOMMENDED.value: 0,
        }

        for candidate in ranked_candidates:
            distribution[candidate.recommendation.value] += 1

        return distribution

    def get_score_statistics(
        self,
        ranked_candidates: list[RankedCandidate],
    ) -> dict[str, float]:
        """Get statistical summary of scores.

        Args:
            ranked_candidates: List of ranked candidates

        Returns:
            Dictionary with score statistics
        """
        if not ranked_candidates:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}

        scores = [c.score for c in ranked_candidates]
        scores_sorted = sorted(scores)
        n = len(scores)

        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / n,
            "median": scores_sorted[n // 2] if n % 2 else (scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2,
        }
