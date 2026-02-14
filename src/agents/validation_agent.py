"""Validation Agent for validating rankings against historical decisions."""

import logging
from typing import Any, Optional

from src.agents.base import AgentConfig, AgentContext, AgentResult, BaseAgent
from src.models.schemas import (
    HistoricalDecision,
    RankedCandidate,
    ValidationResult,
)
from src.services.mongo_service import MongoService

logger = logging.getLogger(__name__)


class ValidationInput:
    """Input for Validation Agent."""

    def __init__(
        self,
        ranked_candidates: list[RankedCandidate],
        job_id: str,
        historical_decisions: Optional[list[HistoricalDecision]] = None,
    ):
        self.ranked_candidates = ranked_candidates
        self.job_id = job_id
        self.historical_decisions = historical_decisions or []


class ValidationOutput:
    """Output from Validation Agent."""

    def __init__(
        self,
        validation_result: ValidationResult,
    ):
        self.validation_result = validation_result

    @property
    def is_valid(self) -> bool:
        return self.validation_result.is_valid

    @property
    def agreement_rate(self) -> float:
        return self.validation_result.agreement_rate


class ValidationAgent(BaseAgent[ValidationInput, ValidationOutput]):
    """Agent responsible for validating rankings against historical decisions.

    This agent:
    - Compares system rankings with historical hiring decisions
    - Calculates agreement rate (target: 90%)
    - Detects ranking anomalies
    - Generates validation report
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        mongo_service: Optional[MongoService] = None,
    ):
        """Initialize the Validation Agent.

        Args:
            config: Agent configuration
            mongo_service: MongoDB service for fetching historical data
        """
        super().__init__(config or AgentConfig(name="validation"))
        self.mongo_service = mongo_service
        self.agreement_threshold = self.settings.validation_agreement_threshold

    async def execute(
        self,
        input_data: ValidationInput,
        context: AgentContext,
    ) -> AgentResult[ValidationOutput]:
        """Execute validation against historical decisions.

        Args:
            input_data: Rankings and historical decisions
            context: Pipeline context

        Returns:
            AgentResult with validation results
        """
        try:
            # Fetch historical decisions if not provided
            historical_decisions = input_data.historical_decisions
            if not historical_decisions and self.mongo_service:
                historical_decisions = await self._fetch_historical_decisions(
                    input_data.job_id
                )

            # Perform validation
            validation_result = self._validate_rankings(
                ranked_candidates=input_data.ranked_candidates,
                historical_decisions=historical_decisions,
                job_id=input_data.job_id,
                session_id=context.session_id,
            )

            # Check if meets threshold
            if validation_result.agreement_rate < self.agreement_threshold:
                logger.warning(
                    f"Validation below threshold: {validation_result.agreement_rate:.2%} < {self.agreement_threshold:.2%}"
                )

            logger.info(
                f"Validation complete: {validation_result.agreement_rate:.2%} agreement, "
                f"{validation_result.total_comparisons} comparisons"
            )

            return AgentResult.success_result(
                ValidationOutput(validation_result),
                self.name,
            )

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return AgentResult.failure_result([str(e)], self.name)

    async def _fetch_historical_decisions(
        self,
        job_id: str,
    ) -> list[HistoricalDecision]:
        """Fetch historical decisions from database.

        Args:
            job_id: Job ID to fetch decisions for

        Returns:
            List of historical decisions
        """
        if not self.mongo_service:
            return []

        try:
            await self.mongo_service.connect()
            decisions_data = await self.mongo_service.get_historical_decisions(job_id)
            return [
                HistoricalDecision.model_validate(d)
                for d in decisions_data
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch historical decisions: {e}")
            return []

    def _validate_rankings(
        self,
        ranked_candidates: list[RankedCandidate],
        historical_decisions: list[HistoricalDecision],
        job_id: str,
        session_id: str,
    ) -> ValidationResult:
        """Validate rankings against historical decisions.

        Args:
            ranked_candidates: Current system rankings
            historical_decisions: Past hiring decisions
            job_id: Job ID
            session_id: Session ID

        Returns:
            ValidationResult with metrics
        """
        if not historical_decisions:
            # No historical data to compare
            logger.info("No historical decisions available for validation")
            return ValidationResult(
                job_id=job_id,
                session_id=session_id,
                agreement_rate=1.0,  # No data = no disagreement
                total_comparisons=0,
                matches=0,
                mismatches=0,
                is_valid=True,
            )

        # Create lookup maps
        current_rankings = {c.candidate_id: c for c in ranked_candidates}
        historical_map = {d.candidate_id: d for d in historical_decisions}

        # Count matches and mismatches
        matches = 0
        mismatches = 0
        anomalies = []

        # For each historical decision, check if our ranking agrees
        for decision in historical_decisions:
            if decision.candidate_id not in current_rankings:
                # Candidate not in current pool, skip
                continue

            current = current_rankings[decision.candidate_id]

            # Agreement criteria:
            # - If hired historically, should be in top tier (HIGHLY_RECOMMENDED or RECOMMENDED)
            # - If shortlisted, should be at least CONSIDER
            # - If rejected, should be NOT_RECOMMENDED or CONSIDER

            if decision.was_hired:
                is_match = current.recommendation.value in [
                    "Highly Recommended",
                    "Recommended",
                ]
            elif decision.was_shortlisted:
                is_match = current.recommendation.value in [
                    "Highly Recommended",
                    "Recommended",
                    "Consider",
                ]
            else:
                # Rejected
                is_match = current.recommendation.value in [
                    "Consider",
                    "Not Recommended",
                ]

            if is_match:
                matches += 1
            else:
                mismatches += 1
                anomalies.append(
                    f"Candidate {decision.candidate_id}: "
                    f"Historical={'hired' if decision.was_hired else 'shortlisted' if decision.was_shortlisted else 'rejected'}, "
                    f"Current={current.recommendation.value}"
                )

        total = matches + mismatches
        agreement_rate = matches / total if total > 0 else 1.0

        return ValidationResult(
            job_id=job_id,
            session_id=session_id,
            agreement_rate=agreement_rate,
            total_comparisons=total,
            matches=matches,
            mismatches=mismatches,
            anomalies=anomalies[:10],  # Limit to 10 anomalies
            is_valid=agreement_rate >= self.agreement_threshold,
        )

    def detect_ranking_anomalies(
        self,
        ranked_candidates: list[RankedCandidate],
    ) -> list[str]:
        """Detect statistical anomalies in rankings.

        Args:
            ranked_candidates: List of ranked candidates

        Returns:
            List of anomaly descriptions
        """
        anomalies = []

        if not ranked_candidates:
            return anomalies

        # Check for score clustering
        scores = [c.score for c in ranked_candidates]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        if variance < 0.01:
            anomalies.append(
                f"Low score variance ({variance:.4f}): candidates may not be well differentiated"
            )

        # Check for extreme scores
        if max(scores) > 0.95:
            anomalies.append(
                f"Very high top score ({max(scores):.2f}): verify scoring calibration"
            )

        if min(scores) < 0.1:
            anomalies.append(
                f"Very low bottom score ({min(scores):.2f}): verify scoring calibration"
            )

        # Check recommendation distribution
        rec_counts = {}
        for c in ranked_candidates:
            rec = c.recommendation.value
            rec_counts[rec] = rec_counts.get(rec, 0) + 1

        total = len(ranked_candidates)
        if rec_counts.get("Highly Recommended", 0) > total * 0.5:
            anomalies.append(
                "Over 50% of candidates are Highly Recommended: scoring may be too lenient"
            )

        if rec_counts.get("Not Recommended", 0) > total * 0.7:
            anomalies.append(
                "Over 70% of candidates are Not Recommended: scoring may be too strict"
            )

        return anomalies
