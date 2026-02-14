"""Agent Orchestrator for coordinating the screening pipeline."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from src.agents.base import AgentConfig, AgentContext, AgentResult, BaseAgent
from src.agents.data_fetching_agent import (
    DataFetchingAgent,
    DataFetchingInput,
    DataFetchingOutput,
)
from src.agents.explanation_agent import (
    ExplanationAgent,
    ExplanationInput,
    ExplanationOutput,
)
from src.agents.fairness_agent import (
    FairnessAgent,
    FairnessInput,
    FairnessOutput,
)
from src.agents.ranking_agent import (
    RankingAgent,
    RankingInput,
    RankingOutput,
)
from src.agents.screening_agent import (
    ScreeningAgent,
    ScreeningInput,
    ScreeningOutput,
)
from src.agents.validation_agent import (
    ValidationAgent,
    ValidationInput,
    ValidationOutput,
)
from src.config.settings import Settings, get_settings
from src.models.schemas import (
    PipelineStatus,
    ScreeningOptions,
    ScreeningResult,
    ScreeningSession,
)
from src.services.embedding_service import EmbeddingService
from src.services.mongo_service import MongoService
from src.services.talentmatch_client import TalentMatchClient

logger = logging.getLogger(__name__)


class PipelineState:
    """State tracking for pipeline execution."""

    def __init__(self, job_id: str, session_id: Optional[str] = None):
        self.job_id = job_id
        self.session_id = session_id or str(uuid4())
        self.status = PipelineStatus.PENDING
        self.current_step: Optional[str] = None
        self.steps_completed: list[str] = []
        self.step_results: dict[str, Any] = {}
        self.errors: list[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def mark_step_complete(self, step: str, result: Any) -> None:
        self.steps_completed.append(step)
        self.step_results[step] = result

    def mark_step_failed(self, step: str, error: str) -> None:
        self.errors.append(f"{step}: {error}")


class AgentOrchestrator:
    """Orchestrator for coordinating the multi-agent screening pipeline.

    Pipeline steps:
    1. Data Fetching - Retrieve job, applications, students, resumes
    2. Screening - Evaluate candidates against job requirements
    3. Ranking - Order candidates by score
    4. Validation - Compare with historical decisions (parallel with Fairness)
    5. Fairness - Check for bias (parallel with Validation)
    6. Explanation - Generate human-readable explanations
    """

    PIPELINE_STEPS = [
        "fetch",
        "screen",
        "rank",
        "validate",
        "fairness",
        "explain",
    ]

    def __init__(
        self,
        settings: Optional[Settings] = None,
        mongo_service: Optional[MongoService] = None,
        api_client: Optional[TalentMatchClient] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize the orchestrator.

        Args:
            settings: Application settings
            mongo_service: MongoDB service
            api_client: TalentMatch API client
            embedding_service: Embedding service
        """
        self.settings = settings or get_settings()
        self.mongo_service = mongo_service or MongoService.get_instance()
        self.api_client = api_client or TalentMatchClient(self.settings)
        self.embedding_service = embedding_service or EmbeddingService.get_instance()

        # Initialize agents
        self._init_agents()

        # Active sessions
        self._sessions: dict[str, PipelineState] = {}

    def _init_agents(self) -> None:
        """Initialize all pipeline agents."""
        self.data_fetching_agent = DataFetchingAgent(
            api_client=self.api_client,
            mongo_service=self.mongo_service,
        )
        self.screening_agent = ScreeningAgent(
            embedding_service=self.embedding_service,
        )
        self.ranking_agent = RankingAgent()
        self.validation_agent = ValidationAgent(
            mongo_service=self.mongo_service,
        )
        self.fairness_agent = FairnessAgent()
        self.explanation_agent = ExplanationAgent()

    async def run_pipeline(
        self,
        job_id: str,
        options: Optional[ScreeningOptions] = None,
    ) -> ScreeningResult:
        """Run the complete screening pipeline.

        Args:
            job_id: Job listing ID to screen candidates for
            options: Screening options

        Returns:
            ScreeningResult with complete results
        """
        options = options or ScreeningOptions()
        state = PipelineState(job_id=job_id)
        state.status = PipelineStatus.RUNNING
        state.start_time = time.time()

        self._sessions[state.session_id] = state

        # Create pipeline context
        context = AgentContext(
            job_id=job_id,
            session_id=state.session_id,
        )

        try:
            # Step 1: Fetch data
            state.current_step = "fetch"
            fetch_result = await self._run_fetch_step(context, options)
            if not fetch_result.success:
                raise Exception(f"Fetch failed: {fetch_result.errors}")
            state.mark_step_complete("fetch", fetch_result.data)
            fetch_data: DataFetchingOutput = fetch_result.data

            # Step 2: Screen candidates
            state.current_step = "screen"
            screen_result = await self._run_screen_step(context, fetch_data)
            if not screen_result.success:
                raise Exception(f"Screening failed: {screen_result.errors}")
            state.mark_step_complete("screen", screen_result.data)
            screen_data: ScreeningOutput = screen_result.data

            # Step 3: Rank candidates
            state.current_step = "rank"
            rank_result = await self._run_rank_step(context, screen_data, options)
            if not rank_result.success:
                raise Exception(f"Ranking failed: {rank_result.errors}")
            state.mark_step_complete("rank", rank_result.data)
            rank_data: RankingOutput = rank_result.data

            # Steps 4 & 5: Run validation and fairness in parallel
            validation_result = None
            fairness_result = None

            if options.run_validation or options.run_fairness_check:
                state.current_step = "validate/fairness"

                tasks = []
                if options.run_validation:
                    tasks.append(
                        self._run_validation_step(context, rank_data, fetch_data)
                    )
                if options.run_fairness_check:
                    tasks.append(
                        self._run_fairness_step(context, rank_data, fetch_data, screen_data)
                    )

                results = await asyncio.gather(*tasks, return_exceptions=True)

                idx = 0
                if options.run_validation:
                    validation_result = results[idx]
                    if isinstance(validation_result, Exception):
                        logger.error(f"Validation error: {validation_result}")
                        validation_result = None
                    elif validation_result.success:
                        state.mark_step_complete("validate", validation_result.data)
                    idx += 1

                if options.run_fairness_check:
                    fairness_result = results[idx]
                    if isinstance(fairness_result, Exception):
                        logger.error(f"Fairness error: {fairness_result}")
                        fairness_result = None
                    elif fairness_result.success:
                        state.mark_step_complete("fairness", fairness_result.data)

            # Step 6: Generate explanations
            explanation_data = None
            if options.include_explanations:
                state.current_step = "explain"
                explain_result = await self._run_explanation_step(
                    context, rank_data, fetch_data, options
                )
                if explain_result.success:
                    state.mark_step_complete("explain", explain_result.data)
                    explanation_data = explain_result.data

            # Complete pipeline
            state.status = PipelineStatus.COMPLETED
            state.end_time = time.time()
            state.current_step = None

            logger.info(
                f"Pipeline completed in {state.elapsed_time:.2f}s for job {job_id}"
            )

            # Build result
            session = ScreeningSession(
                _id=state.session_id,
                job_id=job_id,
                status=state.status,
                options=options,
                steps_completed=state.steps_completed,
                total_candidates=len(fetch_data.students),
                processed_candidates=len(screen_data.scores),
                started_at=datetime.fromtimestamp(state.start_time),
                completed_at=datetime.fromtimestamp(state.end_time),
            )

            return ScreeningResult(
                session=session,
                ranking=rank_data.ranking_result,
                fairness_report=fairness_result.data.fairness_report if fairness_result and fairness_result.success else None,
                explanations=explanation_data.explanations if explanation_data else [],
                validation_agreement=validation_result.data.agreement_rate if validation_result and validation_result.success else None,
            )

        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.end_time = time.time()
            state.errors.append(str(e))
            logger.error(f"Pipeline failed: {e}")
            raise

    async def _run_fetch_step(
        self,
        context: AgentContext,
        options: ScreeningOptions,
    ) -> AgentResult[DataFetchingOutput]:
        """Run data fetching step."""
        logger.info(f"Fetching data for job {context.job_id}")

        fetch_input = DataFetchingInput(
            job_id=context.job_id,
            fetch_students=True,
            fetch_applications=True,
            fetch_resumes=True,
            sync_to_db=True,
        )

        return await self.data_fetching_agent.run(fetch_input, context)

    async def _run_screen_step(
        self,
        context: AgentContext,
        fetch_data: DataFetchingOutput,
    ) -> AgentResult[ScreeningOutput]:
        """Run screening step."""
        logger.info(f"Screening {len(fetch_data.students)} candidates")

        # Build resume lookup
        resumes = {r.student_id: r for r in fetch_data.resumes}

        screen_input = ScreeningInput(
            candidates=fetch_data.students,
            job=fetch_data.job,
            resumes=resumes,
        )

        return await self.screening_agent.run(screen_input, context)

    async def _run_rank_step(
        self,
        context: AgentContext,
        screen_data: ScreeningOutput,
        options: ScreeningOptions,
    ) -> AgentResult[RankingOutput]:
        """Run ranking step."""
        logger.info(f"Ranking {len(screen_data.scores)} candidates")

        rank_input = RankingInput(
            scores=screen_data.scores,
            top_k=options.top_k,
        )

        return await self.ranking_agent.run(rank_input, context)

    async def _run_validation_step(
        self,
        context: AgentContext,
        rank_data: RankingOutput,
        fetch_data: DataFetchingOutput,
    ) -> AgentResult[ValidationOutput]:
        """Run validation step."""
        logger.info("Running validation against historical decisions")

        validation_input = ValidationInput(
            ranked_candidates=rank_data.ranking_result.ranked_candidates,
            job_id=context.job_id,
        )

        return await self.validation_agent.run(validation_input, context)

    async def _run_fairness_step(
        self,
        context: AgentContext,
        rank_data: RankingOutput,
        fetch_data: DataFetchingOutput,
        screen_data: ScreeningOutput,
    ) -> AgentResult[FairnessOutput]:
        """Run fairness analysis step."""
        logger.info("Running fairness analysis")

        # Build scores lookup
        scores = {s.candidate_id: s for s in screen_data.scores}

        fairness_input = FairnessInput(
            ranked_candidates=rank_data.ranking_result.ranked_candidates,
            candidates=fetch_data.students,
            scores=scores,
            job_id=context.job_id,
        )

        return await self.fairness_agent.run(fairness_input, context)

    async def _run_explanation_step(
        self,
        context: AgentContext,
        rank_data: RankingOutput,
        fetch_data: DataFetchingOutput,
        options: ScreeningOptions,
    ) -> AgentResult[ExplanationOutput]:
        """Run explanation generation step."""
        logger.info(f"Generating explanations for top {options.top_k} candidates")

        # Build lookups
        candidates = {s.id: s for s in fetch_data.students}
        resumes = {r.student_id: r for r in fetch_data.resumes}

        explain_input = ExplanationInput(
            ranked_candidates=rank_data.ranking_result.ranked_candidates,
            candidates=candidates,
            resumes=resumes,
            job=fetch_data.job,
            top_k=options.top_k,
        )

        return await self.explanation_agent.run(explain_input, context)

    def get_session_status(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get status of a pipeline session.

        Args:
            session_id: Session ID

        Returns:
            Status dictionary or None if not found
        """
        state = self._sessions.get(session_id)
        if not state:
            return None

        total_steps = len(self.PIPELINE_STEPS)
        completed_steps = len(state.steps_completed)
        progress = (completed_steps / total_steps) * 100

        return {
            "session_id": state.session_id,
            "job_id": state.job_id,
            "status": state.status.value,
            "current_step": state.current_step,
            "steps_completed": state.steps_completed,
            "steps_remaining": [
                s for s in self.PIPELINE_STEPS
                if s not in state.steps_completed
            ],
            "progress_percent": progress,
            "elapsed_time_seconds": state.elapsed_time,
            "errors": state.errors,
        }

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a running pipeline session.

        Args:
            session_id: Session ID to cancel

        Returns:
            True if cancelled, False if not found or already complete
        """
        state = self._sessions.get(session_id)
        if not state:
            return False

        if state.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
            return False

        state.status = PipelineStatus.CANCELLED
        state.end_time = time.time()
        logger.info(f"Cancelled session {session_id}")
        return True
