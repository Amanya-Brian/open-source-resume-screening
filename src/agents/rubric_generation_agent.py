"""Agent for generating a tailored scoring rubric based on a generic template.

This agent starts from the default evaluation criteria defined in
``src.models.scoring.DefaultCriteria`` (the generic rubric shown in the
requirements document) and uses the job listing details to adjust weights and
descriptions.  It leverages the LLM service when available to make the rubric
more specific to the position, but will gracefully fall back to the base
rubric if the model is unavailable or returns invalid output.

The generated rubric is represented as a list of
``EvaluationCriterion`` instances and may be used by other components (for
example, during candidate evaluation or when presenting the rubric to a human
reviewer).
"""

import json
import logging
from typing import Any, Optional

from src.agents.base import AgentConfig, AgentContext, AgentResult, BaseAgent
from src.models.scoring import (
    DefaultCriteria,
    EvaluationCriterion,
)
from src.models.schemas import JobListing

logger = logging.getLogger(__name__)


class RubricGenerationInput:
    """Input for the rubric generation agent.

    Attributes:
        job: The job listing to tailor the rubric for.
        base_rubric: Optional list of criteria to start from; defaults to the
            generic rubric defined in ``DefaultCriteria``.
    """

    def __init__(
        self,
        job: JobListing,
        base_rubric: Optional[list[EvaluationCriterion]] = None,
    ):
        self.job = job
        self.base_rubric = base_rubric or DefaultCriteria.get_all()


class RubricGenerationOutput:
    """Output produced by the rubric generation agent."""

    def __init__(self, criteria: list[EvaluationCriterion]):
        self.criteria = criteria


class RubricGenerationAgent(BaseAgent[RubricGenerationInput, RubricGenerationOutput]):
    """Agent responsible for creating a tailored evaluation rubric.

    The agent uses the LLM service to adjust the generic rubric weights and
    descriptions based on the provided job listing.  If the LLM is disabled or
    fails, the agent simply returns the unmodified base rubric.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_service: Optional[Any] = None,
    ):
        super().__init__(config or AgentConfig(name="rubric_generation"))
        # Note: type for llm_service is Any to avoid circular import in typing
        # but at runtime we expect an instance of src.services.llm_service.LLMService
        from src.services.llm_service import LLMService

        self.llm_service = llm_service or LLMService.get_instance()

    async def execute(
        self,
        input_data: RubricGenerationInput,
        context: AgentContext,
    ) -> AgentResult[RubricGenerationOutput]:
        """Generate or tailor a rubric based on job details.

        Args:
            input_data: Contains the job listing and optional base rubric.
            context: Pipeline context (unused currently).

        Returns:
            AgentResult wrapping ``RubricGenerationOutput`` with the list of
            evaluation criteria.
        """
        try:
            criteria = self._tailor_rubric(input_data.job, input_data.base_rubric)
            output = RubricGenerationOutput(criteria=criteria)
            return AgentResult.success_result(output, self.name)
        except Exception as e:  # pragma: no cover - log and return failure
            logger.error(f"Rubric generation error: {e}")
            return AgentResult.failure_result([str(e)], self.name)

    def _tailor_rubric(
        self,
        job: JobListing,
        base_rubric: list[EvaluationCriterion],
    ) -> list[EvaluationCriterion]:
        """Attempt to adjust the base rubric using the LLM.

        The LLM prompt provides the generic rubric and the job's title,
        requirements and skills.  The model is expected to return a JSON list of
        objects with the same fields as ``EvaluationCriterion``.  We validate
        the output and normalize weights if necessary.
        """

        # Ensure LLM is initialized before use
        try:
            self.llm_service.initialize()
            # Check if the selected model is available, if not, try preferred models
            if self.llm_service.model_name not in self.llm_service._available_models and self.llm_service._available_models:
                # Try mistral first, then meditron, then any available
                preferred_order = ['mistral:latest', 'meditron:latest'] + self.llm_service._available_models
                for model in preferred_order:
                    if model in self.llm_service._available_models:
                        logger.warning(f"Configured model '{self.llm_service.model_name}' not available. Using '{model}' instead.")
                        self.llm_service.model_name = model
                        break
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}, using base rubric")
            return base_rubric

        # Build textual representation of the generic rubric
        rubric_lines = []
        for crit in base_rubric:
            rubric_lines.append(f"- {crit.name} ({crit.weight_percentage}%): {crit.description}")

        prompt = (
    "You are an expert recruiter and talent evaluator. Your task is to adapt a standard "
    "evaluation rubric to fit a specific job posting.\n\n"
    "STANDARD RUBRIC (your starting point):\n"
     + "\n".join(rubric_lines)+
    "\n\nADAPTATION RULES — follow these strictly:\n"
    "1. KEEP all standard criteria unless one is clearly irrelevant to this role (e.g. remove "
    "'Leadership Experience' only if the role has zero supervisory responsibilities).\n"
    "2. REWRITE each kept criterion's 'description' to be role-specific — reference actual "
    "tools, certifications, regulations, or responsibilities from the job posting. "
    "Never keep a generic description like 'Years of relevant work experience in the field'.\n"
    "3. ADD at most 1–2 new criteria only if the job posting has a major theme not covered "
    "by any standard criterion (e.g. 'Laboratory Safety' for a lab role, 'Regulatory Compliance' "
    "for a finance role). Do NOT add criteria that overlap with existing ones.\n"
    "4. REBALANCE weights to reflect how critical each criterion is for THIS role specifically. "
    "Weights must sum exactly to 1.0.\n"
    "5. Remove 'Relevant Experience (Years)' only if years of experience are NOT explicitly "
    "stated anywhere in the requirements.\n"
    "6. Return ONLY valid JSON — a list of objects with keys: "
    "'name', 'key', 'weight' (decimal 0–1), 'description'. No explanation, no markdown.\n\n"
    f"JOB TITLE: {job.title}\n\n"
    "REQUIREMENTS:\n- " + "\n- ".join(job.requirements or [])
    + (
        "\n\nPREFERRED QUALIFICATIONS:\n- " + "\n- ".join(job.preferred_qualifications or [])
        if job.preferred_qualifications else ""
    )
    + (
        "\n\nREQUIRED SKILLS:\n- " + "\n- ".join(job.required_skills or [])
        if job.required_skills else ""
    )
    + (
        "\n\nKEY RESPONSIBILITIES:\n- " + "\n- ".join(getattr(job, 'responsibilities', []) or [])
        if getattr(job, 'responsibilities', None) else ""
    )
    + "\n\nRemember: start from the standard rubric and adapt — do not invent a rubric from scratch.")

        system_prompt = (
            "You are a helpful assistant specialized in HR and hiring."
        )

        try:
            response = self.llm_service.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.2,
            )
            logger.info(f"LLM response received, length: {len(response)}")
            if not response.strip():
                raise ValueError("Empty response from LLM")
            parsed = json.loads(response)
            tailored: list[EvaluationCriterion] = []
            for obj in parsed:
                # create EvaluationCriterion; this will validate the fields
                tailored.append(EvaluationCriterion(**obj))
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {response[:200]}... Error: {e}")
            return base_rubric
        except Exception as e:
            logger.warning(f"LLM rubric generation failed ({e}), using base rubric")
            return base_rubric

        # normalize weights if numerical issues occur
        total_weight = sum(c.weight for c in tailored)
        if abs(total_weight - 1.0) > 0.01 and total_weight > 0:
            logger.warning("Rubric weights do not sum to 1, normalizing")
            for c in tailored:
                c.weight = c.weight / total_weight

        return tailored
