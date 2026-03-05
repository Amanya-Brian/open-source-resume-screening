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
            "You are an expert recruiter. Here is a generic candidate evaluation "
            "rubric with criteria and their weights:\n"
            + "\n".join(rubric_lines)
            + "\n\nThe following job posting requires you to adjust the rubric where necessary so that "
            "the weights and descriptions reflect the priorities for this role. "
            "Remove the relevant experience (Years) criterion unless the job explicitly requires it."
            # "You may also add new criteria that are important for this position "
            # "even if they were not in the generic rubric."
            "Return ONLY valid JSON: a list of objects with keys 'name', 'key', "
            "'weight' (decimal between 0 and 1) and 'description'. Ensure the "
            "weights sum to 1.0.\n\n"
            f"JOB TITLE: {job.title}\n"
            f"REQUIREMENTS:\n- "
            + "\n- ".join(job.requirements or [])
            + ("\nPREFERRED QUALIFICATIONS:\n- " + "\n- ".join(job.preferred_qualifications or []) if job.preferred_qualifications else "")
            + ("\nSKILLS:\n- " + "\n- ".join(job.required_skills or []) if job.required_skills else "")
            + ("\nRESPONSIBILITIES:\n- " + "\n- ".join(getattr(job, 'responsibilities', []) or []) if getattr(job, 'responsibilities', None) else ""))

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
