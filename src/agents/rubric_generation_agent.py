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
import re
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
        job: The job listing (used for metadata like title/company).
        raw_job: The raw MongoDB document for the job. Preferred over ``job``
            for prompt construction because it contains the actual field names
            used by TalentMatch (e.g. ``qualifications``, ``responsibilities``).
        base_rubric: Optional list of criteria to start from; defaults to the
            generic rubric defined in ``DefaultCriteria``.
    """

    def __init__(
        self,
        job: JobListing,
        raw_job: Optional[dict] = None,
        base_rubric: Optional[list[EvaluationCriterion]] = None,
    ):
        self.job = job
        self.raw_job = raw_job or {}
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
            criteria = self._tailor_rubric(input_data.job, input_data.base_rubric, input_data.raw_job)
            output = RubricGenerationOutput(criteria=criteria)
            return AgentResult.success_result(output, self.name)
        except Exception as e:  # pragma: no cover - log and return failure
            logger.error(f"Rubric generation error: {e}")
            return AgentResult.failure_result([str(e)], self.name)

    def _tailor_rubric(
        self,
        job: JobListing,
        base_rubric: list[EvaluationCriterion],
        raw_job: Optional[dict] = None,
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

        # Extract job fields — prefer raw MongoDB doc (has actual field names from TalentMatch)
        # over the JobListing model (which may have mismatched field names / empty lists)
        d = raw_job or {}
        title = d.get("title") or job.title

        # TalentMatch uses "qualifications"; JobListing model uses "requirements"
        qualifications = (
            d.get("qualifications")
            or d.get("requirements")
            or job.requirements
            or []
        )
        responsibilities = d.get("responsibilities") or []
        required_skills = d.get("required_skills") or job.required_skills or []
        preferred_qualifications = (
            d.get("preferred_qualifications") or job.preferred_qualifications or []
        )
        description = d.get("description") or job.description or ""

        if not any([qualifications, responsibilities, required_skills, preferred_qualifications]):
            logger.warning(
                f"Job '{title}' has no qualifications, responsibilities, or skills — "
                "rubric will only vary by title. Check TalentMatch field names."
            )

        # Build textual representation of the generic rubric
        rubric_lines = [
            f"- {crit.name} ({crit.weight_percentage}%): {crit.description}"
            for crit in base_rubric
        ]

        sections = [
            f"JOB TITLE: {title}",
        ]
        if description:
            sections.append(f"JOB DESCRIPTION:\n{description[:500]}")
        if qualifications:
            sections.append("REQUIRED QUALIFICATIONS:\n- " + "\n- ".join(qualifications))
        if responsibilities:
            sections.append("KEY RESPONSIBILITIES:\n- " + "\n- ".join(responsibilities))
        if required_skills:
            sections.append("REQUIRED SKILLS:\n- " + "\n- ".join(required_skills))
        if preferred_qualifications:
            sections.append("PREFERRED QUALIFICATIONS:\n- " + "\n- ".join(preferred_qualifications))

        job_context = "\n\n".join(sections)

        prompt = (
    "You are an expert recruiter and talent evaluator. Your task is to adapt a standard "
    "evaluation rubric to fit a specific job posting.\n\n"
    "STANDARD RUBRIC (your starting point):\n"
    + "\n".join(rubric_lines)
    + "\n\nADAPTATION RULES — follow these strictly:\n"
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
    "5. Return ONLY valid JSON — a list of objects with keys: "
    "'name', 'key', 'weight' (decimal 0–1), 'description'. No explanation, no markdown.\n\n"
    + job_context
    + "\n\nRemember: start from the standard rubric and adapt — do not invent a rubric from scratch."
        )

        system_prompt = (
            "You are a helpful assistant specialized in HR and hiring."
        )

        try:
            response = self.llm_service.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.2,
            )
            logger.info(f"LLM rubric response received, length: {len(response)}")
            if not response.strip():
                raise ValueError("Empty response from LLM")

            parsed = self._parse_criteria_json(response)
            if not parsed:
                raise ValueError("No criteria parsed from LLM response")

            tailored: list[EvaluationCriterion] = []
            for obj in parsed:
                tailored.append(EvaluationCriterion(**obj))
        except Exception as e:
            logger.warning(f"LLM rubric generation failed ({e}), using base rubric")
            return base_rubric

        return self._normalize_weights(tailored)

    def _parse_criteria_json(self, response: str) -> list[dict]:
        """Extract a JSON array of criteria from LLM output.

        Handles markdown fences, leading/trailing prose, and trailing commas
        that LLMs commonly produce.
        """
        text = response.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Remove trailing commas before ] or } (common LLM mistake)
        text = re.sub(r',\s*]', ']', text)
        text = re.sub(r',\s*}', '}', text)

        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract the JSON array from anywhere in the text
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not extract JSON array from LLM response: {text[:300]}")
        return []

    def _normalize_weights(self, criteria: list[EvaluationCriterion]) -> list[EvaluationCriterion]:
        """Normalize weights so they sum to exactly 1.0."""
        total_weight = sum(c.weight for c in criteria)
        if abs(total_weight - 1.0) > 0.01 and total_weight > 0:
            logger.warning("Rubric weights do not sum to 1, normalizing")
            for c in criteria:
                c.weight = c.weight / total_weight
        return criteria
