"""Rubric Generator Service for creating job-specific evaluation criteria."""

import logging
from typing import Any, Optional

from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RubricGenerator:
    """Service for generating custom evaluation rubrics for jobs."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize rubric generator.

        Args:
            llm_service: LLM service instance
        """
        self.llm = llm_service or LLMService.get_instance()

    def generate_rubric(
        self,
        job_title: str,
        job_description: str,
        qualifications: list[str],
        responsibilities: list[str],
        num_criteria: int = 6,
    ) -> dict[str, Any]:
        """Generate a custom rubric for a specific job.

        Args:
            job_title: Job title
            job_description: Job description text
            qualifications: List of qualifications
            responsibilities: List of responsibilities
            num_criteria: Number of evaluation criteria to generate (default 6)

        Returns:
            Dictionary with generated rubric:
            {
                "criteria": [
                    {
                        "key": "domain_expertise",
                        "name": "Domain Expertise",
                        "weight": 0.25,
                        "description": "Knowledge of healthcare industry",
                        "examples": {
                            "5": "10+ years healthcare experience",
                            "3": "Some healthcare exposure",
                            "1": "No healthcare experience"
                        }
                    },
                    ...
                ],
                "total_weight": 1.0,
                "rationale": "Why these criteria are important for this role"
            }
        """
        system_prompt = """You are an expert HR consultant who designs evaluation rubrics.
Create job-specific evaluation criteria that are:
- Directly relevant to the role
- Measurable and objective
- Weighted by importance
- Include clear scoring guidelines

Respond with ONLY valid JSON, no other text."""

        # Prepare job context
        quals_text = "\n".join([f"- {q}" for q in qualifications[:10]])
        resps_text = "\n".join([f"- {r}" for r in responsibilities[:10]])

        prompt = f"""Design a custom evaluation rubric for this job position.

JOB TITLE: {job_title}

JOB DESCRIPTION: {job_description[:500]}

QUALIFICATIONS:
{quals_text}

RESPONSIBILITIES:
{resps_text}

Create {num_criteria} evaluation criteria that are SPECIFIC to this role (not generic).

For example:
- For a Healthcare Manager: "Healthcare Industry Knowledge", "Clinical Operations Experience"
- For a Software Engineer: "Backend Development Skills", "System Design Experience"
- For a Marketing Manager: "Digital Marketing Expertise", "Campaign Management"

Each criterion should:
1. Have a clear name (2-4 words)
2. Have a weight (all weights must sum to 1.0)
3. Have a description explaining what it measures
4. Have examples for scores 5, 3, and 1

Respond with ONLY this JSON format:
{{
  "criteria": [
    {{
      "key": "relevant_snake_case_name",
      "name": "Readable Name",
      "weight": 0.25,
      "description": "What this criterion measures",
      "examples": {{
        "5": "Example of exceeding this criterion",
        "3": "Example of meeting this criterion",
        "1": "Example of not meeting this criterion"
      }}
    }}
  ],
  "rationale": "Why these criteria are important for this specific role"
}}"""

        try:
            # Check if LLM is available
            if not self.llm.is_available():
                logger.warning("LLM not available, using default rubric generator")
                return self._generate_default_rubric(
                    job_title, qualifications, responsibilities, num_criteria
                )

            # Generate rubric using LLM
            response = self.llm.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7,  # Slightly higher for creativity
            )

            # Parse response
            result = self.llm._parse_json_response(response)

            # Validate rubric
            if not self._validate_rubric(result, num_criteria):
                logger.warning("Generated rubric failed validation, using fallback")
                return self._generate_default_rubric(
                    job_title, qualifications, responsibilities, num_criteria
                )

            # Normalize weights to ensure they sum to 1.0
            result = self._normalize_weights(result)

            logger.info(f"Generated custom rubric with {len(result['criteria'])} criteria")
            return result

        except Exception as e:
            logger.error(f"Rubric generation failed: {e}", exc_info=True)
            return self._generate_default_rubric(
                job_title, qualifications, responsibilities, num_criteria
            )

    def _validate_rubric(self, rubric: dict[str, Any], expected_count: int) -> bool:
        """Validate generated rubric structure.

        Args:
            rubric: Generated rubric dictionary
            expected_count: Expected number of criteria

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(rubric, dict):
            return False

        criteria = rubric.get("criteria", [])
        if not isinstance(criteria, list):
            return False

        if len(criteria) < 3 or len(criteria) > 10:
            logger.warning(f"Invalid criteria count: {len(criteria)}")
            return False

        # Validate each criterion
        for c in criteria:
            if not isinstance(c, dict):
                return False

            required_keys = ["key", "name", "weight", "description"]
            if not all(k in c for k in required_keys):
                logger.warning(f"Criterion missing required keys: {c}")
                return False

            # Validate weight is numeric
            try:
                weight = float(c["weight"])
                if weight <= 0 or weight > 1:
                    logger.warning(f"Invalid weight: {weight}")
                    return False
            except (ValueError, TypeError):
                return False

        # Check total weight (allow 5% tolerance)
        total_weight = sum(float(c["weight"]) for c in criteria)
        if abs(total_weight - 1.0) > 0.05:
            logger.warning(f"Total weight not close to 1.0: {total_weight}")
            # This is fixable, so don't fail validation
            # We'll normalize it

        return True

    def _normalize_weights(self, rubric: dict[str, Any]) -> dict[str, Any]:
        """Normalize criterion weights to sum to exactly 1.0.

        Args:
            rubric: Rubric dictionary

        Returns:
            Rubric with normalized weights
        """
        criteria = rubric.get("criteria", [])
        if not criteria:
            return rubric

        # Calculate current total
        total = sum(float(c["weight"]) for c in criteria)

        if total == 0:
            # Equal weights if all are 0
            equal_weight = 1.0 / len(criteria)
            for c in criteria:
                c["weight"] = equal_weight
        else:
            # Normalize to sum to 1.0
            for c in criteria:
                c["weight"] = float(c["weight"]) / total

        rubric["total_weight"] = 1.0
        return rubric

    def _generate_default_rubric(
        self,
        job_title: str,
        qualifications: list[str],
        responsibilities: list[str],
        num_criteria: int = 6,
    ) -> dict[str, Any]:
        """Generate a default rubric when LLM is unavailable.

        Args:
            job_title: Job title
            qualifications: List of qualifications
            responsibilities: List of responsibilities
            num_criteria: Number of criteria

        Returns:
            Default rubric dictionary
        """
        # Analyze job to determine focus areas
        job_lower = job_title.lower()
        all_text = (job_title + " " + " ".join(qualifications) + " " + " ".join(responsibilities)).lower()

        criteria = []

        # Education (always included)
        criteria.append({
            "key": "education",
            "name": "Education & Qualifications",
            "weight": 0.15,
            "description": "Academic background and certifications",
            "examples": {
                "5": "Advanced degree in relevant field",
                "3": "Bachelor's degree in relevant field",
                "1": "No relevant formal education"
            }
        })

        # Experience (always included)
        criteria.append({
            "key": "experience",
            "name": "Relevant Experience",
            "weight": 0.25,
            "description": "Years of experience in similar roles",
            "examples": {
                "5": "8+ years in similar role",
                "3": "3-5 years in related role",
                "1": "No relevant experience"
            }
        })

        # Technical/Job-specific skills
        if any(word in all_text for word in ["software", "developer", "engineer", "technical", "programming"]):
            criteria.append({
                "key": "technical_skills",
                "name": "Technical Skills",
                "weight": 0.30,
                "description": "Proficiency in required technical tools and technologies",
                "examples": {
                    "5": "Expert in all required technologies",
                    "3": "Proficient in most required technologies",
                    "1": "Limited technical skills"
                }
            })
        elif any(word in all_text for word in ["marketing", "sales", "business"]):
            criteria.append({
                "key": "business_skills",
                "name": "Business & Marketing Skills",
                "weight": 0.30,
                "description": "Business acumen and marketing expertise",
                "examples": {
                    "5": "Proven track record of successful campaigns",
                    "3": "Some marketing project experience",
                    "1": "No marketing experience"
                }
            })
        else:
            criteria.append({
                "key": "domain_skills",
                "name": "Domain-Specific Skills",
                "weight": 0.30,
                "description": "Skills specific to this industry/role",
                "examples": {
                    "5": "Expert in domain",
                    "3": "Competent in domain",
                    "1": "Limited domain knowledge"
                }
            })

        # Leadership (if manager/senior role)
        if any(word in all_text for word in ["manager", "lead", "director", "senior", "head", "supervisor"]):
            criteria.append({
                "key": "leadership",
                "name": "Leadership Experience",
                "weight": 0.20,
                "description": "Team management and leadership abilities",
                "examples": {
                    "5": "Led large teams (10+ people)",
                    "3": "Led small teams or projects",
                    "1": "No leadership experience"
                }
            })
        else:
            criteria.append({
                "key": "collaboration",
                "name": "Collaboration Skills",
                "weight": 0.20,
                "description": "Ability to work in teams",
                "examples": {
                    "5": "Extensive team project experience",
                    "3": "Some collaborative work",
                    "1": "Limited teamwork experience"
                }
            })

        # Communication (always important)
        criteria.append({
            "key": "communication",
            "name": "Communication Skills",
            "weight": 0.10,
            "description": "Written and verbal communication abilities",
            "examples": {
                "5": "Exceptional communication, well-written materials",
                "3": "Clear communication",
                "1": "Poor communication skills"
            }
        })

        # Trim to requested number or pad if needed
        if len(criteria) > num_criteria:
            criteria = criteria[:num_criteria]

        # Normalize weights
        result = {
            "criteria": criteria,
            "rationale": f"Default evaluation criteria for {job_title} role",
            "total_weight": 1.0
        }

        return self._normalize_weights(result)