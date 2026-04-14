"""Screening Agent for evaluating candidates against job requirements."""

import logging
from typing import Any, Optional

from src.agents.base import AgentConfig, AgentContext, AgentResult, BaseAgent
from src.models.schemas import (
    ComponentScore,
    JobListing,
    ParsedResume,
    Resume,
    ScreeningScore,
    Student,
)
from src.services.embedding_service import EmbeddingService
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class ScreeningInput:
    """Input for Screening Agent."""

    def __init__(
        self,
        candidates: list[Student],
        job: JobListing,
        resumes: dict[str, Resume],  # student_id -> Resume
        rubric: dict = None,
    ):
        self.candidates = candidates
        self.job = job
        self.resumes = resumes
        self.rubric = rubric


class ScreeningOutput:
    """Output from Screening Agent."""

    def __init__(
        self,
        scores: list[ScreeningScore],
        filtered_candidates: list[str],  # IDs of candidates passing threshold
    ):
        self.scores = scores
        self.filtered_candidates = filtered_candidates

    @property
    def candidate_count(self) -> int:
        return len(self.scores)


class ScreeningAgent(BaseAgent[ScreeningInput, ScreeningOutput]):
    """Agent responsible for screening candidates against job requirements.

    This agent:
    - Extracts skills and experience from resumes
    - Computes semantic similarity between candidate profile and job
    - Calculates component scores for different criteria
    - Filters candidates below minimum threshold
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize the Screening Agent.

        Args:
            config: Agent configuration
            embedding_service: Service for computing embeddings
        """
        super().__init__(config or AgentConfig(name="screening"))
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.min_score_threshold = self.settings.screening_min_score_threshold

    async def execute(
        self,
        input_data: ScreeningInput,
        context: AgentContext,
    ) -> AgentResult[ScreeningOutput]:
        """Execute candidate screening.

        Args:
            input_data: Candidates, job, and resumes to screen
            context: Pipeline context

        Returns:
            AgentResult with screening scores
        """
        scores = []
        filtered_candidates = []

        try:
            # Process each candidate
            for candidate in input_data.candidates:
                resume = input_data.resumes.get(candidate.id)

                score = await self._screen_candidate(
                    candidate=candidate,
                    job=input_data.job,
                    resume=resume,
                    rubric=input_data.rubric,
                )

                scores.append(score)

                # Check if passes threshold
                if score.overall_score >= self.min_score_threshold:
                    filtered_candidates.append(candidate.id)

                # Store in context
                context.scores[candidate.id] = score

            logger.info(
                f"Screened {len(scores)} candidates, "
                f"{len(filtered_candidates)} passed threshold"
            )

            output = ScreeningOutput(
                scores=scores,
                filtered_candidates=filtered_candidates,
            )

            return AgentResult.success_result(output, self.name)

        except Exception as e:
            logger.error(f"Screening error: {e}")
            return AgentResult.failure_result([str(e)], self.name)

    async def _screen_candidate(
        self,
        candidate: Student,
        job: JobListing,
        resume: Optional[Resume],
        rubric: Optional[dict] = None,
    ) -> ScreeningScore:
        resume_text = ""
        parsed_resume = ParsedResume()
        if resume:
            resume_text = resume.raw_text
            parsed_resume = resume.parsed_data

        # Use rubric criteria + LLM scoring
        if rubric and rubric.get("criteria"):
            return await self._screen_with_rubric(
                candidate, job, resume_text, parsed_resume, rubric
            )

        # Fallback: rule-based scoring with default weights
        return await self._screen_with_defaults(
            candidate, job, resume_text, parsed_resume
        )

    async def _screen_with_rubric(
        self,
        candidate: Student,
        job: JobListing,
        resume_text: str,
        parsed_resume: ParsedResume,
        rubric: dict,
    ) -> ScreeningScore:
        """Score candidate using the generated rubric via LLM."""

        # Build candidate text
        candidate_text = resume_text or ""
        if parsed_resume.skills:
            candidate_text += "\nSkills: " + ", ".join(parsed_resume.skills)
        if not candidate_text.strip():
            candidate_text = f"{candidate.first_name} {candidate.last_name} — no resume text available"

        # Map rubric criteria to LLM service format (needs "key" field)
        criteria = [
            {
                "key":         c.get("id", f"C{i}"),
                "name":        c.get("name", ""),
                "weight":      float(c.get("weight", 0)),
                "description": c.get("description", ""),
            }
            for i, c in enumerate(rubric["criteria"])
        ]

        job_requirements = {
            "qualifications":  job.requirements or [],
            "responsibilities": [job.description] if job.description else [],
        }

        # Call LLM — runs synchronously, offload to thread
        import asyncio
        llm = LLMService.get_instance()
        loop = asyncio.get_event_loop()
        llm_result = await loop.run_in_executor(
            None,
            lambda: llm.evaluate_candidate(candidate_text, job_requirements, criteria)
        )

        # Map LLM scores (0–5) to ComponentScore (0–1)
        scores_by_key = {s["criterion"]: s for s in llm_result.get("scores", [])}
        component_scores = []
        for c in criteria:
            raw = scores_by_key.get(c["key"], {})
            score_0_to_1 = round(float(raw.get("score", 0)) / 5.0, 4)
            component_scores.append(ComponentScore(
                name=c["key"],
                score=score_0_to_1,
                weight=c["weight"],
                weighted_score=round(score_0_to_1 * c["weight"], 4),
                details=raw.get("evidence", ""),
            ))

        overall_score = min(1.0, sum(cs.weighted_score for cs in component_scores))

        return ScreeningScore(
            candidate_id=candidate.id,
            job_id=job.id,
            overall_score=overall_score,
            component_scores=component_scores,
            matching_skills=[],
            missing_skills=[],
            experience_match=0.0,
            education_match=0.0,
        )

    async def _screen_with_defaults(
        self,
        candidate: Student,
        job: JobListing,
        resume_text: str,
        parsed_resume: ParsedResume,
    ) -> ScreeningScore:
        """Fallback: rule-based scoring with default weights from settings."""
        component_scores = []

        skills_score, matching_skills, missing_skills = self._compute_skills_match(
            candidate_skills=candidate.skills + parsed_resume.skills,
            required_skills=job.required_skills,
        )
        component_scores.append(ComponentScore(
            name="skills_match",
            score=skills_score,
            weight=self.settings.weight_skills_match,
            weighted_score=skills_score * self.settings.weight_skills_match,
            details=f"Matched {len(matching_skills)}/{len(job.required_skills)} skills",
        ))

        experience_score = self._compute_experience_match(parsed_resume=parsed_resume, job=job)
        component_scores.append(ComponentScore(
            name="experience",
            score=experience_score,
            weight=self.settings.weight_experience,
            weighted_score=experience_score * self.settings.weight_experience,
            details=f"{parsed_resume.total_experience_years:.1f} years experience",
        ))

        education_score = self._compute_education_match(candidate=candidate, parsed_resume=parsed_resume, job=job)
        component_scores.append(ComponentScore(
            name="education",
            score=education_score,
            weight=self.settings.weight_education,
            weighted_score=education_score * self.settings.weight_education,
        ))

        projects_score = await self._compute_projects_relevance(parsed_resume=parsed_resume, job=job)
        component_scores.append(ComponentScore(
            name="projects",
            score=projects_score,
            weight=self.settings.weight_projects,
            weighted_score=projects_score * self.settings.weight_projects,
        ))

        certifications_score = self._compute_certifications_score(parsed_resume=parsed_resume, job=job)
        component_scores.append(ComponentScore(
            name="certifications",
            score=certifications_score,
            weight=self.settings.weight_certifications,
            weighted_score=certifications_score * self.settings.weight_certifications,
        ))

        relevance_score = await self._compute_semantic_relevance(resume_text=resume_text, job_description=job.description)
        component_scores.append(ComponentScore(
            name="semantic_relevance",
            score=relevance_score,
            weight=self.settings.weight_soft_skills,
            weighted_score=relevance_score * self.settings.weight_soft_skills,
        ))

        overall_score = sum(cs.weighted_score for cs in component_scores)

        return ScreeningScore(
            candidate_id=candidate.id,
            job_id=job.id,
            overall_score=min(1.0, overall_score),
            component_scores=component_scores,
            matching_skills=matching_skills,
            missing_skills=missing_skills,
            experience_match=experience_score,
            education_match=education_score,
        )

    def _compute_skills_match(
        self,
        candidate_skills: list[str],
        required_skills: list[str],
    ) -> tuple[float, list[str], list[str]]:
        """Compute skills match score.

        Returns:
            Tuple of (score, matching_skills, missing_skills)
        """
        if not required_skills:
            return 1.0, candidate_skills, []

        if not candidate_skills:
            return 0.0, [], required_skills

        # Use embedding service for semantic matching
        score, matching, missing = self.embedding_service.compute_skill_match_score(
            resume_skills=candidate_skills,
            required_skills=required_skills,
        )

        return score, matching, missing

    def _compute_experience_match(
        self,
        parsed_resume: ParsedResume,
        job: JobListing,
    ) -> float:
        """Compute experience match score."""
        years = parsed_resume.total_experience_years

        min_years = job.experience_years_min or 0
        max_years = job.experience_years_max or 10

        if years < min_years:
            # Partial score for less experience
            return max(0.3, years / max(min_years, 1))
        elif years > max_years:
            # Slight penalty for overqualification
            return 0.9
        else:
            # Perfect match range
            return 1.0

    def _compute_education_match(
        self,
        candidate: Student,
        parsed_resume: ParsedResume,
        job: JobListing,
    ) -> float:
        """Compute education match score."""
        score = 0.5  # Base score

        # Check GPA
        if candidate.gpa:
            if candidate.gpa >= 3.5:
                score += 0.3
            elif candidate.gpa >= 3.0:
                score += 0.2
            elif candidate.gpa >= 2.5:
                score += 0.1

        # Check for relevant education
        education = parsed_resume.education
        if education:
            # Having higher education
            for edu in education:
                degree_lower = edu.degree.lower()
                if "master" in degree_lower or "mba" in degree_lower or "phd" in degree_lower:
                    score += 0.2
                    break
                elif "bachelor" in degree_lower or "bs" in degree_lower or "ba" in degree_lower:
                    score += 0.1
                    break

        return min(1.0, score)

    async def _compute_projects_relevance(
        self,
        parsed_resume: ParsedResume,
        job: JobListing,
    ) -> float:
        """Compute projects relevance score."""
        if not parsed_resume.projects:
            return 0.3  # Base score for no projects

        # Combine project descriptions
        project_texts = [
            f"{p.name} {p.description} {' '.join(p.technologies)}"
            for p in parsed_resume.projects
        ]

        if not project_texts:
            return 0.3

        # Compute relevance to job
        combined_projects = " ".join(project_texts)
        job_text = f"{job.title} {job.description} {' '.join(job.required_skills)}"

        relevance = self.embedding_service.compute_text_relevance(
            source_text=combined_projects,
            target_text=job_text,
        )

        return relevance

    def _compute_certifications_score(
        self,
        parsed_resume: ParsedResume,
        job: JobListing,
    ) -> float:
        """Compute certifications score."""
        if not parsed_resume.certifications:
            return 0.5  # Neutral for no certifications

        # More certifications = higher score
        cert_count = len(parsed_resume.certifications)

        if cert_count >= 5:
            return 1.0
        elif cert_count >= 3:
            return 0.8
        elif cert_count >= 1:
            return 0.6

        return 0.5

    async def _compute_semantic_relevance(
        self,
        resume_text: str,
        job_description: str,
    ) -> float:
        """Compute semantic relevance between resume and job."""
        if not resume_text:
            return 0.3

        return self.embedding_service.compute_text_relevance(
            source_text=resume_text,
            target_text=job_description,
        )

    def validate_input(self, data: ScreeningInput) -> bool:
        """Validate screening input."""
        if not data.candidates:
            logger.warning("No candidates to screen")
            return False
        if not data.job:
            logger.error("No job provided for screening")
            return False
        return True
