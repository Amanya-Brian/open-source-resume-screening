"""Explanation Agent for generating human-readable explanations."""

import logging
from typing import Any, Optional

from src.agents.base import AgentConfig, AgentContext, AgentResult, BaseAgent
from src.models.schemas import (
    CandidateExplanation,
    GapItem,
    JobListing,
    RankedCandidate,
    Recommendation,
    Resume,
    ScreeningScore,
    StrengthItem,
    Student,
)

logger = logging.getLogger(__name__)


class ExplanationInput:
    """Input for Explanation Agent."""

    def __init__(
        self,
        ranked_candidates: list[RankedCandidate],
        candidates: dict[str, Student],
        resumes: dict[str, Resume],
        job: JobListing,
        top_k: int = 10,
    ):
        self.ranked_candidates = ranked_candidates
        self.candidates = candidates
        self.resumes = resumes
        self.job = job
        self.top_k = top_k


class ExplanationOutput:
    """Output from Explanation Agent."""

    def __init__(
        self,
        explanations: list[CandidateExplanation],
    ):
        self.explanations = explanations

    @property
    def explanation_count(self) -> int:
        return len(self.explanations)


class ExplanationAgent(BaseAgent[ExplanationInput, ExplanationOutput]):
    """Agent responsible for generating human-readable explanations.

    This agent:
    - Generates structured explanations for each candidate
    - Identifies strengths and gaps
    - Provides actionable recommendations
    - Ensures 100% structured output
    """

    # Templates for different recommendation levels
    SUMMARY_TEMPLATES = {
        Recommendation.HIGHLY_RECOMMENDED: (
            "This candidate demonstrates exceptional alignment with the {job_title} role, "
            "with strong matches in {strength_areas}. Their {experience_summary} "
            "makes them a top contender for this position."
        ),
        Recommendation.RECOMMENDED: (
            "This candidate shows solid qualifications for the {job_title} position, "
            "particularly in {strength_areas}. While there are some gaps in {gap_areas}, "
            "their overall profile aligns well with requirements."
        ),
        Recommendation.CONSIDER: (
            "This candidate has partial alignment with the {job_title} requirements. "
            "Strengths in {strength_areas} are notable, but gaps exist in {gap_areas} "
            "that would require development or training."
        ),
        Recommendation.NOT_RECOMMENDED: (
            "This candidate does not currently meet the core requirements for the "
            "{job_title} role. Key gaps in {gap_areas} would significantly impact "
            "their ability to perform essential job functions."
        ),
    }

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        use_llm: bool = False,
    ):
        """Initialize the Explanation Agent.

        Args:
            config: Agent configuration
            use_llm: Whether to use LLM for generation (future enhancement)
        """
        super().__init__(config or AgentConfig(name="explanation"))
        self.use_llm = use_llm
        # LLM integration can be added here

    async def execute(
        self,
        input_data: ExplanationInput,
        context: AgentContext,
    ) -> AgentResult[ExplanationOutput]:
        """Execute explanation generation.

        Args:
            input_data: Candidates and ranking data
            context: Pipeline context

        Returns:
            AgentResult with explanations
        """
        try:
            explanations = []

            # Generate explanations for top K candidates
            candidates_to_explain = input_data.ranked_candidates[:input_data.top_k]

            for ranked in candidates_to_explain:
                candidate = input_data.candidates.get(ranked.candidate_id)
                resume = input_data.resumes.get(ranked.candidate_id)

                if not candidate:
                    logger.warning(f"Candidate not found: {ranked.candidate_id}")
                    continue

                explanation = await self._generate_explanation(
                    ranked=ranked,
                    candidate=candidate,
                    resume=resume,
                    job=input_data.job,
                )

                explanations.append(explanation)

            # Validate all explanations are complete
            for exp in explanations:
                self._validate_and_complete(exp)

            logger.info(f"Generated {len(explanations)} explanations")

            return AgentResult.success_result(
                ExplanationOutput(explanations),
                self.name,
            )

        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return AgentResult.failure_result([str(e)], self.name)

    async def _generate_explanation(
        self,
        ranked: RankedCandidate,
        candidate: Student,
        resume: Optional[Resume],
        job: JobListing,
    ) -> CandidateExplanation:
        """Generate explanation for a single candidate.

        Args:
            ranked: Ranked candidate info
            candidate: Candidate profile
            resume: Candidate's resume
            job: Job listing

        Returns:
            CandidateExplanation
        """
        score = ranked.screening_score

        # Identify strengths
        strengths = self._identify_strengths(score, candidate, resume)

        # Identify gaps
        gaps = self._identify_gaps(score, job)

        # Generate summary
        summary = self._generate_summary(
            ranked=ranked,
            strengths=strengths,
            gaps=gaps,
            job=job,
        )

        # Generate experience summary
        experience_summary = self._generate_experience_summary(candidate, resume)

        # Generate education summary
        education_summary = self._generate_education_summary(candidate, resume)

        return CandidateExplanation(
            candidate_id=candidate.id,
            job_id=job.id,
            rank=ranked.rank,
            overall_score=ranked.score,
            recommendation=ranked.recommendation,
            summary=summary,
            strengths=strengths,
            gaps=gaps,
            matching_skills=score.matching_skills,
            experience_summary=experience_summary,
            education_summary=education_summary,
            confidence_score=score.confidence_score,
        )

    def _identify_strengths(
        self,
        score: ScreeningScore,
        candidate: Student,
        resume: Optional[Resume],
    ) -> list[StrengthItem]:
        """Identify candidate strengths.

        Args:
            score: Screening score
            candidate: Candidate profile
            resume: Resume data

        Returns:
            List of strength items
        """
        strengths = []

        # Skills strength
        if score.matching_skills:
            skill_score = len(score.matching_skills) / max(1, len(score.matching_skills) + len(score.missing_skills))
            if skill_score >= 0.5:
                strengths.append(StrengthItem(
                    category="skills",
                    description=f"Strong technical skill match with {len(score.matching_skills)} relevant skills including {', '.join(score.matching_skills[:3])}",
                    impact_score=skill_score,
                    evidence=", ".join(score.matching_skills),
                ))

        # Experience strength
        if score.experience_match >= 0.7:
            strengths.append(StrengthItem(
                category="experience",
                description="Demonstrates relevant work experience aligned with role requirements",
                impact_score=score.experience_match,
            ))

        # Education strength
        if score.education_match >= 0.7:
            education_desc = f"Strong educational background"
            if candidate.university:
                education_desc += f" from {candidate.university}"
            if candidate.gpa and candidate.gpa >= 3.5:
                education_desc += f" with excellent academic performance (GPA: {candidate.gpa})"

            strengths.append(StrengthItem(
                category="education",
                description=education_desc,
                impact_score=score.education_match,
            ))

        # Component-based strengths
        for component in score.component_scores:
            if component.score >= 0.8 and component.name not in ["skills_match", "experience", "education"]:
                strengths.append(StrengthItem(
                    category=component.name,
                    description=component.details or f"Strong performance in {component.name}",
                    impact_score=component.score,
                ))

        # Ensure at least one strength
        if not strengths:
            strengths.append(StrengthItem(
                category="general",
                description="Candidate meets basic requirements for the position",
                impact_score=0.5,
            ))

        return strengths[:5]  # Limit to top 5

    def _identify_gaps(
        self,
        score: ScreeningScore,
        job: JobListing,
    ) -> list[GapItem]:
        """Identify candidate gaps.

        Args:
            score: Screening score
            job: Job listing

        Returns:
            List of gap items
        """
        gaps = []

        # Missing skills
        for skill in score.missing_skills[:5]:
            severity = "high" if skill in job.required_skills else "medium"
            gaps.append(GapItem(
                category="skills",
                required=skill,
                candidate_status="missing",
                suggestion=f"Consider training or certification in {skill}",
                severity=severity,
            ))

        # Experience gap
        if score.experience_match < 0.5:
            gaps.append(GapItem(
                category="experience",
                required="Relevant industry experience",
                candidate_status="partial" if score.experience_match > 0.2 else "missing",
                suggestion="Candidate may benefit from mentorship or extended onboarding",
                severity="medium",
            ))

        # Education gap
        if score.education_match < 0.5:
            gaps.append(GapItem(
                category="education",
                required="Required educational qualifications",
                candidate_status="partial",
                suggestion="Consider equivalent experience or additional certifications",
                severity="low",
            ))

        return gaps

    def _generate_summary(
        self,
        ranked: RankedCandidate,
        strengths: list[StrengthItem],
        gaps: list[GapItem],
        job: JobListing,
    ) -> str:
        """Generate summary text.

        Args:
            ranked: Ranked candidate
            strengths: List of strengths
            gaps: List of gaps
            job: Job listing

        Returns:
            Summary string
        """
        template = self.SUMMARY_TEMPLATES.get(
            ranked.recommendation,
            self.SUMMARY_TEMPLATES[Recommendation.CONSIDER],
        )

        strength_areas = ", ".join([s.category for s in strengths[:3]]) or "general qualifications"
        gap_areas = ", ".join([g.category for g in gaps[:3]]) or "some areas"

        # Build experience summary snippet
        score = ranked.screening_score
        if score.experience_match >= 0.7:
            experience_summary = "solid relevant experience"
        elif score.experience_match >= 0.4:
            experience_summary = "some relevant experience"
        else:
            experience_summary = "limited directly relevant experience"

        return template.format(
            job_title=job.title,
            strength_areas=strength_areas,
            gap_areas=gap_areas,
            experience_summary=experience_summary,
        )

    def _generate_experience_summary(
        self,
        candidate: Student,
        resume: Optional[Resume],
    ) -> str:
        """Generate experience summary.

        Args:
            candidate: Candidate profile
            resume: Resume data

        Returns:
            Experience summary string
        """
        if not resume or not resume.parsed_data:
            return "Experience details not available from resume."

        parsed = resume.parsed_data
        experience = parsed.experience

        if not experience:
            return "No formal work experience listed."

        # Summarize experience
        total_years = parsed.total_experience_years
        companies = [e.company for e in experience if e.company][:3]
        titles = [e.title for e in experience if e.title][:3]

        summary_parts = []
        if total_years > 0:
            summary_parts.append(f"{total_years:.1f} years of experience")
        if titles:
            summary_parts.append(f"in roles including {', '.join(titles)}")
        if companies:
            summary_parts.append(f"at organizations such as {', '.join(companies)}")

        return " ".join(summary_parts) + "." if summary_parts else "Experience listed in resume."

    def _generate_education_summary(
        self,
        candidate: Student,
        resume: Optional[Resume],
    ) -> str:
        """Generate education summary.

        Args:
            candidate: Candidate profile
            resume: Resume data

        Returns:
            Education summary string
        """
        parts = []

        if candidate.university:
            parts.append(candidate.university)

        if candidate.major:
            parts.append(f"studying {candidate.major}")

        if candidate.graduation_year:
            parts.append(f"class of {candidate.graduation_year}")

        if candidate.gpa:
            parts.append(f"GPA: {candidate.gpa:.2f}")

        if parts:
            return " | ".join(parts)

        if resume and resume.parsed_data and resume.parsed_data.education:
            edu = resume.parsed_data.education[0]
            return f"{edu.degree} from {edu.institution}"

        return "Education details not available."

    def _validate_and_complete(self, explanation: CandidateExplanation) -> None:
        """Validate and complete explanation structure.

        Ensures 100% structured output by filling missing fields.

        Args:
            explanation: Explanation to validate
        """
        # Ensure summary exists
        if not explanation.summary:
            explanation.summary = f"Candidate ranked #{explanation.rank} with score {explanation.overall_score:.2f}."

        # Ensure at least one strength
        if not explanation.strengths:
            explanation.strengths.append(StrengthItem(
                category="general",
                description="Candidate meets basic qualification requirements",
                impact_score=0.5,
            ))

        # Ensure recommendation is valid
        if not explanation.recommendation:
            if explanation.overall_score >= 0.85:
                explanation.recommendation = Recommendation.HIGHLY_RECOMMENDED
            elif explanation.overall_score >= 0.70:
                explanation.recommendation = Recommendation.RECOMMENDED
            elif explanation.overall_score >= 0.50:
                explanation.recommendation = Recommendation.CONSIDER
            else:
                explanation.recommendation = Recommendation.NOT_RECOMMENDED

        # Ensure experience and education summaries
        if not explanation.experience_summary:
            explanation.experience_summary = "See resume for experience details."

        if not explanation.education_summary:
            explanation.education_summary = "See resume for education details."
