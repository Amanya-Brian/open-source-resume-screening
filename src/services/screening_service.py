"""Screening service for evaluating candidates from MongoDB."""

import logging
import re
from typing import Any, Optional

from src.agents.fairness_agent import FairnessAgent, FairnessInput
from src.agents.base import AgentContext
from src.models.schemas import ComponentScore, RankedCandidate, ScreeningScore, Student, FairnessReport
from src.models.scoring import (
    CandidateEvaluation,
    CriterionScore,
    DefaultCriteria,
    EvaluationCriterion,
    RecommendationLevel,
    ScoreLevel,
    ScoringConfiguration,
)
from src.services.embedding_service import EmbeddingService
from src.services.mongo_service import MongoService

logger = logging.getLogger(__name__)

# Global LLM service (lazy loaded)
_llm_service = None


def get_llm_service():
    """Get LLM service instance (lazy loaded)."""
    global _llm_service
    if _llm_service is None:
        from src.services.llm_service import LLMService
        _llm_service = LLMService.get_instance()
    return _llm_service


class ScreeningService:
    """Service for screening candidates using data from MongoDB."""

    def __init__(
        self,
        mongo_service: MongoService,
        embedding_service: Optional[EmbeddingService] = None,
        scoring_config: Optional[ScoringConfiguration] = None,
        use_llm: bool = True,  # Enable LLM by default
    ):
        self.mongo = mongo_service
        self.embedding = embedding_service or EmbeddingService.get_instance()
        self.config = scoring_config or ScoringConfiguration()
        self.use_llm = use_llm
        self._llm_initialized = False

    async def screen_job_candidates(
        self,
        job_id: str,
    ) -> list[CandidateEvaluation]:
        """Screen all candidates for a job.

        Args:
            job_id: Job listing ID

        Returns:
            List of candidate evaluations, sorted by score
        """
        # Fetch job from MongoDB
        job = await self.mongo.find_one("job_listings", {"_id": job_id})
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        # Fetch applications for this job
        applications = await self.mongo.find_many(
            "applications",
            {"job_id": job_id}
        )

        if not applications:
            logger.warning(f"No applications found for job {job_id}")
            return []

        evaluations = []

        for app in applications:
            # Fetch resume/cover letter
            resume = await self.mongo.find_one(
                "resumes",
                {"student_id": app.get("student_id")}
            )

            evaluation = await self._evaluate_candidate(
                application=app,
                job=job,
                resume=resume,
            )
            evaluations.append(evaluation)

        # Sort by total weighted score (descending)
        evaluations.sort(key=lambda e: e.total_weighted_score, reverse=True)

        # Assign ranks
        for i, eval in enumerate(evaluations, 1):
            eval.detailed_notes = f"Rank #{i} of {len(evaluations)} candidates"

        # Run fairness analysis
        fairness_report = await self._run_fairness_analysis(job_id, evaluations, applications)

        # Store results in MongoDB (including fairness report)
        await self._store_results(job_id, evaluations, fairness_report)

        return evaluations

    async def _evaluate_candidate(
        self,
        application: dict[str, Any],
        job: dict[str, Any],
        resume: Optional[dict[str, Any]],
    ) -> CandidateEvaluation:
        """Evaluate a single candidate.

        Args:
            application: Application data
            job: Job listing data
            resume: Resume/cover letter data

        Returns:
            CandidateEvaluation with scores
        """
        candidate_name = f"{application.get('student_firstname', '')} {application.get('student_lastname', '')}".strip()
        candidate_id = application.get("student_id", "")

        # Get text content for analysis
        cover_letter = application.get("cover_letter", "")
        resume_text = resume.get("raw_text", "") if resume else ""
        full_text = f"{cover_letter}\n{resume_text}".strip()

        # Job requirements
        qualifications = job.get("qualifications", [])
        responsibilities = job.get("responsibilities", [])
        job_title = job.get("title", "")

        # Create evaluation
        evaluation = CandidateEvaluation(
            candidate_id=candidate_id,
            candidate_name=candidate_name or "Unknown",
            job_id=job.get("_id", ""),
            job_title=job_title,
        )

        # Try LLM-based evaluation if enabled
        llm_succeeded = False
        if self.use_llm and full_text:
            try:
                criteria_scores, strengths, concerns = await self._evaluate_with_llm(
                    full_text, qualifications, responsibilities
                )
                # Check if LLM actually scored (not all zeros from failed matching)
                if criteria_scores and any(cs.raw_score > 0 for cs in criteria_scores):
                    evaluation.criteria_scores = criteria_scores
                    evaluation.calculate_totals()

                    # Use strengths/concerns from the LLM evaluation directly
                    # (skip the separate explanation call to save time)
                    evaluation.strengths = strengths if strengths else self._extract_strengths(criteria_scores, full_text)
                    evaluation.concerns = concerns if concerns else self._extract_concerns(criteria_scores, qualifications)

                    # Build summary from the scores themselves
                    evaluation.detailed_notes = (
                        f"{candidate_name} scored {evaluation.percentage:.0f}% for {job_title}. "
                        f"Recommendation: {evaluation.recommendation}."
                    )

                    llm_succeeded = True
                    logger.info(f"LLM evaluation for {candidate_name}: {evaluation.total_weighted_score:.2f}/5 ({evaluation.recommendation})")
                else:
                    logger.warning(f"LLM returned empty/zero scores for {candidate_name}, using rule-based")
            except Exception as e:
                logger.warning(f"LLM evaluation failed for {candidate_name}, falling back to rule-based: {e}")

        # Fallback to rule-based evaluation (always runs if LLM failed)
        if not llm_succeeded:
            criteria_scores = []

            # 1. Education & Qualifications
            edu_score = self._score_education(full_text, qualifications)
            criteria_scores.append(edu_score)

            # 2. Relevant Experience
            exp_score = self._score_experience(full_text, qualifications)
            criteria_scores.append(exp_score)

            # 3. Technical Skills
            tech_score = self._score_technical_skills(full_text, qualifications)
            criteria_scores.append(tech_score)

            # 4. Industry Knowledge
            industry_score = self._score_industry_knowledge(full_text, job_title, responsibilities)
            criteria_scores.append(industry_score)

            # 5. Leadership Experience
            leadership_score = self._score_leadership(full_text)
            criteria_scores.append(leadership_score)

            # 6. Communication Skills
            comm_score = self._score_communication(cover_letter)
            criteria_scores.append(comm_score)

            evaluation.criteria_scores = criteria_scores
            evaluation.calculate_totals()

            # Extract detailed strengths and concerns
            evaluation.strengths = self._extract_strengths(criteria_scores, full_text)
            evaluation.concerns = self._extract_concerns(criteria_scores, qualifications)

            evaluation.detailed_notes = (
                f"{candidate_name} scored {evaluation.percentage:.0f}% for {job_title} "
                f"(rule-based evaluation). Recommendation: {evaluation.recommendation}."
            )

            logger.info(f"Rule-based evaluation for {candidate_name}: {evaluation.total_weighted_score:.2f}/5 ({evaluation.recommendation})")

        return evaluation

    async def _evaluate_with_llm(
        self,
        candidate_text: str,
        qualifications: list[str],
        responsibilities: list[str],
    ) -> tuple[list[CriterionScore], list[str], list[str]]:
        """Evaluate candidate using LLM.

        Args:
            candidate_text: Combined resume and cover letter
            qualifications: Job qualifications
            responsibilities: Job responsibilities

        Returns:
            Tuple of (criteria_scores, strengths, concerns)
        """
        llm = get_llm_service()

        # Initialize LLM if not done
        if not self._llm_initialized:
            logger.info("Initializing LLM for screening...")
            llm.initialize()
            self._llm_initialized = True

        # Prepare criteria for LLM
        criteria = [
            {"key": c.key, "name": c.name, "weight": c.weight, "description": c.description}
            for c in DefaultCriteria.get_all()
        ]

        # Get LLM evaluation
        job_requirements = {
            "qualifications": qualifications,
            "responsibilities": responsibilities,
        }

        result = llm.evaluate_candidate(candidate_text, job_requirements, criteria)

        # Parse LLM response into CriterionScore objects
        criteria_scores = []
        llm_scores = result.get("scores", [])
        matched_count = 0

        # Pre-normalize all LLM score criterion names for matching
        normalized_llm = {}
        for s in llm_scores:
            raw_name = s.get("criterion", "")
            norm_key = raw_name.lower().strip().replace(" ", "_").replace("-", "_").replace("&", "and")
            normalized_llm[norm_key] = s
            # Also store by raw name for exact match
            normalized_llm[raw_name.lower().strip()] = s

        for criterion in DefaultCriteria.get_all():
            # Try multiple matching strategies
            llm_score = None
            norm_crit = criterion.key.lower()

            # Strategy 1: Exact key match
            llm_score = normalized_llm.get(norm_crit)

            # Strategy 2: Normalized name match
            if not llm_score:
                norm_name = criterion.name.lower().replace(" ", "_").replace("&", "and")
                llm_score = normalized_llm.get(norm_name)

            # Strategy 3: Partial key match (e.g., "technical" matches "technical_skills")
            if not llm_score:
                for llm_key, s in normalized_llm.items():
                    if norm_crit in llm_key or llm_key in norm_crit:
                        llm_score = s
                        break

            # Strategy 4: Match by position in the list (if same count)
            if not llm_score and len(llm_scores) == len(DefaultCriteria.get_all()):
                idx = list(DefaultCriteria.get_all()).index(criterion)
                if idx < len(llm_scores):
                    llm_score = llm_scores[idx]
                    logger.debug(f"Matched {criterion.key} by position [{idx}]")

            if llm_score:
                score = min(5, max(0, int(float(llm_score.get("score", 0)))))
                evidence = llm_score.get("evidence", "LLM evaluation")
                matched_count += 1
            else:
                score = 0
                evidence = "Not evaluated - criterion not matched in LLM response"
                logger.warning(f"No LLM score for '{criterion.key}'. Available: {list(normalized_llm.keys())}")

            criteria_scores.append(CriterionScore(
                criterion_key=criterion.key,
                criterion_name=criterion.name,
                weight=criterion.weight,
                raw_score=score,
                evidence=evidence,
            ))

        logger.info(f"LLM criterion matching: {matched_count}/{len(DefaultCriteria.get_all())} matched")

        strengths = result.get("strengths", [])
        concerns = result.get("concerns", [])

        return criteria_scores, strengths, concerns

    async def _generate_llm_explanation(
        self,
        candidate_name: str,
        job_title: str,
        scores: list[CriterionScore],
        total_score: float,
        percentage: float,
        recommendation: str,
    ) -> dict[str, Any]:
        """Generate explanation using LLM.

        Args:
            candidate_name: Name of candidate
            job_title: Job title
            scores: List of criterion scores
            total_score: Total weighted score
            percentage: Percentage score
            recommendation: Recommendation level

        Returns:
            Dictionary with summary, strengths, concerns
        """
        try:
            llm = get_llm_service()

            scores_data = [
                {
                    "criterion_name": s.criterion_name,
                    "raw_score": s.raw_score,
                    "evidence": s.evidence,
                }
                for s in scores
            ]

            return llm.generate_explanation(
                candidate_name=candidate_name,
                job_title=job_title,
                scores=scores_data,
                total_score=total_score,
                percentage=percentage,
                recommendation=recommendation,
            )
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}")
            return {}

    def _score_education(
        self,
        text: str,
        qualifications: list[str],
    ) -> CriterionScore:
        """Score education and qualifications."""
        criterion = DefaultCriteria.EDUCATION
        text_lower = text.lower()

        score = 1  # Default: does not meet
        evidence_parts = []

        # Check for degree levels
        if any(deg in text_lower for deg in ["phd", "doctorate", "doctoral"]):
            score = 5
            evidence_parts.append("PhD/Doctorate degree")
        elif any(deg in text_lower for deg in ["master", "mba", "msc", "ma"]):
            score = 4
            evidence_parts.append("Master's degree")
        elif any(deg in text_lower for deg in ["bachelor", "bsc", "ba", "degree", "university"]):
            score = 3
            evidence_parts.append("Bachelor's degree")
        elif any(deg in text_lower for deg in ["diploma", "certificate", "certification"]):
            score = 2
            evidence_parts.append("Diploma/Certificate")

        # Check for relevant certifications
        cert_keywords = ["certified", "certification", "certificate", "cpa", "pmp", "aws", "google"]
        if any(kw in text_lower for kw in cert_keywords):
            score = min(5, score + 1)
            evidence_parts.append("Professional certifications mentioned")

        return CriterionScore(
            criterion_key=criterion.key,
            criterion_name=criterion.name,
            weight=criterion.weight,
            raw_score=score,
            evidence="; ".join(evidence_parts) if evidence_parts else "Limited educational information"
        )

    def _score_experience(
        self,
        text: str,
        qualifications: list[str],
    ) -> CriterionScore:
        """Score relevant experience."""
        criterion = DefaultCriteria.EXPERIENCE
        text_lower = text.lower()

        score = 1
        evidence_parts = []

        # Extract years of experience
        years_pattern = r'(\d+)\+?\s*(?:years?|yrs?)'
        matches = re.findall(years_pattern, text_lower)

        if matches:
            max_years = max(int(y) for y in matches)
            evidence_parts.append(f"{max_years}+ years experience mentioned")

            if max_years >= 8:
                score = 5
            elif max_years >= 5:
                score = 4
            elif max_years >= 3:
                score = 3
            elif max_years >= 1:
                score = 2

        # Check for company mentions
        company_indicators = ["worked at", "employed at", "position at", "role at", "internship"]
        if any(ind in text_lower for ind in company_indicators):
            score = max(score, 2)
            evidence_parts.append("Work experience mentioned")

        # Check for progressive experience
        if any(word in text_lower for word in ["promoted", "advanced", "progression", "grew"]):
            score = min(5, score + 1)
            evidence_parts.append("Career progression indicated")

        return CriterionScore(
            criterion_key=criterion.key,
            criterion_name=criterion.name,
            weight=criterion.weight,
            raw_score=score,
            evidence="; ".join(evidence_parts) if evidence_parts else "Limited experience information"
        )

    def _score_technical_skills(
        self,
        text: str,
        qualifications: list[str],
    ) -> CriterionScore:
        """Score technical skills."""
        criterion = DefaultCriteria.TECHNICAL_SKILLS
        text_lower = text.lower()
        qual_text = " ".join(qualifications).lower()

        # Extract required skills from qualifications
        skill_keywords = [
            "excel", "word", "powerpoint", "microsoft", "google workspace",
            "crm", "salesforce", "hubspot",
            "digital marketing", "seo", "sem", "social media",
            "data analysis", "analytics", "power bi", "tableau",
            "python", "java", "sql", "programming",
            "adobe", "photoshop", "illustrator", "canva", "figma",
            "project management", "agile", "scrum",
        ]

        # Count matching skills
        required_skills = [sk for sk in skill_keywords if sk in qual_text]
        candidate_skills = [sk for sk in skill_keywords if sk in text_lower]

        evidence_parts = []

        if not required_skills:
            # No specific technical skills required
            score = 3 if candidate_skills else 2
            evidence_parts.append(f"Skills found: {', '.join(candidate_skills[:5])}" if candidate_skills else "General skills")
        else:
            match_count = len(set(required_skills) & set(candidate_skills))
            match_ratio = match_count / len(required_skills) if required_skills else 0

            if match_ratio >= 0.8:
                score = 5
            elif match_ratio >= 0.6:
                score = 4
            elif match_ratio >= 0.4:
                score = 3
            elif match_ratio >= 0.2:
                score = 2
            else:
                score = 1

            evidence_parts.append(f"Matched {match_count}/{len(required_skills)} required skills")
            if candidate_skills:
                evidence_parts.append(f"Skills: {', '.join(candidate_skills[:5])}")

        return CriterionScore(
            criterion_key=criterion.key,
            criterion_name=criterion.name,
            weight=criterion.weight,
            raw_score=score,
            evidence="; ".join(evidence_parts) if evidence_parts else "Technical skills assessment"
        )

    def _score_industry_knowledge(
        self,
        text: str,
        job_title: str,
        responsibilities: list[str],
    ) -> CriterionScore:
        """Score industry knowledge."""
        criterion = DefaultCriteria.INDUSTRY_KNOWLEDGE
        text_lower = text.lower()
        job_lower = job_title.lower()
        resp_text = " ".join(responsibilities).lower()

        score = 2  # Default: partially meets
        evidence_parts = []

        # Check for industry-specific terms
        industry_terms = {
            "marketing": ["marketing", "campaign", "brand", "advertising", "promotion", "market research"],
            "sales": ["sales", "revenue", "client", "customer", "leads", "targets", "quotas"],
            "tech": ["software", "development", "programming", "engineering", "technology"],
            "finance": ["finance", "accounting", "budget", "financial", "investment"],
            "hr": ["human resources", "recruitment", "hiring", "talent", "personnel"],
        }

        # Determine job industry
        job_industry = None
        for industry, terms in industry_terms.items():
            if any(term in job_lower or term in resp_text for term in terms):
                job_industry = industry
                break

        if job_industry:
            industry_term_count = sum(
                1 for term in industry_terms[job_industry]
                if term in text_lower
            )

            if industry_term_count >= 5:
                score = 5
                evidence_parts.append(f"Strong {job_industry} industry knowledge")
            elif industry_term_count >= 3:
                score = 4
                evidence_parts.append(f"Good {job_industry} industry experience")
            elif industry_term_count >= 1:
                score = 3
                evidence_parts.append(f"Some {job_industry} industry exposure")
            else:
                score = 2
                evidence_parts.append("Limited industry-specific experience")
        else:
            score = 3
            evidence_parts.append("General industry knowledge")

        return CriterionScore(
            criterion_key=criterion.key,
            criterion_name=criterion.name,
            weight=criterion.weight,
            raw_score=score,
            evidence="; ".join(evidence_parts) if evidence_parts else "Industry knowledge assessment"
        )

    def _score_leadership(
        self,
        text: str,
    ) -> CriterionScore:
        """Score leadership experience."""
        criterion = DefaultCriteria.LEADERSHIP
        text_lower = text.lower()

        score = 1
        evidence_parts = []

        # Leadership indicators
        leadership_keywords = [
            "led", "managed", "supervised", "directed", "oversaw",
            "team lead", "manager", "head of", "director",
            "coordinated", "mentored", "trained",
        ]

        leadership_count = sum(1 for kw in leadership_keywords if kw in text_lower)

        # Team size mentions
        team_pattern = r'team\s*of\s*(\d+)'
        team_matches = re.findall(team_pattern, text_lower)

        if team_matches:
            max_team = max(int(t) for t in team_matches)
            evidence_parts.append(f"Managed team of {max_team}")

            if max_team >= 10:
                score = 5
            elif max_team >= 5:
                score = 4
            elif max_team >= 2:
                score = 3

        elif leadership_count >= 3:
            score = 4
            evidence_parts.append("Multiple leadership experiences mentioned")
        elif leadership_count >= 1:
            score = 3
            evidence_parts.append("Some leadership experience")
        else:
            score = 2
            evidence_parts.append("Limited leadership experience shown")

        # Budget management
        if any(word in text_lower for word in ["budget", "revenue", "p&l", "financial"]):
            score = min(5, score + 1)
            evidence_parts.append("Budget/financial responsibility")

        return CriterionScore(
            criterion_key=criterion.key,
            criterion_name=criterion.name,
            weight=criterion.weight,
            raw_score=score,
            evidence="; ".join(evidence_parts) if evidence_parts else "Leadership assessment"
        )

    def _score_communication(
        self,
        cover_letter: str,
    ) -> CriterionScore:
        """Score communication skills based on cover letter quality."""
        criterion = DefaultCriteria.COMMUNICATION

        if not cover_letter or len(cover_letter.strip()) < 50:
            return CriterionScore(
                criterion_key=criterion.key,
                criterion_name=criterion.name,
                weight=criterion.weight,
                raw_score=1,
                evidence="No cover letter or very brief application"
            )

        score = 3  # Default: meets requirements
        evidence_parts = []

        # Length analysis
        word_count = len(cover_letter.split())
        if word_count >= 200:
            score = 4
            evidence_parts.append(f"Well-developed cover letter ({word_count} words)")
        elif word_count >= 100:
            score = 3
            evidence_parts.append(f"Adequate cover letter ({word_count} words)")
        else:
            score = 2
            evidence_parts.append(f"Brief cover letter ({word_count} words)")

        # Structure analysis
        if any(greeting in cover_letter.lower() for greeting in ["dear", "hello", "hi"]):
            evidence_parts.append("Proper greeting")

        if any(closing in cover_letter.lower() for closing in ["sincerely", "regards", "thank you"]):
            score = min(5, score + 1)
            evidence_parts.append("Professional closing")

        # Personalization
        if "your company" in cover_letter.lower() or "your organization" in cover_letter.lower():
            evidence_parts.append("Shows interest in company")

        # Quantifiable achievements
        if re.search(r'\d+%|\$\d+|\d+\s*(years?|people|team)', cover_letter.lower()):
            score = min(5, score + 1)
            evidence_parts.append("Includes quantifiable achievements")

        return CriterionScore(
            criterion_key=criterion.key,
            criterion_name=criterion.name,
            weight=criterion.weight,
            raw_score=score,
            evidence="; ".join(evidence_parts) if evidence_parts else "Communication assessment"
        )

    def _extract_strengths(
        self,
        scores: list[CriterionScore],
        text: str,
    ) -> list[str]:
        """Extract detailed candidate strengths based on scores and text."""
        strengths = []

        # Sort scores descending - strongest areas first
        sorted_scores = sorted(scores, key=lambda s: s.raw_score, reverse=True)

        for score in sorted_scores:
            if score.raw_score >= 4:
                strengths.append(
                    f"Strong {score.criterion_name} (scored {score.raw_score}/5): {score.evidence}"
                )
            elif score.raw_score == 3 and len(strengths) < 2:
                # Include "meets" scores if we don't have enough strengths yet
                strengths.append(
                    f"Meets {score.criterion_name} requirements (scored {score.raw_score}/5): {score.evidence}"
                )

        # Text-based strengths
        text_lower = text.lower()

        if re.search(r'\d+\+?\s*years?', text_lower):
            years_match = re.search(r'(\d+)\+?\s*years?', text_lower)
            if years_match:
                strengths.append(f"Has {years_match.group(1)}+ years of professional experience")

        if "track record" in text_lower or "successfully" in text_lower:
            strengths.append("Demonstrates a track record of successful outcomes")

        if "award" in text_lower or "recognized" in text_lower or "honour" in text_lower:
            strengths.append("Has received professional awards or recognition")

        if re.search(r'team\s*of\s*\d+', text_lower):
            team_match = re.search(r'team\s*of\s*(\d+)', text_lower)
            if team_match:
                strengths.append(f"Experience leading a team of {team_match.group(1)} people")

        # Ensure at least one strength
        if not strengths:
            best = sorted_scores[0] if sorted_scores else None
            if best:
                strengths.append(f"Strongest area is {best.criterion_name} (scored {best.raw_score}/5): {best.evidence}")

        return strengths[:5]

    def _extract_concerns(
        self,
        scores: list[CriterionScore],
        qualifications: list[str],
    ) -> list[str]:
        """Extract detailed concerns about the candidate."""
        concerns = []

        # Sort scores ascending - weakest areas first
        sorted_scores = sorted(scores, key=lambda s: s.raw_score)

        for score in sorted_scores:
            if score.raw_score <= 1:
                concerns.append(
                    f"Weak {score.criterion_name} (scored {score.raw_score}/5): {score.evidence}. "
                    f"This criterion carries {int(score.weight * 100)}% weight."
                )
            elif score.raw_score == 2:
                concerns.append(
                    f"Below requirements in {score.criterion_name} (scored {score.raw_score}/5): {score.evidence}"
                )

        # Qualification gaps
        if qualifications:
            concern_quals = []
            for qual in qualifications[:5]:
                qual_lower = qual.lower()
                # Check if any key terms from the qualification appear in scores evidence
                found = False
                for score in scores:
                    if any(word in score.evidence.lower() for word in qual_lower.split()[:3] if len(word) > 3):
                        found = True
                        break
                if not found:
                    concern_quals.append(qual)
            if concern_quals:
                concerns.append(f"May not meet these qualifications: {'; '.join(concern_quals[:2])}")

        # Overall assessment
        low_scores = [s for s in scores if s.raw_score <= 2]
        high_weight_low = [s for s in low_scores if s.weight >= 0.2]
        if high_weight_low:
            names = [s.criterion_name for s in high_weight_low]
            concerns.append(f"Underperforms in high-weight areas: {', '.join(names)}")

        # Ensure at least one concern
        if not concerns:
            weakest = sorted_scores[0] if sorted_scores else None
            if weakest:
                concerns.append(f"Weakest area is {weakest.criterion_name} (scored {weakest.raw_score}/5): {weakest.evidence}")

        return concerns[:5]

    async def _run_fairness_analysis(
        self,
        job_id: str,
        evaluations: list[CandidateEvaluation],
        applications: list[dict[str, Any]],
    ) -> Optional[FairnessReport]:
        """Run fairness analysis on screening results.

        Args:
            job_id: Job listing ID
            evaluations: Candidate evaluations with scores
            applications: Raw application data with demographic info

        Returns:
            FairnessReport or None if analysis fails
        """
        try:
            logger.info(f"Running fairness analysis for job {job_id}...")

            # Build ScreeningScore objects for each evaluation
            scores = {}
            for eval in evaluations:
                # Normalize total_weighted_score (0-5 scale) to 0-1 scale
                normalized_overall = min(1.0, eval.total_weighted_score / 5.0)

                # Convert criteria scores to ComponentScore objects
                component_scores = [
                    ComponentScore(
                        name=cs.criterion_name,
                        score=min(1.0, cs.raw_score / 5.0),
                        weight=cs.weight,
                        weighted_score=min(1.0, cs.weighted_score / 5.0),
                        details=cs.evidence,
                    )
                    for cs in eval.criteria_scores
                ]

                scores[eval.candidate_id] = ScreeningScore(
                    candidate_id=eval.candidate_id,
                    job_id=job_id,
                    overall_score=normalized_overall,
                    component_scores=component_scores,
                )

            # Convert evaluations to RankedCandidate format
            ranked_candidates = []
            for idx, eval in enumerate(evaluations):
                normalized_score = min(1.0, eval.total_weighted_score / 5.0)
                ranked_candidates.append(
                    RankedCandidate(
                        candidate_id=eval.candidate_id,
                        rank=idx + 1,
                        score=normalized_score,
                        percentile=100 * (len(evaluations) - idx) / len(evaluations),
                        screening_score=scores[eval.candidate_id],
                        recommendation=eval.recommendation,
                    )
                )

            # Convert applications to Student format
            students = []
            for app in applications:
                student = Student(
                    _id=app.get("student_id", ""),
                    first_name=app.get("student_firstname", "Unknown"),
                    last_name=app.get("student_lastname", "Unknown"),
                    email=app.get("student_email", ""),
                    gender=app.get("gender"),
                    age_group=app.get("age_group"),
                    ethnicity=app.get("ethnicity"),
                    nationality=app.get("nationality"),
                )
                students.append(student)

            # Create FairnessAgent and run analysis
            fairness_agent = FairnessAgent()
            fairness_input = FairnessInput(
                ranked_candidates=ranked_candidates,
                candidates=students,
                scores=scores,
                job_id=job_id,
            )

            context = AgentContext(job_id=job_id, session_id=f"screening-{job_id}")
            result = await fairness_agent.execute(fairness_input, context)

            if result.success and result.data:
                fairness_report = result.data.fairness_report

                # Log fairness results
                logger.info(
                    f"Fairness analysis complete: "
                    f"DIR={fairness_report.metrics.disparate_impact_ratio:.3f}, "
                    f"Compliant={fairness_report.is_compliant}"
                )

                if not fairness_report.is_compliant:
                    logger.warning(f"Fairness violations detected: {fairness_report.violations}")

                return fairness_report
            else:
                logger.error(f"Fairness analysis failed: {result.errors}")
                return None

        except Exception as e:
            logger.error(f"Fairness analysis error: {e}", exc_info=True)
            return None

    async def _store_results(
        self,
        job_id: str,
        evaluations: list[CandidateEvaluation],
        fairness_report: Optional[FairnessReport] = None,
    ) -> None:
        """Store screening results in MongoDB."""
        for eval in evaluations:
            result_doc = {
                "_id": f"{job_id}-{eval.candidate_id}",
                "job_id": job_id,
                "candidate_id": eval.candidate_id,
                "candidate_name": eval.candidate_name,
                "job_title": eval.job_title,
                "total_weighted_score": eval.total_weighted_score,
                "percentage": eval.percentage,
                "recommendation": eval.recommendation,
                "criteria_scores": [
                    {
                        "criterion_key": cs.criterion_key,
                        "criterion_name": cs.criterion_name,
                        "weight": cs.weight,
                        "raw_score": cs.raw_score,
                        "weighted_score": cs.weighted_score,
                        "evidence": cs.evidence,
                    }
                    for cs in eval.criteria_scores
                ],
                "strengths": eval.strengths,
                "concerns": eval.concerns,
            }

            await self.mongo.update_one(
                "screening_results",
                {"_id": result_doc["_id"]},
                {"$set": result_doc},
                upsert=True
            )

        logger.info(f"Stored {len(evaluations)} screening results for job {job_id}")

        # Store fairness report
        if fairness_report:
            fairness_doc = {
                "_id": f"fairness-{job_id}",
                "job_id": job_id,
                "session_id": fairness_report.session_id,
                "is_compliant": fairness_report.is_compliant,
                "metrics": {
                    "disparate_impact_ratio": fairness_report.metrics.disparate_impact_ratio,
                    "demographic_parity": fairness_report.metrics.demographic_parity,
                    "equal_opportunity": fairness_report.metrics.equal_opportunity,
                    "attribute_variance": fairness_report.metrics.attribute_variance,
                },
                "violations": fairness_report.violations,
                "recommendations": fairness_report.recommendations,
                "generated_at": fairness_report.generated_at.isoformat() if fairness_report.generated_at else None,
            }

            await self.mongo.update_one(
                "fairness_reports",
                {"_id": fairness_doc["_id"]},
                {"$set": fairness_doc},
                upsert=True
            )

            logger.info(
                f"Stored fairness report for job {job_id}: "
                f"Compliant={fairness_report.is_compliant}, "
                f"DIR={fairness_report.metrics.disparate_impact_ratio:.3f}"
            )
