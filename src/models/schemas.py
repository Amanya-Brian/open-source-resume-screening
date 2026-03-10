"""Pydantic data models for the resume screening system."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from src.models.scoring import EvaluationCriterion


# === Enums ===


class ApplicationStatus(str, Enum):
    """Status of a job application."""

    PENDING = "pending"
    REVIEWED = "reviewed"
    SHORTLISTED = "shortlisted"
    REJECTED = "rejected"
    HIRED = "hired"


class JobType(str, Enum):
    """Type of job listing."""

    FULL_TIME = "full-time"
    PART_TIME = "part-time"
    INTERNSHIP = "internship"
    CONTRACT = "contract"


class Recommendation(str, Enum):
    """Candidate recommendation level."""

    HIGHLY_RECOMMENDED = "Highly Recommended"
    RECOMMENDED = "Recommended"
    CONSIDER = "Consider"
    NOT_RECOMMENDED = "Not Recommended"


class PipelineStatus(str, Enum):
    """Status of screening pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# === Core Entities ===


class EducationEntry(BaseModel):
    """Education history entry."""

    institution: str
    degree: str
    field_of_study: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0)
    is_current: bool = False


class ExperienceEntry(BaseModel):
    """Work experience entry."""

    company: str
    title: str
    description: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_current: bool = False
    skills_used: list[str] = Field(default_factory=list)

    @property
    def duration_months(self) -> Optional[int]:
        """Calculate duration in months."""
        if not self.start_date:
            return None
        end = self.end_date or datetime.now()
        return (end.year - self.start_date.year) * 12 + (end.month - self.start_date.month)


class ProjectEntry(BaseModel):
    """Project entry from resume."""

    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    url: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class ContactInfo(BaseModel):
    """Contact information."""

    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None


class ParsedResume(BaseModel):
    """Parsed resume data structure."""

    contact_info: ContactInfo = Field(default_factory=ContactInfo)
    summary: Optional[str] = None
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    raw_text: str = ""

    @property
    def total_experience_months(self) -> int:
        """Calculate total experience in months."""
        return sum(
            exp.duration_months or 0
            for exp in self.experience
        )

    @property
    def total_experience_years(self) -> float:
        """Calculate total experience in years."""
        return self.total_experience_months / 12


class Student(BaseModel):
    """Student/candidate profile."""

    id: str = Field(alias="_id", default="")
    first_name: str
    last_name: str
    email: str
    university: Optional[str] = None
    graduation_year: Optional[int] = None
    major: Optional[str] = None
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0)
    resume_url: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    # Demographics (for fairness analysis - anonymized in processing)
    gender: Optional[str] = None
    age_group: Optional[str] = None
    ethnicity: Optional[str] = None
    nationality: Optional[str] = None

    class Config:
        populate_by_name = True

    @property
    def full_name(self) -> str:
        """Return full name."""
        return f"{self.first_name} {self.last_name}"


class JobListing(BaseModel):
    """Job posting details."""

    id: str = Field(alias="_id", default="")
    title: str
    company: str
    description: str
    requirements: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    job_type: JobType = JobType.FULL_TIME
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    experience_years_min: Optional[int] = None
    experience_years_max: Optional[int] = None
    posted_at: datetime = Field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    is_active: bool = True
     # Optional reference to a stored rubric document
    rubric_id: Optional[str] = None

    class Config:
        populate_by_name = True


class Rubric(BaseModel):
    """Stored evaluation rubric.

    This defines the scoring dimensions (criteria) and their weights.
    It is independent of any particular candidate evaluation; a job can
    reference the rubric via its ``rubric_id``.
    """

    id: str = Field(alias="_id", default="")
    name: Optional[str] = None
    description: Optional[str] = None
    criteria: list[EvaluationCriterion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        populate_by_name = True


class Application(BaseModel):
    """Job application."""

    id: str = Field(alias="_id", default="")
    student_id: str
    job_id: str
    status: ApplicationStatus = ApplicationStatus.PENDING
    applied_at: datetime = Field(default_factory=datetime.now)
    cover_letter: Optional[str] = None
    resume_id: Optional[str] = None

    class Config:
        populate_by_name = True


class Resume(BaseModel):
    """Resume document."""

    id: str = Field(alias="_id", default="")
    student_id: str
    raw_text: str
    parsed_data: ParsedResume = Field(default_factory=ParsedResume)
    file_type: str = "pdf"  # pdf, docx
    file_url: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.now)

    class Config:
        populate_by_name = True


# === Screening Results ===


class ComponentScore(BaseModel):
    """Individual component score."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0, le=1.0)
    details: Optional[str] = None


class ScreeningScore(BaseModel):
    """Screening score for a candidate."""

    candidate_id: str
    job_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    component_scores: list[ComponentScore] = Field(default_factory=list)
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_match: float = Field(ge=0.0, le=1.0, default=0.0)
    education_match: float = Field(ge=0.0, le=1.0, default=0.0)
    computed_at: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.8)

    @property
    def component_scores_dict(self) -> dict[str, float]:
        """Return component scores as dictionary."""
        return {cs.name: cs.score for cs in self.component_scores}


class RankedCandidate(BaseModel):
    """Ranked candidate with scores."""

    candidate_id: str
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    percentile: float = Field(ge=0.0, le=100.0)
    screening_score: ScreeningScore
    recommendation: Recommendation = Recommendation.CONSIDER

    @field_validator("recommendation", mode="before")
    @classmethod
    def parse_recommendation(cls, v: Any) -> Recommendation:
        """Parse recommendation from string or enum."""
        if isinstance(v, Recommendation):
            return v
        if isinstance(v, str):
            # Try exact match first
            try:
                return Recommendation(v)
            except ValueError:
                pass
            # Try case-insensitive match
            for rec in Recommendation:
                if v.lower() == rec.value.lower():
                    return rec
        return Recommendation.CONSIDER


class RankingResult(BaseModel):
    """Complete ranking result for a job."""

    job_id: str
    total_candidates: int
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)


# === Fairness Models ===


class CounterfactualResult(BaseModel):
    """Result of counterfactual fairness test."""

    candidate_id: str
    original_rank: int
    modified_attribute: str
    original_value: str
    modified_value: str
    new_rank: int
    rank_change: int = 0  # Should be 0 for fair system

    @property
    def is_fair(self) -> bool:
        """Check if this result shows fair treatment."""
        return self.rank_change == 0


class FairnessMetrics(BaseModel):
    """Fairness metrics for a screening session."""

    disparate_impact_ratio: float = Field(ge=0.0, le=2.0, default=1.0)
    demographic_parity: float = Field(ge=0.0, le=1.0, default=1.0)
    equal_opportunity: float = Field(ge=0.0, le=1.0, default=1.0)
    attribute_variance: dict[str, float] = Field(default_factory=dict)
    counterfactual_results: list[CounterfactualResult] = Field(default_factory=list)

    @property
    def max_variance(self) -> float:
        """Return maximum variance across attributes."""
        if not self.attribute_variance:
            return 0.0
        return max(self.attribute_variance.values())


class FairnessReport(BaseModel):
    """Complete fairness analysis report."""

    job_id: str
    session_id: str
    metrics: FairnessMetrics = Field(default_factory=FairnessMetrics)
    is_compliant: bool = True
    violations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)

    def check_compliance(self, dir_threshold: float = 0.8, variance_threshold: float = 0.001) -> bool:
        """Check if metrics meet compliance thresholds."""
        dir_ok = self.metrics.disparate_impact_ratio >= dir_threshold
        variance_ok = self.metrics.max_variance <= variance_threshold
        return dir_ok and variance_ok


# === Explanation Models ===


class StrengthItem(BaseModel):
    """Strength identified in candidate evaluation."""

    category: str  # skills, experience, education, projects
    description: str
    impact_score: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: Optional[str] = None


class GapItem(BaseModel):
    """Gap identified in candidate evaluation."""

    category: str
    required: str
    candidate_status: str  # missing, partial, alternative
    suggestion: Optional[str] = None
    severity: str = "medium"  # low, medium, high


class CandidateExplanation(BaseModel):
    """Detailed explanation for a candidate's evaluation."""

    candidate_id: str
    job_id: str
    rank: int
    overall_score: float = Field(ge=0.0, le=1.0)
    recommendation: Recommendation
    summary: str  # 2-3 sentence summary
    strengths: list[StrengthItem] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    matching_skills: list[str] = Field(default_factory=list)
    experience_summary: str = ""
    education_summary: str = ""
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.8)
    generated_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_complete(self) -> bool:
        """Check if explanation has all required fields."""
        return bool(
            self.summary
            and self.recommendation
            and len(self.strengths) >= 1
        )


# === Pipeline Models ===


class ScreeningOptions(BaseModel):
    """Options for screening pipeline."""

    top_k: int = Field(default=10, ge=1)
    min_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    include_explanations: bool = True
    run_fairness_check: bool = True
    run_validation: bool = True
    weight_config: Optional[dict[str, float]] = None


class PipelineStep(BaseModel):
    """Pipeline step definition."""

    name: str
    agent_name: str
    dependencies: list[str] = Field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3


class ScreeningSession(BaseModel):
    """Screening session tracking."""

    id: str = Field(alias="_id", default="")
    job_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    options: ScreeningOptions = Field(default_factory=ScreeningOptions)
    current_step: Optional[str] = None
    steps_completed: list[str] = Field(default_factory=list)
    steps_remaining: list[str] = Field(default_factory=list)
    progress_percent: float = Field(ge=0.0, le=100.0, default=0.0)
    total_candidates: int = 0
    processed_candidates: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        populate_by_name = True

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate session duration."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


class ScreeningResult(BaseModel):
    """Complete screening result."""

    session: ScreeningSession
    ranking: RankingResult
    fairness_report: Optional[FairnessReport] = None
    explanations: list[CandidateExplanation] = Field(default_factory=list)
    validation_agreement: Optional[float] = None


# === Historical Data ===


class HistoricalDecision(BaseModel):
    """Historical hiring decision for validation."""

    id: str = Field(alias="_id", default="")
    job_id: str
    candidate_id: str
    was_hired: bool
    was_shortlisted: bool
    final_rank: Optional[int] = None
    decision_date: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None

    class Config:
        populate_by_name = True


class ValidationResult(BaseModel):
    """Validation against historical decisions."""

    job_id: str
    session_id: str
    agreement_rate: float = Field(ge=0.0, le=1.0)
    total_comparisons: int = 0
    matches: int = 0
    mismatches: int = 0
    anomalies: list[str] = Field(default_factory=list)
    is_valid: bool = True
    generated_at: datetime = Field(default_factory=datetime.now)

    @property
    def mismatch_rate(self) -> float:
        """Calculate mismatch rate."""
        if self.total_comparisons == 0:
            return 0.0
        return self.mismatches / self.total_comparisons
