"""Scoring models and criteria for resume screening."""

from enum import IntEnum
from typing import Optional
from pydantic import BaseModel, Field


class ScoreLevel(IntEnum):
    """Scoring levels (0-5 scale)."""
    NOT_APPLICABLE = 0      # Not applicable/No evidence
    DOES_NOT_MEET = 1       # Does not meet requirements
    PARTIALLY_MEETS = 2     # Partially meets requirements
    MEETS = 3               # Meets requirements
    EXCEEDS = 4             # Exceeds requirements
    EXCEEDS_SIGNIFICANTLY = 5  # Exceeds requirements significantly

    @property
    def description(self) -> str:
        descriptions = {
            0: "Not applicable/No evidence",
            1: "Does not meet requirements",
            2: "Partially meets requirements",
            3: "Meets requirements",
            4: "Exceeds requirements",
            5: "Exceeds requirements significantly",
        }
        return descriptions[self.value]


class RecommendationLevel:
    """Recommendation levels based on weighted score percentage."""
    STRONG_YES = "STRONG YES - Interview"
    YES = "YES - Interview"
    MAYBE = "Maybe - Hold"
    NO = "No - Reject"

    @classmethod
    def from_percentage(cls, percentage: float) -> str:
        """Get recommendation based on percentage score.

        Tuned for higher recall to minimize false negatives:
        the cost of excluding a qualified candidate outweighs the
        operational cost of reviewing additional applicants.
        """
        if percentage >= 80:
            return cls.STRONG_YES
        elif percentage >= 65:
            return cls.YES
        elif percentage >= 50:
            return cls.MAYBE
        else:
            return cls.NO


class EvaluationCriterion(BaseModel):
    """Single evaluation criterion with weight."""
    name: str
    key: str  # Unique identifier
    weight: float = Field(ge=0.0, le=1.0)  # Weight as decimal (0.15 = 15%)
    description: str = ""

    @property
    def weight_percentage(self) -> int:
        return int(self.weight * 100)


class DefaultCriteria:
    """Default evaluation criteria matching the provided scoring matrix."""

    EDUCATION = EvaluationCriterion(
        name="Education & Qualifications",
        key="education",
        weight=0.15,
        description="Relevant degrees, certifications, and academic achievements"
    )

    EXPERIENCE = EvaluationCriterion(
        name="Relevant Experience (Years)",
        key="experience",
        weight=0.25,
        description="Years of relevant work experience in the field"
    )

    TECHNICAL_SKILLS = EvaluationCriterion(
        name="Technical Skills",
        key="technical_skills",
        weight=0.20,
        description="Required technical skills and tools proficiency"
    )

    INDUSTRY_KNOWLEDGE = EvaluationCriterion(
        name="Industry Knowledge",
        key="industry_knowledge",
        weight=0.15,
        description="Understanding of industry trends, practices, and domain expertise"
    )

    LEADERSHIP = EvaluationCriterion(
        name="Leadership Experience",
        key="leadership",
        weight=0.15,
        description="Team management, project leadership, and supervisory experience"
    )

    COMMUNICATION = EvaluationCriterion(
        name="Communication Skills",
        key="communication",
        weight=0.10,
        description="Written and verbal communication abilities demonstrated in application"
    )

    @classmethod
    def get_all(cls) -> list[EvaluationCriterion]:
        """Get all default criteria."""
        return [
            cls.EDUCATION,
            cls.EXPERIENCE,
            cls.TECHNICAL_SKILLS,
            cls.INDUSTRY_KNOWLEDGE,
            cls.LEADERSHIP,
            cls.COMMUNICATION,
        ]

    @classmethod
    def get_weights_dict(cls) -> dict[str, float]:
        """Get criteria weights as dictionary."""
        return {c.key: c.weight for c in cls.get_all()}


class CriterionScore(BaseModel):
    """Score for a single criterion."""
    criterion_key: str
    criterion_name: str
    weight: float
    raw_score: int = Field(ge=0, le=5)  # 0-5 scale
    weighted_score: float = 0.0
    evidence: str = ""  # Evidence/notes supporting the score

    def calculate_weighted(self) -> float:
        """Calculate weighted score."""
        self.weighted_score = self.raw_score * self.weight
        return self.weighted_score

    @property
    def score_description(self) -> str:
        return ScoreLevel(self.raw_score).description


class CandidateEvaluation(BaseModel):
    """Complete evaluation for a candidate."""
    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    criteria_scores: list[CriterionScore] = Field(default_factory=list)
    total_weighted_score: float = 0.0
    max_possible_score: float = 5.0
    percentage: float = 0.0
    recommendation: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    detailed_notes: str = ""

    def calculate_totals(self) -> None:
        """Calculate total weighted score and recommendation."""
        # Calculate weighted scores for each criterion
        for cs in self.criteria_scores:
            cs.calculate_weighted()

        # Sum weighted scores
        self.total_weighted_score = sum(cs.weighted_score for cs in self.criteria_scores)

        # Calculate percentage (out of max 5.0)
        self.percentage = (self.total_weighted_score / self.max_possible_score) * 100

        # Determine recommendation
        self.recommendation = RecommendationLevel.from_percentage(self.percentage)

    def get_score_for_criterion(self, key: str) -> Optional[CriterionScore]:
        """Get score for a specific criterion."""
        for cs in self.criteria_scores:
            if cs.criterion_key == key:
                return cs
        return None


class ScoringConfiguration(BaseModel):
    """Configuration for the scoring system."""
    criteria: list[EvaluationCriterion] = Field(default_factory=DefaultCriteria.get_all)
    score_levels: dict[int, str] = Field(default_factory=lambda: {
        0: "Not applicable/No evidence",
        1: "Does not meet requirements",
        2: "Partially meets requirements",
        3: "Meets requirements",
        4: "Exceeds requirements",
        5: "Exceeds requirements significantly",
    })
    recommendation_thresholds: dict[str, tuple[float, float]] = Field(default_factory=lambda: {
        "STRONG YES - Interview": (80, 100),
        "YES - Interview": (65, 79.99),
        "Maybe - Hold": (50, 64.99),
        "No - Reject": (0, 49.99),
    })

    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = sum(c.weight for c in self.criteria)
        return abs(total - 1.0) < 0.001
