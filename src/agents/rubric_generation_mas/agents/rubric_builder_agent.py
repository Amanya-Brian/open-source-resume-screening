# agents/rubric_builder_agent.py

import logging
import re

logger = logging.getLogger(__name__)

# Fixed criteria — always in this order with these keys and base weights
_FIXED_CRITERIA = [
    {
        "id":     "education",
        "name":   "Education and Qualifications",
        "weight": 0.25,
        "keywords": ["degree", "qualif", "certif", "licens", "diploma", "bachelor",
                     "master", "phd", "accredit", "registr", "training", "educat"],
    },
    {
        "id":     "experience",
        "name":   "Experience",
        "weight": 0.35,
        "keywords": ["experience", "year", "worked", "role", "position", "employ",
                     "background", "history", "previous", "prior", "proven"],
    },
    {
        "id":     "technical_skills",
        "name":   "Technical Skills",
        "weight": 0.25,
        "keywords": ["skill", "technical", "software", "tool", "system", "equipment",
                     "technolog", "proficien", "competen", "knowleg", "operat",
                     "perform", "procedure", "protocol", "method"],
    },
    {
        "id":     "communication_skills",
        "name":   "Communication Skills",
        "weight": 0.15,
        "keywords": ["communicat", "report", "document", "record", "present",
                     "liaise", "coordinat", "team", "interpersonal", "verbal",
                     "written", "collaborat", "stakeholder"],
    },
]


def _build_description(criterion: dict, qualifications: list, responsibilities: list) -> str:
    """Build a criterion description by pulling relevant lines from the job context."""
    keywords = criterion["keywords"]
    matched = []

    for source in list(qualifications) + list(responsibilities):
        source_lower = source.lower()
        if any(kw in source_lower for kw in keywords):
            clean = source.strip().rstrip(".,;:")
            if clean and clean not in matched:
                matched.append(clean)

    if matched:
        # Use up to 3 most relevant lines joined into a description sentence
        return "; ".join(matched[:3])

    # Fallback: generic description
    return f"Evaluate candidate's {criterion['name'].lower()} against job requirements"


def _normalize_weights(criteria: list) -> list:
    """Ensure weights sum to exactly 1.0."""
    total = sum(float(c.get("weight", 0)) for c in criteria)
    if total <= 0:
        equal = round(1.0 / len(criteria), 4) if criteria else 0
        for c in criteria:
            c["weight"] = equal
    elif abs(total - 1.0) > 0.001:
        for c in criteria:
            c["weight"] = round(float(c.get("weight", 0)) / total, 4)
    return criteria


def rubric_builder_agent(
    document_context: dict,
    key_responsibilities: list,
    current_rubric: dict = None,
) -> dict:
    """Build a rubric with 4 fixed criteria derived from job qualifications and responsibilities.

    Always produces: Education & Qualifications, Experience, Technical Skills,
    Communication Skills. Descriptions are populated from the job's actual
    qualifications/requirements and responsibilities so scoring is job-specific.
    """
    qualifications = document_context.get("qualifications", [])
    # key_responsibilities is passed separately; also try document_context fallback
    responsibilities = key_responsibilities or document_context.get("responsibilities", [])

    criteria = []
    for template in _FIXED_CRITERIA:
        description = _build_description(template, qualifications, responsibilities)
        criteria.append({
            "id":          template["id"],
            "key":         template["id"],   # screening_service and rubrics API use "key"
            "name":        template["name"],
            "description": description,
            "weight":      template["weight"],
        })

    criteria = _normalize_weights(criteria)

    seniority = document_context.get("seniority", "")
    title = f"Rubric for {seniority}" if seniority else "Job Rubric"

    rubric = {
        "title":        title,
        "version":      1,
        "criteria":     criteria,
        "total_weight": round(sum(c["weight"] for c in criteria), 4),
    }

    logger.info(
        f"rubric_builder_agent: built {len(criteria)} fixed criteria — "
        f"{[c['name'] for c in criteria]}"
    )
    return rubric
