# agents/rubric_builder_agent.py

import logging

logger = logging.getLogger(__name__)

# Keywords that signal a responsibility is high-priority
_HIGH_WEIGHT_KEYWORDS = [
    "safety", "compli", "certif", "licens", "primary", "core", "critical",
    "ensure", "maintain", "manag", "supervis", "lead", "operat", "perform",
    "adher", "regulat", "standard", "protocol", "assess", "diagnos",
]

# Keywords that signal a responsibility is lower-priority / supporting
_LOW_WEIGHT_KEYWORDS = [
    "assist", "support", "help", "participat", "contribut", "attend",
    "liaise", "coordin", "report", "document", "record",
]


def _name_from_responsibility(r: str) -> str:
    """Turn a responsibility sentence into a short criterion name (first 5 words)."""
    words = r.rstrip(".,;:").split()
    return " ".join(words[:5]) if len(words) > 5 else r.rstrip(".,;:")


def _importance_score(responsibility: str) -> float:
    """Return a relative importance score (0.7 – 1.5) based on keywords."""
    r = responsibility.lower()
    for kw in _HIGH_WEIGHT_KEYWORDS:
        if kw in r:
            return 1.5
    for kw in _LOW_WEIGHT_KEYWORDS:
        if kw in r:
            return 0.7
    return 1.0


def _assign_weights(responsibilities: list) -> list:
    """Assign varied weights based on responsibility importance.

    Uses keyword analysis so weights reflect actual job priorities without
    requiring an LLM call — works reliably even when Ollama is unavailable.
    """
    scores = [_importance_score(r) for r in responsibilities]
    total = sum(scores)
    return [round(s / total, 4) for s in scores]


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
    current_rubric: dict = None,  # kept for API compatibility, not used
) -> dict:
    """Build a rubric from job responsibilities using pure-logic weight assignment.

    No LLM call — weights are derived deterministically from keyword importance
    so the rubric is always generated regardless of Ollama availability.
    """
    if not key_responsibilities:
        return {
            "title": "Draft Rubric",
            "version": 1,
            "criteria": [],
            "total_weight": 0.0,
        }

    weights = _assign_weights(key_responsibilities)

    criteria = [
        {
            "id":                     f"C{i+1}",
            "name":                   _name_from_responsibility(r),
            "description":            r,
            "linked_responsibility":  r,
            "weight":                 weights[i],
        }
        for i, r in enumerate(key_responsibilities)
    ]

    # Final safety pass — guarantee weights sum to 1.0
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
        f"rubric_builder_agent: built {len(criteria)} criteria — "
        f"weights: {[c['weight'] for c in criteria]}"
    )
    return rubric
