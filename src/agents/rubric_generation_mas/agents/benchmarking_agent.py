# agents/benchmarking_agent.py
# Pure logic — no LLM call needed.

import logging
from src.agents.rubric_generation_mas.tools.weight_validator import weight_validator

logger = logging.getLogger(__name__)


def benchmarking_agent(
    current_rubric: dict,
    _document_context: dict,
    _key_responsibilities: list,
    _previous_violations: list = None
) -> dict:
    weight_check = weight_validator(current_rubric.get("criteria", []))

    violations = []
    if not weight_check["valid"]:
        violations.append({
            "principle_violated": "Balance",
            "criterion_affected": "all",
            "reason": str(weight_check.get("issues", "Weights do not sum to 1.0")),
            "severity": "High",
            "suggestion": "Normalize weights to sum to 1.0"
        })

    return {
        "agent": "benchmarking",
        "overall_score": 1.0 if weight_check["valid"] else 0.5,
        "violations": violations,
        "fixed_from_previous": [],
        "still_unresolved": []
    }
