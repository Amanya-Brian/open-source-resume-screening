# agents/coverage_agent.py
# Pure logic — no LLM call needed.

import logging
from src.agents.rubric_generation_mas.tools.coverage_checker import coverage_checker

logger = logging.getLogger(__name__)


def coverage_agent(current_rubric: dict, key_responsibilities: list) -> dict:
    result = coverage_checker(key_responsibilities, current_rubric.get("criteria", []))

    flags = [
        {
            "responsibility": r,
            "issue": "No criterion covers this responsibility",
            "severity": "High"
        }
        for r in result.get("uncovered", [])
    ]

    return {
        "agent": "coverage",
        "coverage_score": result["coverage_score"],
        "uncovered": result.get("uncovered", []),
        "flags": flags
    }
