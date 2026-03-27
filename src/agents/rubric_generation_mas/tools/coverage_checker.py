# tools/coverage_checker.py

import logging

logger = logging.getLogger(__name__)

def coverage_checker(
    key_responsibilities: list,
    criteria: list
) -> dict:
    """
    Compares responsibilities list against
    criteria linked_responsibility fields.
    Pure logic — no LLM.

    Args:
        key_responsibilities: list of responsibility strings
        criteria: list of criterion dicts each with
                  linked_responsibility field

    Returns:
        dict with covered, uncovered, coverage_score
    """

    # collect all linked responsibilities
    # from criteria — lowercase for comparison
    linked = [
        c.get("linked_responsibility", "").lower().strip()
        for c in criteria
    ]

    covered   = []
    uncovered = []

    for responsibility in key_responsibilities:
        resp_lower = responsibility.lower().strip()

        # check if this responsibility is
        # linked to at least one criterion
        is_covered = any(
            resp_lower in linked_resp or
            linked_resp in resp_lower
            for linked_resp in linked
        )

        if is_covered:
            covered.append(responsibility)
        else:
            uncovered.append(responsibility)

    total          = len(key_responsibilities)
    covered_count  = len(covered)
    coverage_score = covered_count / total if total > 0 else 0.0

    logger.info(f"coverage_checker: "
                f"{covered_count}/{total} "
                f"responsibilities covered "
                f"({coverage_score:.0%})")

    return {
        "covered":        covered,
        "uncovered":      uncovered,
        "coverage_score": round(coverage_score, 2),
        "total":          total,
        "covered_count":  covered_count
    }