# tools/weight_validator.py

import logging

logger = logging.getLogger(__name__)

def weight_validator(criteria: list) -> dict:
    """
    Checks all criteria weights sum to 1.0
    and flags any missing or zero weights.
    Pure math — no LLM.

    Args:
        criteria: list of criterion dicts
                  each with weight field

    Returns:
        dict with valid bool, total, issues list
    """

    issues = []
    total  = 0.0

    for criterion in criteria:
        weight = criterion.get("weight")
        cid    = criterion.get("id", "unknown")

        if weight is None:
            issues.append(
                f"{cid} has no weight assigned"
            )
            continue

        if weight == 0:
            issues.append(
                f"{cid} has weight of zero"
            )

        if weight < 0:
            issues.append(
                f"{cid} has negative weight"
            )

        total += weight

    # allow small floating point tolerance
    tolerance = 0.01
    sums_to_one = abs(total - 1.0) <= tolerance

    if not sums_to_one:
        issues.append(
            f"weights sum to {total:.3f} "
            f"not 1.0 — difference: "
            f"{abs(total - 1.0):.3f}"
        )

    valid = sums_to_one and len(issues) == 0

    if valid:
        logger.info("weight_validator: "
                    "all weights valid — "
                    f"sum = {total:.3f}")
    else:
        logger.warning(f"weight_validator: "
                       f"issues found — {issues}")

    return {
        "valid":  valid,
        "total":  round(total, 4),
        "issues": issues
    }