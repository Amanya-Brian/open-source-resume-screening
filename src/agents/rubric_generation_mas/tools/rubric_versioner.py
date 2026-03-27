# tools/rubric_versioner.py

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def rubric_versioner(
    updated_rubric: dict,
    previous_rubric: dict
) -> dict:
    """
    Increments rubric version number.
    Records what changed from previous version.
    No LLM — pure comparison logic.

    Args:
        updated_rubric:  newly refined rubric
        previous_rubric: rubric before refinement

    Returns:
        updated_rubric with version incremented
        and diff summary added
    """

    # increment version
    previous_version = previous_rubric.get("version", 1)
    updated_rubric["version"] = previous_version + 1
    updated_rubric["versioned_at"] = (
        datetime.utcnow().isoformat()
    )

    # compare criteria ids
    previous_ids = {
        c["id"]
        for c in previous_rubric.get("criteria", [])
    }
    updated_ids = {
        c["id"]
        for c in updated_rubric.get("criteria", [])
    }

    added_criteria   = updated_ids - previous_ids
    removed_criteria = previous_ids - updated_ids

    # compare weights
    previous_weights = {
        c["id"]: c.get("weight")
        for c in previous_rubric.get("criteria", [])
    }
    updated_weights = {
        c["id"]: c.get("weight")
        for c in updated_rubric.get("criteria", [])
    }

    weight_changes = {}
    for cid, new_weight in updated_weights.items():
        old_weight = previous_weights.get(cid)
        if old_weight is not None:
            if old_weight != new_weight:
                weight_changes[cid] = {
                    "from": old_weight,
                    "to":   new_weight
                }

    diff = {
        "from_version":   previous_version,
        "to_version":     updated_rubric["version"],
        "added_criteria": list(added_criteria),
        "removed_criteria": list(removed_criteria),
        "weight_changes": weight_changes
    }

    updated_rubric["diff"] = diff

    logger.info(
        f"rubric_versioner: "
        f"v{previous_version} → "
        f"v{updated_rubric['version']} | "
        f"added: {list(added_criteria)} | "
        f"removed: {list(removed_criteria)} | "
        f"weight changes: {len(weight_changes)}"
    )

    return updated_rubric