# agents/refiner_agent.py
# Pure logic — no LLM call needed.
# Applies weight fixes and human decisions directly.

import logging
from src.agents.rubric_generation_mas.tools.weight_validator import weight_validator
from src.agents.rubric_generation_mas.tools.rubric_versioner import rubric_versioner

logger = logging.getLogger(__name__)


def refiner_agent(
    current_rubric: dict,
    improvement_brief: dict,
    document_context: dict,
    key_responsibilities: list,
    human_decisions: list,
    previous_rubric_versions: list
) -> dict:
    import copy
    rubric = copy.deepcopy(current_rubric)
    criteria = rubric.get("criteria", [])

    # Apply human weight changes
    weight_changes = improvement_brief.get("weight_changes", {})
    for c in criteria:
        if c.get("id") in weight_changes:
            c["weight"] = float(weight_changes[c["id"]])
            logger.info(f"refiner: applied weight change {c['id']} → {c['weight']}")

    # Add any human-requested criteria
    for issue in improvement_brief.get("issues_to_fix", []):
        if issue.get("source") == "human" and issue.get("action") == "add_criterion":
            new_c = issue.get("criterion")
            if new_c:
                criteria.append(new_c)
                logger.info(f"refiner: added criterion {new_c.get('id')}")

    # Normalize weights to sum to 1.0
    total = sum(float(c.get("weight", 0)) for c in criteria)
    if total > 0 and abs(total - 1.0) > 0.001:
        for c in criteria:
            c["weight"] = round(float(c.get("weight", 0)) / total, 4)

    rubric["criteria"] = criteria
    rubric["total_weight"] = round(sum(c.get("weight", 0) for c in criteria), 4)

    weight_check = weight_validator(criteria)
    if not weight_check["valid"]:
        rubric["weight_warning"] = weight_check.get("issues")

    return rubric_versioner(rubric, current_rubric)
