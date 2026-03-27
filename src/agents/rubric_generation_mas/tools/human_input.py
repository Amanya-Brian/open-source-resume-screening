# tools/human_input.py

import logging

logger = logging.getLogger(__name__)

def request_human_input(
    conflict_context: dict,
    current_rubric: dict
) -> dict:
    """
    Packages conflict and rubric state clearly.
    Calls present_to_human to surface to user.
    Only two actions allowed:
      1. tweak_weights
      2. add_criterion

    Args:
        conflict_context: what conflict was detected
        current_rubric:   current state of rubric

    Returns:
        human response dict with action and data
    """

    package = {

        "context": {
            "conflict":       conflict_context,
            "current_rubric": current_rubric
        },

        "allowed_actions": [
            {
                "action": "tweak_weights",
                "description": (
                    "Adjust weight of one or more "
                    "criteria. All weights must "
                    "still sum to 1.0"
                ),
                "expected_format": {
                    "action": "tweak_weights",
                    "changes": [
                        {
                            "criterion_id": "C1",
                            "new_weight":   0.25
                        }
                    ]
                }
            },
            {
                "action": "add_criterion",
                "description": (
                    "Add a new criterion missing "
                    "from the rubric"
                ),
                "expected_format": {
                    "action": "add_criterion",
                    "criterion": {
                        "id":          "C7",
                        "name":        "...",
                        "description": "...",
                        "levels": {
                            "1": "...",
                            "2": "...",
                            "3": "...",
                            "4": "...",
                            "5": "..."
                        },
                        "linked_responsibility": "...",
                        "weight": 0.00
                    }
                }
            }
        ]
    }

    # hand to whatever presents it to human
    # this function does not care how
    human_response = present_to_human(package)

    # validate response is one of two actions
    allowed = ["tweak_weights", "add_criterion"]
    if human_response.get("action") not in allowed:
        logger.error(
            f"human_input: invalid action received "
            f"— {human_response.get('action')}"
        )
        raise ValueError(
            f"Invalid action. Allowed: {allowed}"
        )

    logger.info(
        f"human_input: received action "
        f"— {human_response.get('action')}"
    )

    return human_response


def present_to_human(package: dict) -> dict:
    """
    Responsible for giving the package
    to the human and returning response.
    Parent app implements this.
    This is just the contract.

    Args:
        package: formatted question for human

    Returns:
        human response in expected format
    """
    # parent app overrides this
    raise NotImplementedError(
        "present_to_human must be implemented "
        "by the parent app"
    )