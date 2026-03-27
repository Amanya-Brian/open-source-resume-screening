# agents/refiner_agent.py

import json
import logging
from src.services.llm_service import LLMService
from tools.weight_validator import weight_validator
from tools.rubric_versioner import rubric_versioner

logger = logging.getLogger(__name__)

def refiner_agent(
    current_rubric: dict,
    improvement_brief: dict,
    document_context: dict,
    key_responsibilities: list,
    human_decisions: list,
    previous_rubric_versions: list
) -> dict:

    llm = LLMService.get_instance()

    prompt = f"""
    CURRENT RUBRIC TO REFINE:
    {json.dumps(current_rubric, indent=2)}

    IMPROVEMENT BRIEF (what needs fixing):
    {json.dumps(improvement_brief, indent=2)}

    HUMAN DECISIONS (highest priority — override everything):
    {json.dumps(human_decisions, indent=2)}

    DOCUMENT CONTEXT (maintain tone and depth):
    {json.dumps(document_context, indent=2)}

    KEY RESPONSIBILITIES (all criteria must still trace here):
    {json.dumps(key_responsibilities, indent=2)}

    PREVIOUS RUBRIC VERSIONS (do not repeat these mistakes):
    {json.dumps(previous_rubric_versions, indent=2)}
    """

    response = llm.generate(
        prompt=prompt,
        system_prompt="""You are an expert rubric editor.

                  You will receive:
                  - current rubric to refine
                  - improvement brief: what to fix
                  - human decisions: explicitly
                    requested changes — these
                    override everything else
                  - document context: maintain
                    appropriate tone and depth
                  - key responsibilities: all
                    criteria must still trace back
                    to these after refinement
                  - previous rubric versions:
                    do not repeat past mistakes

                  YOUR JOB:
                  Apply ALL changes in the
                  improvement brief.
                  Respect ALL human decisions
                  above everything else.
                  Keep unchanged anything not
                  in the brief.
                  Ensure weights still sum to 1.0.
                  Increment version number by 1.

                  Return ONLY valid JSON.
                  No explanation, no preamble,
                  no markdown code blocks.
                  Return the complete updated
                  rubric in the same format
                  as the input rubric.""",
        max_tokens=800,
        temperature=0.2
    )

    try:
        updated_rubric = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("refiner_agent: "
                       "failed to parse JSON response")
        # return current rubric unchanged
        # if refiner fails
        updated_rubric = current_rubric

    # TOOL validates weights after refinement
    weight_check = weight_validator(
        updated_rubric["criteria"]
    )
    if not weight_check["valid"]:
        updated_rubric["weight_warning"] = (
            weight_check["issues"]
        )

    # TOOL versions the rubric
    updated_rubric = rubric_versioner(
        updated_rubric,
        current_rubric
    )

    return updated_rubric