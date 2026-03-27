# agents/benchmarking_agent.py

import json
import logging
from src.services.llm_service import LLMService
from tools.weight_validator import weight_validator

logger = logging.getLogger(__name__)

PRINCIPLES = [
    "Clarity: every criterion must be unambiguous. "
    "Different evaluators must reach same interpretation.",

    "Measurability: each criterion must describe "
    "something observable not a feeling or assumption.",

    "Non-overlap: no two criteria should measure "
    "the same thing from different angles.",

    "Traceability: every criterion must map back "
    "to at least one responsibility.",

    "Balance: weights must reflect actual importance "
    "of responsibilities not be uniform by default.",

    "Completeness: no major responsibility should "
    "be left without a criterion.",

    "Fairness: no criterion should disadvantage "
    "a candidate for reasons unrelated to job performance."
]

def benchmarking_agent(
    current_rubric: dict,
    document_context: dict,
    key_responsibilities: list,
    previous_violations: list = None
) -> dict:

    # TOOL validates weights first — pure logic
    weight_check = weight_validator(
        current_rubric["criteria"]
    )

    llm = LLMService.get_instance()

    prompt = f"""
    CURRENT RUBRIC:
    {json.dumps(current_rubric, indent=2)}

    DOCUMENT CONTEXT:
    {json.dumps(document_context, indent=2)}

    KEY RESPONSIBILITIES:
    {json.dumps(key_responsibilities, indent=2)}

    WEIGHT VALIDATION RESULT:
    {json.dumps(weight_check, indent=2)}

    PREVIOUSLY FLAGGED VIOLATIONS
    (check if they were fixed):
    {json.dumps(previous_violations, indent=2)
     if previous_violations else "none — first iteration"}

    PRINCIPLES TO EVALUATE AGAINST:
    {json.dumps(PRINCIPLES, indent=2)}
    """

    response = llm.generate(
        prompt=prompt,
        system_prompt="""You are a rubric benchmarking expert.

                  You will receive:
                  - current rubric to evaluate
                  - document context for role
                    appropriateness check
                  - key responsibilities for
                    traceability check
                  - weight validation result
                  - previously flagged violations
                    to check if fixed
                  - 7 principles to evaluate against

                  YOUR JOB:
                  Reason carefully against each
                  of the 7 principles.
                  Flag every violation you find.
                  Check if previous violations
                  were fixed or still exist.
                  Use document_context to judge
                  if criteria fit this role.

                  Severity rules:
                  - High: clear violation that
                    will break rubric quality
                  - Medium: notable issue that
                    reduces effectiveness
                  - Low: minor improvement needed

                  Return ONLY valid JSON.
                  No explanation, no preamble,
                  no markdown code blocks.

                  Format:
                  {
                    "agent": "benchmarking",
                    "overall_score": 0.00,
                    "violations": [
                      {
                        "principle_violated": "...",
                        "criterion_affected": "...",
                        "reason": "...",
                        "severity": "High/Medium/Low",
                        "suggestion": "..."
                      }
                    ],
                    "fixed_from_previous": ["..."],
                    "still_unresolved": ["..."]
                  }""",
        max_tokens=800,
        temperature=0.1
    )

    try:
        verdict = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("benchmarking_agent: "
                       "failed to parse JSON response")
        verdict = {
            "agent":             "benchmarking",
            "overall_score":     0.0,
            "violations":        [],
            "fixed_from_previous": [],
            "still_unresolved":  []
        }

    return verdict