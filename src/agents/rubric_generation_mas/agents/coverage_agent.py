# agents/coverage_agent.py

import json
import logging
from src.services.llm_service import LLMService
from tools.coverage_checker import coverage_checker

logger = logging.getLogger(__name__)

def coverage_agent(
    current_rubric: dict,
    key_responsibilities: list
) -> dict:

    # TOOL does comparison first — pure logic
    coverage_result = coverage_checker(
        key_responsibilities,
        current_rubric["criteria"]
    )

    # if perfect coverage no LLM needed
    if coverage_result["coverage_score"] == 1.0:
        return {
            "agent":          "coverage",
            "coverage_score": 1.0,
            "uncovered":      [],
            "flags":          []
        }

    # AGENT reasons about the gaps
    llm = LLMService.get_instance()

    prompt = f"""
    COVERAGE CHECK RESULT:
    {json.dumps(coverage_result, indent=2)}

    FULL RUBRIC:
    {json.dumps(current_rubric, indent=2)}

    KEY RESPONSIBILITIES:
    {json.dumps(key_responsibilities, indent=2)}
    """

    response = llm.generate(
        prompt=prompt,
        system_prompt="""You are a rubric coverage analyst.

                  You will receive:
                  - coverage_result: which
                    responsibilities are uncovered
                  - full rubric: current criteria
                  - key_responsibilities: full list

                  YOUR JOB:
                  Analyse the uncovered
                  responsibilities.
                  For each produce a flag with
                  what is missing and severity.

                  Severity rules:
                  - High: core responsibility
                    with no criterion at all
                  - Medium: partially covered
                    but not clearly linked
                  - Low: minor gap

                  Return ONLY valid JSON.
                  No explanation, no preamble,
                  no markdown code blocks.

                  Format:
                  {
                    "agent": "coverage",
                    "coverage_score": 0.00,
                    "uncovered": ["..."],
                    "flags": [
                      {
                        "responsibility": "...",
                        "issue": "...",
                        "severity": "High/Medium/Low"
                      }
                    ]
                  }""",
        max_tokens=800,
        temperature=0.1
    )

    try:
        verdict = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("coverage_agent: "
                       "failed to parse JSON response")
        verdict = {
            "agent":          "coverage",
            "coverage_score": coverage_result["coverage_score"],
            "uncovered":      coverage_result["uncovered"],
            "flags":          []
        }

    return verdict