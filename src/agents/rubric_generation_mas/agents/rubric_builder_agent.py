# agents/rubric_builder_agent.py

import json
import logging
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)

def rubric_builder_agent(
    document_context: dict,
    key_responsibilities: list,
    current_rubric: dict = None
) -> dict:

    llm = LLMService.get_instance()

    prompt = f"""
    DOCUMENT CONTEXT:
    {json.dumps(document_context, indent=2)}

    KEY RESPONSIBILITIES:
    {json.dumps(key_responsibilities, indent=2)}

    EXISTING RUBRIC (refine if provided, else build fresh):
    {json.dumps(current_rubric, indent=2)
     if current_rubric else "null — build fresh"}
    """

    response = llm.generate(
        prompt=prompt,
        system_prompt="""You are an expert rubric builder
                  for job candidate evaluation.

                  You will receive:
                  - document_context: background
                    about the role
                  - key_responsibilities: exact
                    responsibilities to build from
                  - existing rubric: refine if given,
                    build fresh if null

                  YOUR JOB:
                  Build one criterion per
                  responsibility.
                  Use document_context to calibrate:
                  - language depth and tone
                  - seniority of scoring levels
                  - weight distribution
                    use job_purpose and competencies
                    to decide what weighs more

                  Each criterion must have:
                  - id: "C1", "C2" etc
                  - name: short clear name
                  - description: what is evaluated
                  - levels: scoring 1 to 5 with
                    observable descriptions
                  - linked_responsibility: exact
                    responsibility it maps to
                  - weight: decimal, all weights
                    must sum to exactly 1.0

                  Return ONLY valid JSON.
                  No explanation, no preamble,
                  no markdown code blocks.

                  Format:
                  {
                    "title": "Rubric for [job title]",
                    "version": 1,
                    "total_weight": 1.0,
                    "criteria": [
                      {
                        "id": "C1",
                        "name": "...",
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
                    ]
                  }""",
        max_tokens=800,
        temperature=0.2
    )

    try:
        rubric = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("rubric_builder_agent: "
                       "failed to parse JSON response")
        rubric = {}

    return rubric