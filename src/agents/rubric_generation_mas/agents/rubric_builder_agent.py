# agents/rubric_builder_agent.py

import json
import logging
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _normalize_weights(criteria: list) -> list:
    """Ensure weights sum to exactly 1.0."""
    total = sum(float(c.get("weight", 0)) for c in criteria)
    if total <= 0:
        equal = round(1.0 / len(criteria), 4) if criteria else 0
        for c in criteria:
            c["weight"] = equal
    elif abs(total - 1.0) > 0.001:
        for c in criteria:
            c["weight"] = round(float(c.get("weight", 0)) / total, 4)
    return criteria


def rubric_builder_agent(
    document_context: dict,
    key_responsibilities: list,
    current_rubric: dict = None
) -> dict:

    n = len(key_responsibilities)
    equal_weight = round(1.0 / n, 2) if n else 0.2

    responsibilities_text = "\n".join(
        f"{i+1}. {r}" for i, r in enumerate(key_responsibilities)
    )

    context_hint = ""
    if document_context.get("seniority"):
        context_hint += f"Seniority: {document_context['seniority']}. "
    if document_context.get("job_purpose"):
        context_hint += f"Role purpose: {document_context['job_purpose']}."

    llm = LLMService.get_instance()

    response = llm.generate(
        prompt=(
            f"{context_hint}\n\n"
            f"Responsibilities:\n{responsibilities_text}\n\n"
            "Build one criterion per responsibility. "
            f"Distribute weights (each ~{equal_weight}, total must be 1.0). "
            "Return ONLY valid JSON, no markdown:\n"
            '{"title":"Rubric for [role]","version":1,"total_weight":1.0,'
            '"criteria":[{"id":"C1","name":"...","description":"...","linked_responsibility":"...","weight":0.20}]}'
        ),
        system_prompt=(
            "You are a rubric builder. "
            "Return ONLY valid JSON with no explanation or markdown. "
            "One criterion per responsibility. Weights must sum to 1.0."
        ),
        max_tokens=800,
        temperature=0.1
    )

    try:
        rubric = json.loads(_strip_fences(response))
    except json.JSONDecodeError:
        logger.warning("rubric_builder_agent: failed to parse JSON — using fallback")
        rubric = {
            "title": "Draft Rubric",
            "version": 1,
            "total_weight": 1.0,
            "criteria": [
                {
                    "id": f"C{i+1}",
                    "name": f"Criterion {i+1}",
                    "description": r,
                    "linked_responsibility": r,
                    "weight": equal_weight
                }
                for i, r in enumerate(key_responsibilities)
            ]
        }

    criteria = rubric.get("criteria", [])
    rubric["criteria"] = _normalize_weights(criteria)
    rubric["total_weight"] = round(sum(c["weight"] for c in rubric["criteria"]), 4)

    return rubric
