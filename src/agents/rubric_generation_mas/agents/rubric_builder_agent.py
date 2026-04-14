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


def _name_from_responsibility(r: str) -> str:
    """Turn a responsibility sentence into a short criterion name."""
    words = r.rstrip(".,;:").split()
    return " ".join(words[:5]) if len(words) > 5 else r


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

    # Build a short names list so the LLM only needs to assign weights + short descriptions
    names_text = "\n".join(
        f"C{i+1}: {_name_from_responsibility(r)}"
        for i, r in enumerate(key_responsibilities)
    )

    try:
        response = llm.generate(
            prompt=(
                f"Assign weights to these {n} criteria so they sum to 1.0. "
                f"Vary weights based on importance{(' for ' + document_context['seniority']) if document_context.get('seniority') else ''}.\n"
                f"{names_text}\n"
                "Return ONLY JSON: "
                '{"criteria":[{"id":"C1","weight":0.25},{"id":"C2","weight":0.20}]}'
            ),
            system_prompt="Return ONLY valid JSON. No explanation.",
            max_tokens=200,
            temperature=0.1
        )
        llm_data = json.loads(_strip_fences(response))
        # LLM returns just weights; build the full criteria list ourselves
        weights_by_id = {c["id"]: float(c.get("weight", equal_weight)) for c in llm_data.get("criteria", [])}
        criteria = [
            {
                "id": f"C{i+1}",
                "name": _name_from_responsibility(r),
                "description": r,
                "linked_responsibility": r,
                "weight": weights_by_id.get(f"C{i+1}", equal_weight),
            }
            for i, r in enumerate(key_responsibilities)
        ]
    except Exception:
        logger.warning("rubric_builder_agent: LLM failed — using fallback")
        criteria = [
            {
                "id": f"C{i+1}",
                "name": _name_from_responsibility(r),
                "description": r,
                "linked_responsibility": r,
                "weight": equal_weight,
            }
            for i, r in enumerate(key_responsibilities)
        ]

    criteria = _normalize_weights(criteria)
    rubric = {
        "title": f"Rubric for {document_context.get('seniority', 'Role')}",
        "version": 1,
        "criteria": criteria,
        "total_weight": round(sum(c["weight"] for c in criteria), 4),
    }
    return rubric
