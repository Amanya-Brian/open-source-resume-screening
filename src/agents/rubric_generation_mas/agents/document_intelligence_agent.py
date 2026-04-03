# agents/document_intelligence_agent.py

import json
import logging
from src.services.llm_service import LLMService
from src.agents.rubric_generation_mas.tools.pdf_reader import read_pdf

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def document_intelligence_agent(document_url: str) -> dict:
    raw_text = read_pdf(document_url)

    if not raw_text:
        return {"job_purpose": None, "competencies": [], "seniority": None}

    llm = LLMService.get_instance()

    response = llm.generate(
        prompt=f"Extract rubric-relevant info from this job document:\n\n{raw_text[:2000]}",
        system_prompt=(
            "Extract key info for a job rubric. "
            "Return ONLY valid JSON with these fields: "
            "job_purpose (string), competencies (array of strings), seniority (string). "
            "No explanation. No markdown."
        ),
        max_tokens=300,
        temperature=0.1
    )

    try:
        return json.loads(_strip_fences(response))
    except json.JSONDecodeError:
        logger.warning("document_intelligence_agent: failed to parse JSON")
        return {"job_purpose": None, "competencies": [], "seniority": None}
