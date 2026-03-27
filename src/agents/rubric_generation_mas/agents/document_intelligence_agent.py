# agents/document_intelligence_agent.py

import json
import logging
from src.services.llm_service import LLMService
from tools.pdf_reader import read_pdf

logger = logging.getLogger(__name__)

def document_intelligence_agent(document_url: str) -> dict:

    # TOOL reads — no reasoning
    raw_text = read_pdf(document_url)

    # AGENT thinks — no file handling
    llm = LLMService.get_instance()

    response = llm.generate(
        prompt=raw_text,
        system_prompt="""You are a job document analyst.
                  You will receive raw text from
                  a job posting document.

                  Extract ONLY what is relevant
                  for building an interview rubric:
                  - job_purpose: role summary
                    or purpose statement
                  - reports_to: who this role
                    reports to
                  - employment_type: full-time,
                    part-time etc
                  - competencies: soft skills or
                    competency requirements
                  - seniority: any seniority
                    indicator found
                  - performance_expectations: KPIs
                    or performance hints if any

                  Ignore completely:
                  - compensation and benefits
                  - application instructions
                  - company marketing language
                  - qualifications and education
                  - responsibilities

                  Return ONLY valid JSON.
                  No explanation, no preamble,
                  no markdown code blocks.
                  If a field is not found
                  return null for that field.

                  Format:
                  {
                    "job_purpose": "...",
                    "reports_to": "...",
                    "employment_type": "...",
                    "competencies": ["..."],
                    "seniority": "...",
                    "performance_expectations": "..."
                  }""",
        max_tokens=800,
        temperature=0.1
    )

    try:
        document_context = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("document_intelligence_agent: "
                       "failed to parse JSON response")
        document_context = {
            "job_purpose":               None,
            "reports_to":                None,
            "employment_type":           None,
            "competencies":              [],
            "seniority":                 None,
            "performance_expectations":  None
        }

    return document_context