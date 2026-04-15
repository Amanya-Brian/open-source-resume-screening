"""LLM Service using Ollama for local LLM inference."""

import logging
import json
import re
from typing import Any, Optional

import requests

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based text generation using Ollama."""

    _instance: Optional["LLMService"] = None

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the LLM service.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self.ollama_url = "http://localhost:11434"
        self.model_name = getattr(self.settings, 'ollama_model', 'qwen2.5:1.5b')
        self._initialized = False
        self._available_models = []

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "LLMService":
        """Get singleton instance of LLMService."""
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Force the singleton to be recreated on next get_instance() call."""
        cls._instance = None

    def initialize(self) -> None:
        """Initialize and verify Ollama connection."""
        if self._initialized:
            return

        logger.info(f"Connecting to Ollama at {self.ollama_url}")

        try:
            # Check Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()

            data = response.json()
            self._available_models = [m["name"] for m in data.get("models", [])]

            if self.model_name not in self._available_models:
                # Try to find a matching model
                for model in self._available_models:
                    if "llama" in model.lower():
                        self.model_name = model
                        break

            logger.info(f"Ollama connected. Using model: {self.model_name}")
            logger.info(f"Available models: {', '.join(self._available_models)}")

            self._initialized = True

        except requests.exceptions.ConnectionError:
            logger.error("Ollama is not running. Start it with: ollama serve")
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 800,
        temperature: float = 0.3,
    ) -> str:
        """Generate text using Ollama.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if not self._initialized:
            self.initialize()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "format": "json",   # Constrain to JSON output — faster, no preamble
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": 1024,  # qwen2.5:1.5b fits easily; smaller = faster KV cache
                    },
                },
                timeout=60,  # qwen2.5:1.5b should respond well within 60s; fail fast otherwise
            )
            response.raise_for_status()

                if not response.ok:
                    try:
                        body = response.json()
                    except Exception:
                        body = response.text[:300]
                    logger.error(f"Ollama {response.status_code} (attempt {attempt+1}): {body}")
                    response.raise_for_status()

                data = response.json()
                return data.get("message", {}).get("content", "")

            except Exception as e:
                last_error = e
                logger.error(f"Ollama generation failed (attempt {attempt+1}): {e}")

        raise last_error

    def evaluate_candidate(
        self,
        candidate_text: str,
        job_requirements: dict[str, Any],
        criteria: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate a candidate against job requirements using LLM.

        Args:
            candidate_text: Combined resume and cover letter text
            job_requirements: Job qualifications and responsibilities
            criteria: List of evaluation criteria with weights

        Returns:
            Dictionary with scores and evidence for each criterion
        """
        # Build the exact criterion keys for the prompt
        criteria_keys = [c["key"] for c in criteria]
        criteria_keys_str = ", ".join([f'"{k}"' for k in criteria_keys])

        system_prompt = f"""You are a hiring manager shortlisting candidates using their CV/resume only.
No cover letter is available — base every judgement on documented CV content only.

FIELD CORRELATION CHECK (do this first):
Before scoring anything, determine whether the candidate's CV shows ANY correlation with the job's field/domain.
Correlation means: relevant job titles, employers in the same industry, related academic qualifications, or relevant certifications.
  - Example of correlation for a medical role: worked in a hospital/clinic/pharmacy, studied medicine/nursing/health sciences, certified in any health-related area.
  - Example of NO correlation: only IT, agriculture, finance, or unrelated trade work with no health touchpoint.
If there is NO correlation at all → all scores MUST be 0-1. Stop there.
If any correlation exists → evaluate fully and objectively.

EVALUATION ORDER (for correlated candidates):
  1. WORK EXPERIENCE — match against job responsibilities and rubric criteria. Score what is documented.
  2. EDUCATION & QUALIFICATIONS — match against required qualifications and rubric criteria.
  3. Remaining rubric criteria — score based on CV evidence only.

SCORING (0-5):
  5 = Exceptional — CV evidence clearly exceeds what the role requires
  4 = Strong — CV clearly meets and goes beyond requirements
  3 = Adequate — CV meets the core requirement
  2 = Partial — some relevant evidence but notable gaps
  1 = Weak — little to no relevant CV evidence
  0 = None — absent from CV or completely irrelevant field

STRICT RULES:
- Never infer, assume, or give benefit of the doubt. Score only documented facts.
- Scores must differ across criteria — do NOT give the same score to everything.
- An uncorrelated candidate scores 0-1 on ALL criteria, no exceptions.
- Keep evidence notes factual (max 12 words, quote or reference the CV directly).

Criterion keys to use: [{criteria_keys_str}]

Respond with ONLY valid JSON:
{{"scores": [{{"criterion": "{criteria_keys[0]}", "score": 0, "evidence": "No relevant field experience in CV"}}, {{"criterion": "{criteria_keys[1]}", "score": 2, "evidence": "Brief CV evidence"}}], "strengths": ["CV-backed strength or empty if none"], "concerns": ["Specific gap"]}}"""

        # Sort criteria: experience first, education second, rest after
        def _criterion_sort_key(c):
            name_lower = c.get('name', '').lower()
            key_lower  = c.get('key', '').lower()
            if 'experience' in name_lower or 'experience' in key_lower:
                return 0
            if 'education' in name_lower or 'education' in key_lower or 'qualification' in name_lower:
                return 1
            return 2

        sorted_criteria = sorted(criteria, key=_criterion_sort_key)
        criteria_text = "\n".join([
            f"- {c['key']} ({c['name']}, {int(c['weight']*100)}% weight): {c.get('description', 'Evaluate based on CV evidence')}"
            for c in sorted_criteria
        ])

        # Build full job context: ALL qualifications and responsibilities so the LLM
        # has complete domain context for the field correlation check.
        quals = job_requirements.get("qualifications", [])
        resps = job_requirements.get("responsibilities", [])
        job_text = ""
        if quals:
            job_text += "REQUIRED QUALIFICATIONS:\n" + "\n".join([f"- {q}" for q in quals[:10]])
        if resps:
            job_text += "\n\nKEY RESPONSIBILITIES:\n" + "\n".join([f"- {r}" for r in resps[:10]])

        # Trim CV text — 2000 chars to ensure full experience/education sections are visible
        max_candidate_len = 2000
        if len(candidate_text) > max_candidate_len:
            candidate_text = candidate_text[:max_candidate_len] + "..."

        prompt = f"""Shortlist this candidate using their CV only. No cover letter.

JOB CONTEXT:
{job_text}

RUBRIC CRITERIA (match experience and education against these):
{criteria_text}

CANDIDATE CV:
---
{candidate_text}
---

STEP 1 — FIELD CORRELATION CHECK:
Read the CV's work experience and education/qualifications sections.
Ask: Does anything in this CV relate to the job's field or domain (as described in the qualifications, responsibilities, and rubric above)?
  - YES (any correlation) → proceed to Step 2
  - NO (completely different field) → assign scores 0-1 for ALL criteria, set concerns to explain the mismatch, and output the JSON.

STEP 2 — SCORING (only if Step 1 = YES):
Score each criterion by matching CV evidence against the job qualifications, responsibilities, and rubric description.
Start with work experience, then education, then remaining criteria.
Base every score strictly on what is written in the CV above.

Respond with ONLY the JSON:"""

        try:
            # Ensure initialized (cheap no-op if already done)
            if not self._initialized:
                self.initialize()

            response = self.generate(prompt, system_prompt=system_prompt, max_tokens=500)

            logger.info(f"LLM response length: {len(response)} chars")
            logger.debug(f"LLM raw response: {response[:600]}")

            result = self._parse_json_response(response)

            if not result.get("scores"):
                logger.warning(f"LLM missing 'scores'. Keys: {list(result.keys())}. Raw: {response[:300]}")
                return self._default_evaluation(criteria)

            if len(result["scores"]) == 0:
                logger.warning("LLM returned empty scores array")
                return self._default_evaluation(criteria)

            # Log each score
            for s in result["scores"]:
                logger.info(f"  LLM: {s.get('criterion')}={s.get('score')} | {str(s.get('evidence', ''))[:80]}")

            return result

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}", exc_info=True)
            return self._default_evaluation(criteria)

    def generate_explanation(
        self,
        candidate_name: str,
        job_title: str,
        scores: list[dict[str, Any]],
        total_score: float,
        percentage: float,
        recommendation: str,
    ) -> dict[str, Any]:
        """Generate a detailed explanation for a candidate's evaluation.

        Args:
            candidate_name: Name of the candidate
            job_title: Title of the job
            scores: List of criterion scores
            total_score: Total weighted score
            percentage: Percentage score
            recommendation: Recommendation level

        Returns:
            Dictionary with summary, strengths, and concerns
        """
        system_prompt = """You are a hiring manager writing shortlisting notes based on a candidate's CV only.
Be direct — your notes inform the final hiring decision.

RULES:
- "summary" — 2-3 sentences: state clearly whether the candidate's CV justifies shortlisting, referencing their actual work experience and qualifications
- "strengths" — 2-4 SPECIFIC points from the CV (job titles, qualifications, skills listed) — no generic praise
- "concerns" — list every significant gap between the CV and role requirements; be plain if the candidate is unsuitable
- "interview_questions" — 1-2 questions to verify or probe the most critical gaps

If the overall score is below 50%, the summary MUST explicitly state the candidate is not recommended for shortlisting and explain why based on the CV.

Respond with ONLY valid JSON:
{"summary": "...", "strengths": ["..."], "concerns": ["..."], "interview_questions": ["..."]}"""

        # Format scores with more detail
        scores_text = "\n".join([
            f"- {s.get('criterion_name', s.get('criterion'))}: {s.get('raw_score', s.get('score'))}/5 - {s.get('evidence', 'No evidence')}"
            for s in scores
        ])

        prompt = f"""Write a detailed evaluation summary for this candidate.

Candidate: {candidate_name}
Position: {job_title}
Overall Score: {total_score:.2f}/5.0 ({percentage:.1f}%)
Recommendation: {recommendation}

Detailed Scores:
{scores_text}

Based on these scores, write:
1. A professional summary explaining the candidate's fit for this specific role
2. Their key strengths (be specific, reference the evidence above)
3. Areas of concern (be specific about what's missing or weak)
4. Interview questions to address the gaps

Respond with ONLY JSON:"""

        try:
            response = self.generate(prompt, system_prompt=system_prompt, max_tokens=400)
            result = self._parse_json_response(response)

            # Ensure required fields
            if not result.get("summary"):
                result["summary"] = f"{candidate_name} scored {percentage:.1f}% for {job_title}. {recommendation}."

            return result

        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            return {
                "summary": f"{candidate_name} scored {percentage:.1f}% for {job_title}. {recommendation}.",
                "strengths": [],
                "concerns": [],
            }

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common LLM output issues.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dictionary
        """
        response = response.strip()

        # Fix common LLM JSON issues before parsing
        def clean_json(text: str) -> str:
            """Clean common issues in LLM-generated JSON."""
            # Replace smart quotes with regular quotes
            text = text.replace("\u2018", "'").replace("\u2019", "'")
            text = text.replace("\u201c", '"').replace("\u201d", '"')
            # Replace curly apostrophes
            text = text.replace("\u2032", "'").replace("\u2035", "'")
            # Fix escaped single quotes inside double-quoted strings
            # Remove trailing commas before closing brackets
            text = re.sub(r',\s*}', '}', text)
            text = re.sub(r',\s*]', ']', text)
            return text

        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try with cleaned text
        cleaned = clean_json(response)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            extracted = json_match.group(0)
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                # Last resort: try to fix unescaped quotes in evidence strings
                # Replace single quotes with escaped single quotes in values
                try:
                    fixed = re.sub(r"(?<=: \")([^\"]*?)(?=\"[,}\]])", lambda m: m.group(1).replace("'", "\\'"), extracted)
                    return json.loads(fixed)
                except (json.JSONDecodeError, Exception):
                    pass

        # Try to repair truncated JSON (common when max_tokens cuts off the response)
        repaired = self._repair_truncated_json(cleaned)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from response: {response[:300]}")
        return {}

    def _repair_truncated_json(self, text: str) -> Optional[str]:
        """Attempt to repair truncated JSON by extracting complete score entries.

        When LLM output is cut off mid-response, we can still salvage the
        score entries that were fully written before truncation.

        Args:
            text: Potentially truncated JSON string

        Returns:
            Repaired JSON string or None if repair is not possible
        """
        try:
            # Find all complete score entries using regex
            score_pattern = r'\{\s*"criterion"\s*:\s*"([^"]+)"\s*,\s*"score"\s*:\s*(\d+)\s*,\s*"evidence"\s*:\s*"([^"]*?)"\s*\}'
            matches = re.findall(score_pattern, text)

            if not matches:
                return None

            scores = []
            for criterion, score, evidence in matches:
                scores.append({
                    "criterion": criterion,
                    "score": int(score),
                    "evidence": evidence[:100],  # Truncate long evidence
                })

            # Try to extract strengths and concerns
            strengths = []
            concerns = []

            strengths_match = re.search(r'"strengths"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if strengths_match:
                strengths = re.findall(r'"([^"]+)"', strengths_match.group(1))

            concerns_match = re.search(r'"concerns"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if concerns_match:
                concerns = re.findall(r'"([^"]+)"', concerns_match.group(1))

            result = {
                "scores": scores,
                "strengths": strengths[:5],
                "concerns": concerns[:5],
            }

            logger.info(f"Repaired truncated JSON: extracted {len(scores)} scores")
            return json.dumps(result)

        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
            return None

    def _default_evaluation(self, criteria: list[dict[str, Any]]) -> dict[str, Any]:
        """Return empty evaluation to signal LLM failure (triggers rule-based fallback)."""
        logger.warning("LLM evaluation failed - returning empty scores to trigger rule-based fallback")
        return {
            "scores": [],
            "strengths": [],
            "concerns": [],
        }

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if not self._initialized:
            available = self.is_available()
            return {
                "status": "available" if available else "not_running",
                "model": self.model_name,
                "ollama_url": self.ollama_url,
                "message": "Ollama ready" if available else "Start Ollama with: ollama serve",
            }

        return {
            "status": "ready",
            "model": self.model_name,
            "ollama_url": self.ollama_url,
            "available_models": self._available_models,
        }

    def set_model(self, model_name: str) -> None:
        """Change the model being used.

        Args:
            model_name: Name of the Ollama model to use
        """
        if model_name in self._available_models:
            self.model_name = model_name
            logger.info(f"Switched to model: {model_name}")
        else:
            raise ValueError(f"Model {model_name} not available. Available: {self._available_models}")