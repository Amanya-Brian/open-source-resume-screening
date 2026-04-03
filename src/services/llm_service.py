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
        self.model_name = getattr(self.settings, 'ollama_model', 'llama3:latest')
        self._initialized = False
        self._available_models = []

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "LLMService":
        """Get singleton instance of LLMService."""
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

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
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()

            data = response.json()
            return data.get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

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

        system_prompt = f"""You are an expert HR recruiter evaluating job candidates.
Score each criterion on a 0-5 scale:
  5 = Exceeds significantly (exceptional match, strong evidence)
  4 = Exceeds (above requirements, clear evidence)
  3 = Meets requirements (solid match)
  2 = Partially meets (some evidence but gaps)
  1 = Does not meet (weak or no relevant evidence)
  0 = Not applicable

IMPORTANT:
- Keep evidence BRIEF (max 20 words per criterion)
- Give VARIED scores - do NOT score everything the same
- Use these exact criterion keys: [{criteria_keys_str}]

Respond with ONLY valid JSON in this format:
{{"scores": [{{"criterion": "{criteria_keys[0]}", "score": 3, "evidence": "Brief evidence"}}, {{"criterion": "{criteria_keys[1]}", "score": 4, "evidence": "Brief evidence"}}], "strengths": ["Brief strength"], "concerns": ["Brief concern"]}}"""

        # Format criteria for prompt
        criteria_text = "\n".join([
            f"- {c['key']} ({c['name']}, {int(c['weight']*100)}% weight): {c.get('description', 'Evaluate based on evidence')}"
            for c in criteria
        ])

        # Format job requirements
        quals = job_requirements.get("qualifications", [])
        resps = job_requirements.get("responsibilities", [])
        job_text = ""
        if quals:
            job_text += "REQUIRED QUALIFICATIONS:\n" + "\n".join([f"- {q}" for q in quals[:8]])
        if resps:
            job_text += "\n\nKEY RESPONSIBILITIES:\n" + "\n".join([f"- {r}" for r in resps[:8]])

        # Keep more candidate context
        max_candidate_len = 3000
        if len(candidate_text) > max_candidate_len:
            candidate_text = candidate_text[:max_candidate_len] + "..."

        prompt = f"""Evaluate this candidate for the job.

{job_text}

EVALUATION CRITERIA (score EACH one 0-5 with specific evidence):
{criteria_text}

CANDIDATE'S APPLICATION TEXT:
---
{candidate_text}
---

Score each criterion. Be specific about what you found or didn't find.
Respond with ONLY the JSON:"""

        try:
            # Ensure initialized (cheap no-op if already done)
            if not self._initialized:
                self.initialize()

            response = self.generate(prompt, system_prompt=system_prompt, max_tokens=800)

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
        system_prompt = """You are an HR professional writing candidate evaluation summaries.
Write clear, professional, and constructive feedback that helps hiring managers make decisions.

IMPORTANT:
- "summary" should be 2-3 sentences explaining WHY this candidate is/isn't a good fit
- "strengths" should be 2-4 SPECIFIC strengths with evidence (not generic)
- "concerns" should be 1-3 SPECIFIC gaps or risks with what's missing
- "interview_questions" should be 1-2 questions to probe the concerns

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