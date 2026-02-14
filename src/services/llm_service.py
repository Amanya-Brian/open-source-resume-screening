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
                timeout=120,
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
        system_prompt = """You are an expert HR recruiter evaluating job candidates.
Analyze the candidate's qualifications against the job requirements and score each criterion.
Use a 0-5 scale where:
- 5: Exceeds requirements significantly
- 4: Exceeds requirements
- 3: Meets requirements
- 2: Partially meets requirements
- 1: Does not meet requirements
- 0: No evidence/Not applicable

You MUST respond with ONLY valid JSON in this exact format, no other text:
{
  "scores": [
    {"criterion": "education", "score": 3, "evidence": "Has Bachelor's degree in relevant field"},
    {"criterion": "experience", "score": 4, "evidence": "5+ years in similar role"},
    {"criterion": "technical_skills", "score": 3, "evidence": "Proficient in required tools"},
    {"criterion": "industry_knowledge", "score": 2, "evidence": "Some exposure to industry"},
    {"criterion": "leadership", "score": 3, "evidence": "Led small team projects"},
    {"criterion": "communication", "score": 4, "evidence": "Well-written application"}
  ],
  "strengths": ["Strong technical background", "Relevant experience"],
  "concerns": ["Limited leadership experience"]
}"""

        # Format criteria for prompt
        criteria_text = "\n".join([
            f"- {c['name']} ({int(c['weight']*100)}% weight): {c.get('description', '')}"
            for c in criteria
        ])

        # Format job requirements
        quals = job_requirements.get("qualifications", [])
        resps = job_requirements.get("responsibilities", [])
        job_text = "QUALIFICATIONS:\n" + "\n".join([f"- {q}" for q in quals[:5]])
        if resps:
            job_text += "\n\nRESPONSIBILITIES:\n" + "\n".join([f"- {r}" for r in resps[:5]])

        # Truncate candidate text if too long
        max_candidate_len = 2500
        if len(candidate_text) > max_candidate_len:
            candidate_text = candidate_text[:max_candidate_len] + "..."

        prompt = f"""Evaluate this candidate for the job position.

{job_text}

EVALUATION CRITERIA (score each 0-5):
{criteria_text}

CANDIDATE'S APPLICATION:
{candidate_text}

Respond with ONLY the JSON evaluation, no explanations:"""

        try:
            response = self.generate(prompt, system_prompt=system_prompt, max_tokens=600)
            result = self._parse_json_response(response)

            # Validate result has scores
            if not result.get("scores"):
                logger.warning("LLM response missing scores, using defaults")
                return self._default_evaluation(criteria)

            return result

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
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
Write clear, professional, and constructive feedback.
Respond with ONLY valid JSON, no other text:
{
  "summary": "2-3 sentence professional summary of the candidate",
  "strengths": ["key strength 1", "key strength 2"],
  "concerns": ["area of concern 1"],
  "interview_questions": ["suggested interview question"]
}"""

        # Format scores
        scores_text = "\n".join([
            f"- {s.get('criterion_name', s.get('criterion'))}: {s.get('raw_score', s.get('score'))}/5 ({s.get('evidence', '')})"
            for s in scores
        ])

        prompt = f"""Write an evaluation summary for:

Candidate: {candidate_name}
Position: {job_title}
Overall Score: {total_score:.2f}/5.0 ({percentage:.1f}%)
Recommendation: {recommendation}

Scores:
{scores_text}

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
        """Parse JSON from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dictionary
        """
        response = response.strip()

        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from response: {response[:200]}")
        return {}

    def _default_evaluation(self, criteria: list[dict[str, Any]]) -> dict[str, Any]:
        """Return default evaluation when LLM fails."""
        return {
            "scores": [
                {"criterion": c["key"], "score": 2, "evidence": "Evaluation unavailable"}
                for c in criteria
            ],
            "strengths": [],
            "concerns": ["Automated evaluation unavailable"],
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