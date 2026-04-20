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
                    "format": "json",
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": 2048,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

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

        system_prompt = f"""
You are a STRICT, RULE-BASED HR scoring engine.

You MUST follow a FIXED decision process. Do NOT skip steps.

========================================
STEP 0 — IDENTIFY JOB FIELD
========================================
Determine job field (e.g., medicine, accounting).

========================================
STEP 1 — EXTRACT FROM CV
========================================
Extract EXACT text only:

- education_text
- experience_text
- job_titles
- years_experience (number or null)

If missing → "No evidence found in CV"

========================================
STEP 2 — DETERMINE FIELD MATCH (FIRST)
========================================

Compare candidate vs job field:

- EXACT_MATCH → same profession (doctor ↔ doctor, accountant ↔ accountant)
- RELATED_MATCH → adjacent (nurse ↔ doctor, finance ↔ accounting)
- NO_MATCH → unrelated (logistics ↔ medicine)

========================================
STEP 3 — EDUCATION MATCH LEVEL
========================================

Check education_text:

- EXACT_MATCH → degree directly in required field
- RELATED_MATCH → similar field
- NO_MATCH → unrelated

MANDATORY OVERRIDES:
- "Bachelor of Medicine and Surgery" → EXACT_MATCH
- "Accounting" → EXACT_MATCH
- "Finance" → EXACT_MATCH

========================================
STEP 4 — EXPERIENCE MATCH LEVEL
========================================

Check:
- job_titles
- years_experience

Rules:
- SAME ROLE + enough years → EXACT_MATCH
- RELATED ROLE → RELATED_MATCH
- DIFFERENT FIELD → NO_MATCH

NUMERIC RULES:
- If years < required → NOT EXACT_MATCH
- If no years → max RELATED_MATCH

========================================
STEP 5 — CONVERT MATCH → SCORE
========================================

EDUCATION:
- EXACT_MATCH → 4-5 (or 5 if Master's+)
- RELATED_MATCH → 3-4
- NO_MATCH → 0-1

EXPERIENCE:
- EXACT_MATCH → 4–5
- RELATED_MATCH → 3
- NO_MATCH → 0-1

========================================
STEP 6 — HARD FIELD CONSTRAINT
========================================

If FIELD MATCH = NO_MATCH:
→ education ≤ 2
→ experience ≤ 2

NO EXCEPTIONS

========================================
STEP 7 — EVIDENCE RULES
========================================

- MUST be exact CV text
- <25 words
- NO job description text
- NO placeholders

If invalid:
→ "No evidence found in CV"

========================================
STEP 5b — TECHNICAL SKILLS SCORING
========================================

Check skills_text from CV:

- Skills DIRECTLY match job tools/methods → score 4–5
- Some overlap with job needs → score 2–3
- No relevant skills found → score 1
- No skills mentioned → score 0

========================================
STEP 5c — COMMUNICATION SKILLS SCORING
========================================

Read actual CV sentences and grade writing quality:

- Excellent grammar, clear and professional → score 5
- Good, minor errors only → score 4
- Acceptable, some errors but readable → score 3
- Basic, frequent errors → score 2
- Poor, hard to understand → score 1

Quote one sentence from the CV as evidence.

========================================
STEP 8 — FINAL VALIDATION
========================================

CHECK:

1. If EXACT_MATCH exists → score ≥ 4
2. If NO_MATCH → score ≤ 2
3. If years < required → experience ≤ 2
4. No invalid evidence text

If ANY violation:
→ FIX BEFORE OUTPUT

========================================
OUTPUT JSON ONLY
========================================

{{
  "scores": [
    {",\n    ".join(f'{{"criterion": "{k}", "score": 0, "evidence": "text"}}' for k in criteria_keys)}
  ],
  "strengths": ["text"],
  "concerns": ["text"]
}}
"""

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
            job_text += "REQUIRED QUALIFICATIONS:\n" + "\n".join([f"- {q}" for q in quals[:5]])
        if resps:
            job_text += "\n\nKEY RESPONSIBILITIES:\n" + "\n".join([f"- {r}" for r in resps[:5]])

        # Guard: empty candidate text means nothing to evaluate — don't hallucinate
        if not candidate_text or not candidate_text.strip():
            logger.warning("evaluate_candidate: candidate_text is empty — returning no scores to trigger rule-based fallback")
            return self._default_evaluation(criteria)

        # Extract education before trimming so we always capture the full section
        education_fact = self._extract_education_from_text(candidate_text)

        # Trim candidate text — 2500 chars gives enough content to judge grammar and skills
        max_candidate_len = 2500
        if len(candidate_text) > max_candidate_len:
            candidate_text = candidate_text[:max_candidate_len] + "..."

        prompt = f"""You are scoring a candidate's CV against a job posting.

=== JOB REQUIREMENTS (benchmark only — do NOT copy these words into evidence) ===
{job_text}
=== END JOB REQUIREMENTS ===

=== VERIFIED CANDIDATE EDUCATION (extracted directly from CV — use this exact text for education scoring) ===
{education_fact}
=== END VERIFIED EDUCATION ===

=== CANDIDATE CV (ALL evidence must come from here — read carefully) ===
{candidate_text}
=== END CANDIDATE CV ===

SCORE EACH CRITERION (use only what is written in the CV above):
{criteria_text}

BEFORE writing each score, answer these questions:
- Education: Is the candidate's degree field EXACTLY one of the fields the job requires? YES → score 4. NO but related → score 3. Unrelated → score 2. Master's/PhD in exact field → score 5.
- Experience: Do the candidate's job titles or work history belong to the SAME industry/role as this job? SAME role → score 4–5. ADJACENT → score 3. DIFFERENT field → score 1–2.
- Technical skills: Do the specific tools, technologies, or methods in the CV match what this job needs?
- Communication: Is the CV written clearly with correct grammar and professional language?
- A score of 3 = adequately meets the requirement. A score of 4 = direct match. A score of 5 = exceptional match.

Respond with ONLY the JSON:"""

        try:
            # Ensure initialized (cheap no-op if already done)
            if not self._initialized:
                self.initialize()

            response = self.generate(prompt, system_prompt=system_prompt, max_tokens=500)

            logger.info(f"LLM response length: {len(response)} chars")
            logger.debug(f"LLM raw response: {response[:600]}")

            result = self._parse_json_response(response)

            # Handle flat format: {"education": {"value": 2, ...}, "experience": {...}, ...}
            if not result.get("scores") and any(k in result for k in criteria_keys):
                converted = []
                for k in criteria_keys:
                    entry = result.get(k, {})
                    if isinstance(entry, dict):
                        score_val = entry.get("value") or entry.get("score") or 0
                        evidence  = entry.get("evidence") or entry.get("justification") or "See CV"
                    else:
                        score_val, evidence = (entry if isinstance(entry, int) else 0), "See CV"
                    converted.append({"criterion": k, "score": int(score_val), "evidence": evidence})
                logger.info(f"Converted flat LLM format to scores array ({len(converted)} criteria)")
                result = {"scores": converted, "strengths": result.get("strengths", []), "concerns": result.get("concerns", [])}

            if not result.get("scores"):
                logger.warning(f"LLM missing 'scores'. Keys: {list(result.keys())}. Raw: {response[:300]}")
                return self._default_evaluation(criteria)

            if len(result["scores"]) == 0:
                logger.warning("LLM returned empty scores array")
                return self._default_evaluation(criteria)

            # Log each score
            for s in result["scores"]:
                logger.info(f"  LLM: {s.get('criterion')}={s.get('score')} | {str(s.get('evidence', ''))[:80]}")

            # Post-process constraint: experience >= 4 implies technical_skills >= 3
            scores_by_key = {s.get("criterion"): s for s in result["scores"]}
            exp_score = scores_by_key.get("experience", {}).get("score", 0)
            tech = scores_by_key.get("technical_skills")
            if exp_score >= 4 and tech and tech.get("score", 3) < 3:
                logger.info(f"  Correcting technical_skills from {tech['score']} to 3 (experience={exp_score})")
                tech["score"] = 3

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

    @staticmethod
    def _extract_education_from_text(text: str) -> str:
        """Pull the education section out of candidate text so the LLM cannot substitute job requirements."""
        lines = text.splitlines()
        edu_lines: list[str] = []
        in_section = False

        section_headers = {"education", "qualification", "academic", "schooling", "training"}
        exit_headers    = {"experience", "employment", "work", "skill", "achievement",
                           "reference", "summary", "objective", "profile", "competenc",
                           "additional", "language", "certification", "core"}
        degree_words    = {"bachelor", "master", "phd", "doctorate", "diploma", "degree",
                           "tvet", "certificate", "o-level", "a-level", "bsc", "msc",
                           "mba", "university", "college", "institute", "a2", "high school",
                           "secondary", "primary"}

        for line in lines:
            stripped   = line.strip()
            lower      = stripped.lower()

            if not stripped:
                continue

            # Enter education section
            if any(h in lower for h in section_headers) and len(stripped) < 60:
                in_section = True
                continue

            # Exit education section when another major header appears
            if in_section and any(h in lower for h in exit_headers) and len(stripped) < 60:
                in_section = False

            if in_section and len(stripped) > 4:
                edu_lines.append(stripped)
            elif not in_section and any(w in lower for w in degree_words):
                edu_lines.append(stripped)

        if not edu_lines:
            return "No educational qualifications found in CV"

        # Deduplicate and cap
        seen: set[str] = set()
        unique: list[str] = []
        for l in edu_lines:
            if l not in seen:
                seen.add(l)
                unique.append(l)

        result = " | ".join(unique[:8])
        logger.info(f"  Education extracted: {result}")
        return result

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