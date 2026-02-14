"""Text processing utilities."""

import re
from typing import Optional


class TextProcessor:
    """Utilities for text processing and cleaning."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,;:@\-/()'+#&]", "", text)
        return text.strip()

    @staticmethod
    def extract_years_of_experience(text: str) -> Optional[int]:
        """Extract years of experience from text.

        Args:
            text: Text containing experience information

        Returns:
            Number of years or None
        """
        patterns = [
            r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
            r"(?:experience|exp)(?:\s*:)?\s*(\d+)\+?\s*(?:years?|yrs?)",
            r"(\d+)\+?\s*(?:years?|yrs?)\s*in",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    @staticmethod
    def extract_skills_from_text(text: str, skill_keywords: list[str]) -> list[str]:
        """Extract skills from text based on keyword list.

        Args:
            text: Text to search
            skill_keywords: List of skill keywords to look for

        Returns:
            List of found skills
        """
        found_skills = []
        text_lower = text.lower()

        for skill in skill_keywords:
            # Use word boundary matching
            pattern = rf"\b{re.escape(skill.lower())}\b"
            if re.search(pattern, text_lower):
                found_skills.append(skill)

        return found_skills

    @staticmethod
    def normalize_skill_name(skill: str) -> str:
        """Normalize skill name for consistency.

        Args:
            skill: Raw skill name

        Returns:
            Normalized skill name
        """
        # Common normalizations
        normalizations = {
            "js": "JavaScript",
            "ts": "TypeScript",
            "py": "Python",
            "aws": "AWS",
            "gcp": "GCP",
            "sql": "SQL",
            "nosql": "NoSQL",
            "api": "API",
            "apis": "APIs",
            "css": "CSS",
            "html": "HTML",
            "ml": "Machine Learning",
            "ai": "Artificial Intelligence",
            "nlp": "NLP",
            "dl": "Deep Learning",
        }

        skill_lower = skill.lower().strip()
        if skill_lower in normalizations:
            return normalizations[skill_lower]

        # Capitalize properly
        return skill.strip().title()

    @staticmethod
    def truncate_text(text: str, max_length: int = 500) -> str:
        """Truncate text to maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if not text or len(text) <= max_length:
            return text

        # Try to truncate at a word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")

        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]

        return truncated.rstrip() + "..."
