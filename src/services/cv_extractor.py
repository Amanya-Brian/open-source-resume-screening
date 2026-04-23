"""Structured extraction from CV text and job postings for rule-based screening."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

from src.models.scoring import CriterionScore

logger = logging.getLogger(__name__)

# ─── Degree vocabulary ────────────────────────────────────────────────────────
# Short tokens (≤5 chars) are matched with word-boundary guards to prevent false
# positives such as "msc" matching inside "kimsc" or "bsc" inside "obscure".

DEGREE_LEVELS: dict[str, list[str]] = {
    "phd":         ["ph.d", "phd", "doctorate", "doctoral", "doctor of philosophy"],
    "masters":     ["master of", "masters of", "master's", "masters'",
                    "postgraduate diploma", "postgrad diploma", "pgd",
                    "msc", "m.sc", "mba", "m.b.a", "meng", "m.eng", "llm", "mph"],
    "bachelors":   ["bachelor of", "bachelors of", "bachelor's", "bachelors'",
                    "bachelor degree", "undergraduate degree", "honours degree",
                    "bsc", "b.sc", "b.s.", "beng", "b.eng", "bcom", "b.com",
                    "llb", "mbbs", "mb bch", "mbchb", "mb,bch", "bds", "hons"],
    "diploma":     ["diploma in", "diploma of", "advanced diploma", "higher diploma",
                    "higher national diploma", "hnd", "diploma"],
    "certificate": ["tvet", "nvq", "btec", "professional certificate",
                    "certificate in", "certificate of"],
    "highschool":  ["a-level", "a level", "o-level", "o level", "gcse",
                    "high school diploma", "secondary school certificate",
                    "kcse", "wassce", "senior secondary", "national examination",
                    "secondary certificate"],
}

# Short patterns that need word-boundary matching (prevent substring false positives)
_SHORT_DEGREE_TOKENS: set[str] = {
    "phd", "msc", "mba", "bsc", "beng", "bcom", "llb", "llm", "mph",
    "hnd", "nvq", "bds", "hons", "pgd",
}

DEGREE_LEVEL_RANK: dict[str, int] = {
    "phd": 6, "masters": 5, "bachelors": 4,
    "diploma": 3, "certificate": 2, "highschool": 1,
}

# ─── Field vocabulary ─────────────────────────────────────────────────────────

FIELD_ALIASES: dict[str, list[str]] = {
    "accounting":         ["accounting", "accountancy", "accountant", "accounts",
                           "cpa", "acca", "cima", "chartered accountant"],
    "finance":            ["finance", "financial", "economics", "econom", "banking",
                           "investment", "actuarial"],
    "business":           ["business administration", "business management",
                           "management studies", "business", "commerce", "commercial",
                           "bba", "bcom", "mba"],
    "medicine":           ["medicine", "medical", "mbbs", "mbchb", "mb bch", "surgery",
                           "physician", "doctor", "clinical medicine", "health sciences",
                           "bachelor of medicine"],
    "nursing":            ["nursing", "nurse", "midwifery", "midwife", "health science",
                           "community health"],
    "pharmacy":           ["pharmacy", "pharmaceutical", "pharmacology", "pharmacist"],
    "public_health":      ["public health", "community health", "epidemiology",
                           "environmental health", "global health"],
    "engineering":        ["engineering", "engineer", "mechanical", "electrical", "civil",
                           "structural", "chemical", "industrial engineering"],
    "it":                 ["information technology", "computer science", "software",
                           "computing", "information systems", "data science",
                           "artificial intelligence", "cybersecurity"],
    "law":                ["law", "legal", "jurisprudence", "llb", "llm"],
    "education":          ["education", "teaching", "pedagogy", "educational management"],
    "hr":                 ["human resources", "human resource management", "hrm",
                           "people management", "personnel management"],
    "marketing":          ["marketing", "advertising", "brand management",
                           "public relations", "digital marketing"],
    "project_management": ["project management", "programme management",
                           "operations management"],
    "logistics":          ["logistics", "supply chain", "procurement", "purchasing",
                           "inventory management"],
    "agriculture":        ["agriculture", "agronomy", "agricultural science"],
    "social_work":        ["social work", "social sciences", "sociology", "psychology",
                           "counselling", "development studies"],
    "statistics":         ["statistics", "biostatistics", "data analysis", "mathematics"],
}

# ─── Section headers ──────────────────────────────────────────────────────────

SECTION_MAP: dict[str, list[str]] = {
    "education":      ["education", "academic", "qualifications", "educational background",
                       "academic background", "schooling", "academic qualifications",
                       "educational qualifications", "degrees and diplomas"],
    "experience":     ["experience", "work experience", "employment", "professional experience",
                       "work history", "career history", "employment history",
                       "positions held", "professional background", "career summary",
                       "professional employment"],
    "skills":         ["skills", "technical skills", "competencies", "expertise",
                       "core competencies", "key skills", "professional skills",
                       "computer skills", "it skills"],
    "certifications": ["certifications", "certificates", "licenses", "licences",
                       "professional development", "courses", "training"],
    "summary":        ["summary", "profile", "objective", "about me", "professional summary",
                       "personal statement", "career objective", "personal profile"],
    "languages":      ["languages", "language skills"],
    "achievements":   ["achievements", "accomplishments", "awards", "honors", "honours"],
    "references":     ["references", "referees"],
}

_ALL_SECTION_HEADERS: set[str] = {v for vals in SECTION_MAP.values() for v in vals}

# ─── Universal skill vocabulary ───────────────────────────────────────────────
# Only includes skills common across many job types.
# Domain-specific skills are matched via domain_keywords() which reads them
# directly from job.qualifications and job.responsibilities.

SKILL_VOCAB: list[str] = [
    # Office
    "microsoft office", "ms office", "excel", "word", "powerpoint", "outlook",
    "google workspace", "google docs", "google sheets",
    # Finance / Accounting tools
    "quickbooks", "sage", "tally", "xero", "sap", "oracle financials",
    "accounts payable", "accounts receivable", "reconciliation",
    "financial reporting", "financial analysis", "auditing", "bookkeeping",
    "payroll", "budgeting", "forecasting", "ifrs", "gaap", "erp",
    # Data / Analytics
    "power bi", "tableau", "sql", "python", "data analysis", "data visualization",
    # IT / Dev
    "java", "javascript", "html", "css", "mysql", "postgresql",
    "aws", "azure", "docker", "linux", "git",
    # CRM / Marketing
    "salesforce", "hubspot", "crm", "seo", "google ads", "social media",
    "digital marketing", "email marketing",
    # Project Management
    "project management", "agile", "scrum", "prince2", "pmp", "jira", "asana",
    # Clinical (universal across healthcare roles)
    "patient care", "clinical assessment", "emr", "ehr", "bls", "cpr",
    # HR
    "recruitment", "talent acquisition", "performance management", "hris",
]

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DegreeEntry:
    level: str
    name: str                             # full text as found in CV (≤150 chars)
    fields: list[str] = field(default_factory=list)
    institution: str = ""
    year: Optional[int] = None


@dataclass
class ExperienceEntry:
    title: str
    company: str = ""
    start_year: Optional[int] = None
    end_year: Optional[int] = None        # None = "present"
    duration_years: float = 0.0
    is_current: bool = False


@dataclass
class SectionSpan:
    name: str
    start: int      # line index of the header line
    end: int        # line index of the next header (exclusive)
    content: str    # lines[start+1 : end] joined and stripped


# ─── Text Cleaner ─────────────────────────────────────────────────────────────

class TextCleaner:
    """Normalize raw PDF text before extraction."""

    # Words that can start a wrapped continuation line in tables / columns.
    _CONTINUATION_STARTERS: frozenset = frozenset({
        "and", "or", "nor", "but",
        "in", "of", "for", "to", "the", "a", "an",
        "with", "from", "at", "by", "on", "as",
        "including", "such", "which", "that", "also",
    })

    @classmethod
    def clean(cls, text: str) -> str:
        # Repair hyphenated line breaks (common in PDFs)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

        # Collapse multiple blank lines to one
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Normalize dashes / fancy quotes
        text = re.sub(r'[–—]', '-', text)
        text = text.replace('‘', "'").replace('’', "'")
        text = text.replace('“', '"').replace('”', '"')

        # Remove non-printable characters (except newlines/tabs)
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E -￿]', ' ', text)

        # Collapse excessive spaces within a line (but keep newlines)
        lines = [re.sub(r'[ \t]{2,}', ' ', l) for l in text.splitlines()]
        text  = '\n'.join(lines).strip()

        # Rejoin wrapped continuation lines (table cells that wrap mid-entry)
        return cls._join_wrapped_lines(text)

    @classmethod
    def _join_wrapped_lines(cls, text: str) -> str:
        """Rejoin lines that are continuations of the previous line.

        In PDFs with tables or columns, a cell's content often wraps to the
        next physical line before the page margin.  For example:
            "Senior Financial Analyst"   ← first line
            "and Risk Manager"           ← wrapped tail of the same cell
        These are detected by a lowercase start or a known joining word, and
        joined back to the previous line when that line didn't end a sentence.
        """
        result: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                result.append('')
                continue

            m        = re.match(r'[a-zA-Z]+', stripped)
            fw_lower = m.group(0).lower() if m else ''

            is_continuation = (
                result
                and result[-1]                                  # non-empty previous
                and result[-1][-1] not in '.!?;:'               # prev didn't end sentence
                and (
                    stripped[0].islower()                       # starts lowercase
                    or fw_lower in cls._CONTINUATION_STARTERS  # joining word
                )
            )

            if is_continuation:
                result[-1] = result[-1] + ' ' + stripped
            else:
                result.append(stripped)

        return '\n'.join(result)


# ─── Section Parser ───────────────────────────────────────────────────────────

class CvSectionParser:
    """Span-based CV section detector with strict boundary enforcement.

    Builds positional spans (header line → next header line) so each
    extractor reads *only* the lines that belong to its section — no
    bleed-through into neighbouring sections.
    """

    @classmethod
    def parse(cls, text: str) -> dict[str, SectionSpan]:
        """Return a mapping of section_key → SectionSpan for the given CV text.

        The text must already be cleaned (pass through TextCleaner.clean first).
        """
        lines = text.splitlines()
        raw_headers: list[tuple[int, str]] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 70:
                continue

            clean = stripped.lower().rstrip(":.|-– ")

            matched: Optional[str] = None

            # 1. Exact / prefix match
            for section_key, variants in SECTION_MAP.items():
                if any(clean == v or clean.startswith(v) for v in variants):
                    matched = section_key
                    break

            # 2. Fuzzy fallback (handles "Professional Employment History" etc.)
            if not matched:
                best_ratio = 0.0
                for section_key, variants in SECTION_MAP.items():
                    for v in variants:
                        r = SequenceMatcher(None, clean, v).ratio()
                        if r > best_ratio and r >= 0.85:
                            best_ratio = r
                            matched = section_key

            if matched:
                raw_headers.append((i, matched))
                logger.debug("Section header detected: %r → %s (line %d)", stripped, matched, i)

        # Sort by position; keep only the first occurrence of each section name
        raw_headers.sort()
        seen_names: set[str] = set()
        headers: list[tuple[int, str]] = []
        for pos, name in raw_headers:
            if name not in seen_names:
                seen_names.add(name)
                headers.append((pos, name))

        spans: dict[str, SectionSpan] = {}
        for idx, (start_line, name) in enumerate(headers):
            end_line = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
            content  = "\n".join(lines[start_line + 1 : end_line]).strip()
            spans[name] = SectionSpan(name=name, start=start_line, end=end_line, content=content)
            logger.debug("Section span: %s lines %d-%d (%d chars)", name, start_line, end_line, len(content))

        return spans


# ─── CV Extractor ─────────────────────────────────────────────────────────────

class CvExtractor:
    """Extract structured data from raw CV text."""

    # ── Section parsing ───────────────────────────────────────────────────────

    @classmethod
    def split_sections(cls, text: str) -> dict[str, str]:
        """Legacy shim → delegates to CvSectionParser.  Returns {name: content}."""
        spans   = CvSectionParser.parse(text)
        result  = {name: span.content for name, span in spans.items()}
        # Preserve _preamble (lines before first header)
        lines           = text.splitlines()
        first_header    = min((s.start for s in spans.values()), default=len(lines))
        result["_preamble"] = "\n".join(lines[:first_header]).strip()
        return result

    # ── Education extraction (block-aware) ───────────────────────────────────

    @classmethod
    def extract_education(cls, text: str) -> list[DegreeEntry]:
        """Return degree entries by scanning the entire CV text.

        Does not restrict to a section — degree keywords are distinctive enough
        to find entries anywhere in the document without relying on headers.
        """
        text        = TextCleaner.clean(text)
        search_text = text
        logger.debug("extract_education: scanning full CV text (%d chars)", len(text))

        # Split into blocks (paragraphs separated by blank lines or bullet entries)
        raw_blocks = re.split(r'\n\s*\n|\n(?=[•\-\*\d])', search_text)
        # Also keep individual lines as fallback blocks
        all_blocks = raw_blocks + search_text.splitlines()

        entries: list[DegreeEntry] = []
        seen: set[str] = set()

        for block in all_blocks:
            block = block.strip()
            if not block or len(block) < 8:
                continue
            lower = block.lower()

            # ── Skip non-degree lines / blocks ─────────────────────────────

            # Skip Roman-numeral sub-headers  (IV. SPECIAL TRAINING …)
            if re.match(r'^[ivxlcdm]{1,5}[\.\)\s]', lower):
                continue

            # Skip all-caps generic header phrases without institution or year
            alpha = re.sub(r'[^a-zA-Z\s]', '', block)
            if (alpha == alpha.upper()
                    and len(alpha.split()) <= 6
                    and not cls._extract_institution(block)
                    and not cls._extract_year(block)):
                continue

            # Skip lines matching known section headers
            clean_lower = lower.rstrip(":.|-– ")
            if any(clean_lower == h or clean_lower.startswith(h)
                   for h in _ALL_SECTION_HEADERS):
                continue

            # Skip lines containing phone numbers (references leaking in)
            if re.search(r'\b\d{7,}\b', block):
                continue

            level = cls._detect_degree_level(lower)
            if not level:
                continue

            key = lower[:50]
            if key in seen:
                continue
            seen.add(key)

            # Use only the first line of the block as evidence (cleaner)
            evidence_line = block.splitlines()[0].strip()[:150]

            entries.append(DegreeEntry(
                level=level,
                name=evidence_line,
                fields=cls._extract_degree_fields(lower),
                institution=cls._extract_institution(block),
                year=cls._extract_year(block),
            ))

        entries.sort(key=lambda e: DEGREE_LEVEL_RANK.get(e.level, 0), reverse=True)
        return entries

    # ── Experience extraction ─────────────────────────────────────────────────

    _PUBLICATION_PREFIXES = (
        "perspective", "impact of", "analysis of", "study of", "assessment of",
        "effect of", "role of", "factors affecting", "evaluation of",
        "abstract", "presented at", "published in",
    )

    @classmethod
    def extract_experience(cls, text: str) -> list[ExperienceEntry]:
        """Return work experience entries by scanning the entire CV text.

        Does not restrict to a section — job-title keywords + date patterns are
        specific enough to find entries anywhere without relying on headers.
        _group_entries handles blank-line and secondary-title boundaries.
        """
        text     = TextCleaner.clean(text)
        exp_text = text
        logger.debug("extract_experience: scanning full CV text (%d chars)", len(text))

        if not exp_text.strip():
            return []

        entries: list[ExperienceEntry] = []
        seen: set[str] = set()

        for entry_lines in cls._group_entries(exp_text):
            if not entry_lines:
                continue

            title, company = cls._extract_title_company(entry_lines[0])
            if not title:
                continue

            # Company on the line immediately after the title when not inline
            if not company and len(entry_lines) > 1:
                nl = entry_lines[1]
                if (not re.search(r'\b(19[89]\d|20[012]\d)\b', nl)
                        and not nl.startswith(('-', '•', '*'))
                        and len(nl) < 80):
                    company = nl

            # Search ALL lines of this entry for a date range
            sy, ey, is_curr = None, None, False
            for line in entry_lines:
                sy, ey, is_curr = cls._extract_date_range(line)
                if sy or is_curr:
                    break

            key = title.lower()[:50]
            if key in seen:
                continue
            seen.add(key)

            logger.debug("extract_experience: entry %r @ %r  years=%s-%s current=%s",
                         title, company, sy, ey, is_curr)

            entries.append(ExperienceEntry(
                title=title,
                company=company,
                start_year=sy,
                end_year=ey,
                duration_years=cls._calc_duration(sy, ey, is_curr),
                is_current=is_curr,
            ))

        return entries

    @classmethod
    def _group_entries(cls, section_content: str) -> list[list[str]]:
        """Group section lines into per-entry lists.

        Primary boundary: blank line (most CVs).
        Secondary boundary: a new title line detected within a non-blank block
        (handles dense formatting with no blank lines between roles).
        Every line between two boundaries is kept, so dates and company names
        on lines below the title are never discarded.
        """
        entries: list[list[str]] = []
        current: list[str] = []

        for raw_line in section_content.splitlines():
            stripped = raw_line.strip()

            if not stripped:
                # Blank line → flush current entry
                if current:
                    entries.append(current)
                    current = []
                continue

            lower = stripped.lower()

            # Lines that are never entry starters (noise filters)
            is_noise = (
                re.search(r'\b\d{7,}\b', stripped)          # phone number
                or re.search(r'\S+@\S+\.\S+', stripped)     # email
                or any(lower.startswith(p) for p in cls._PUBLICATION_PREFIXES)
            )
            if is_noise:
                if current:
                    current.append(stripped)
                continue

            # Does this line look like a new job title?
            t, _ = cls._extract_title_company(stripped)
            if t and current:
                # Non-blank block with a new title: secondary split
                entries.append(current)
                current = [stripped]
            else:
                current.append(stripped)

        if current:
            entries.append(current)

        return entries

    @staticmethod
    def total_experience_years(entries: list[ExperienceEntry]) -> float:
        """Career span = latest end year - earliest start year.

        Uses span rather than sum to avoid double-counting concurrent roles.
        Entries with implausible durations (>20 yrs) are dropped.
        """
        current_year = datetime.now().year
        spans: list[tuple[int, int]] = []

        for e in entries:
            if e.start_year is None:
                continue
            end = current_year if (e.is_current or e.end_year is None) else e.end_year
            if 0 <= end - e.start_year <= 20:
                spans.append((e.start_year, end))

        if not spans:
            return 0.0
        return float(max(s[1] for s in spans) - min(s[0] for s in spans))

    # ── Skills extraction ─────────────────────────────────────────────────────

    @classmethod
    def extract_skills(cls, text: str) -> list[str]:
        lower = text.lower()
        return [s for s in SKILL_VOCAB if s in lower]

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_degree_level(lower: str) -> Optional[str]:
        """Match degree level; uses word-boundary regex for short tokens."""
        for level, patterns in DEGREE_LEVELS.items():
            for p in patterns:
                if p in _SHORT_DEGREE_TOKENS:
                    if re.search(r'(?<![a-z])' + re.escape(p) + r'(?![a-z])', lower):
                        return level
                else:
                    if p in lower:
                        return level
        return None

    @staticmethod
    def _extract_degree_fields(lower: str) -> list[str]:
        return [canon for canon, aliases in FIELD_ALIASES.items()
                if any(a in lower for a in aliases)]

    @staticmethod
    def _extract_institution(text: str) -> str:
        inst_kws = ["university", "college", "institute", "school",
                    "polytechnic", "academy", "faculty", "hospital"]
        lower = text.lower()
        for kw in inst_kws:
            idx = lower.find(kw)
            if idx != -1:
                return text[max(0, idx - 10):min(len(text), idx + 50)].strip()
        return ""

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        matches = re.findall(r"\b(19[89]\d|20[012]\d)\b", text)
        return max(int(y) for y in matches) if matches else None

    @staticmethod
    def _extract_title_company(line: str) -> tuple[str, str]:
        """Parse 'Senior Accountant at ACME Corp' → ('Senior Accountant', 'ACME Corp')."""
        TITLE_WORDS = [
            "manager", "officer", "director", "engineer", "analyst", "accountant",
            "consultant", "specialist", "coordinator", "supervisor", "lead",
            "developer", "designer", "administrator", "assistant", "associate",
            "doctor", "nurse", "physician", "pharmacist", "technician",
            "teacher", "lecturer", "professor", "midwife", "practitioner",
            "intern", "trainee", "head of", "chief", "radiographer",
            "technologist", "scientist", "researcher", "therapist",
        ]
        lower = line.lower()
        has_title_word = any(kw in lower for kw in TITLE_WORDS)

        # Pattern 1: "Title at/@ Company"
        m = re.match(r"^(.+?)\s+(?:at|@|-)\s+(.+)$", line, re.IGNORECASE)
        if m:
            t, c = m.group(1).strip(), m.group(2).strip()
            if (len(t) < 70 and len(c) < 90
                    and not re.match(r'^\d', t)       # title doesn't start with digit
                    and not re.search(r'\d{4}', t)    # title doesn't contain a year
                    and len(t.split()) >= 2
                    and (has_title_word or len(t.split()) >= 3)):
                return t, c

        # Pattern 2: "Title, Company"  (only when a title word is present)
        if has_title_word:
            m = re.match(r"^(.+?),\s+(.+)$", line)
            if m:
                t, c = m.group(1).strip(), m.group(2).strip()
                if (len(t) < 60 and len(c) < 80
                        and not re.search(r'\d{4}', t)
                        and len(t.split()) >= 2):
                    return t, c

        # Pattern 3: Line is a bare job title
        if has_title_word and len(line) < 90 and not re.search(r'\d{4}', line):
            return line.strip(), ""

        return "", ""

    @staticmethod
    def _extract_date_range(text: str) -> tuple[Optional[int], Optional[int], bool]:
        lower = text.lower()
        is_current = any(w in lower for w in
                         ["present", "current", "now", "ongoing", "to date", "till date"])
        # Multiple date patterns
        patterns = [
            r'(\w+\s+\d{4})\s*[-–]\s*(present|\w+\s+\d{4})',   # "Jan 2020 - Present"
            r'(\d{4})\s*[-–]\s*(present|\d{4})',                 # "2020 - 2023"
            r'since\s+(\d{4})',                                  # "Since 2022"
        ]
        for p in patterns:
            m = re.search(p, lower)
            if m:
                years_found = re.findall(r'\b(19[89]\d|20[012]\d)\b', m.group(0))
                if years_found:
                    sy = int(years_found[0])
                    ey = int(years_found[1]) if len(years_found) > 1 else None
                    curr = "present" in m.group(0) or "current" in m.group(0)
                    return sy, (None if curr else ey), curr

        years = re.findall(r"\b(19[89]\d|20[012]\d)\b", text)
        if len(years) >= 2:
            s, e = int(years[0]), int(years[1])
            return s, (None if is_current else e), is_current
        elif len(years) == 1:
            return int(years[0]), None, is_current
        elif is_current:
            return None, None, True
        return None, None, False

    @staticmethod
    def _calc_duration(start: Optional[int], end: Optional[int], is_current: bool) -> float:
        if start is None:
            return 0.0
        end_year = datetime.now().year if (is_current or end is None) else end
        raw = float(end_year - start)
        return raw if 0 <= raw <= 20 else 0.0   # discard implausible values


# ─── Job Extractor ────────────────────────────────────────────────────────────

class JobExtractor:
    """Extract structured requirements from job.qualifications and job.responsibilities."""

    @staticmethod
    def required_degree_level(qualifications: list[str]) -> str:
        text = " ".join(qualifications).lower()
        for level, patterns in DEGREE_LEVELS.items():
            for p in patterns:
                if p in _SHORT_DEGREE_TOKENS:
                    if re.search(r'(?<![a-z])' + re.escape(p) + r'(?![a-z])', text):
                        return level
                else:
                    if p in text:
                        return level
        return ""

    @staticmethod
    def required_degree_fields(qualifications: list[str]) -> list[str]:
        text = " ".join(qualifications).lower()
        return [canon for canon, aliases in FIELD_ALIASES.items()
                if any(a in text for a in aliases)]

    @staticmethod
    def required_experience_years(qualifications: list[str]) -> float:
        """Parse minimum years from job.qualifications."""
        text = " ".join(qualifications).lower()
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|work|relevant|clinical|practical)',
            r'minimum\s*(?:of\s*)?(\d+)\s*(?:years?|yrs?)',
            r'at\s*least\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:minimum|min)',
            r'(\d+)\+?\s*(?:years?|yrs?)',
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return float(m.group(1))
        return 0.0

    @staticmethod
    def job_domains(title: str, qualifications: list[str], responsibilities: list[str]) -> list[str]:
        """Detect job domain from title + qualifications + responsibilities."""
        text = (title + " " + " ".join(qualifications) + " " + " ".join(responsibilities)).lower()
        return [canon for canon, aliases in FIELD_ALIASES.items()
                if any(a in text for a in aliases)]

    @staticmethod
    def required_skills(qualifications: list[str], responsibilities: list[str]) -> list[str]:
        """Return SKILL_VOCAB items mentioned in job.qualifications + job.responsibilities."""
        text = (" ".join(qualifications) + " " + " ".join(responsibilities)).lower()
        return [s for s in SKILL_VOCAB if s in text]

    @staticmethod
    def domain_keywords(qualifications: list[str], responsibilities: list[str]) -> list[str]:
        """Extract meaningful domain keywords from job.qualifications and job.responsibilities.

        Used as a fallback when SKILL_VOCAB doesn't cover the job's domain.
        Keywords come entirely from the job posting — no hardcoded domain knowledge.
        """
        text = (" ".join(qualifications) + " " + " ".join(responsibilities)).lower()
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "are", "have",
            "will", "been", "they", "their", "than", "also", "both", "must",
            "able", "using", "used", "into", "which", "when", "such", "each",
            "should", "shall", "within", "strong", "good", "well", "any",
            "provide", "ensure", "maintain", "support", "work", "perform",
            "ability", "skills", "knowledge", "experience", "relevant",
        }
        words = re.findall(r"[a-zA-Z]{4,}", text)
        seen: set[str] = set()
        keywords: list[str] = []
        for w in words:
            if w not in stopwords and w not in seen:
                seen.add(w)
                keywords.append(w)
        return keywords[:40]


# ─── Rule-Based Scorer ────────────────────────────────────────────────────────

class RuleBasedScorer:
    """Score each rubric criterion deterministically.

    Scoring always uses:
      - job.qualifications  → education requirements, experience years, required skills
      - job.responsibilities → job domain, additional skill requirements
    """

    @staticmethod
    def score_education(
        cv_text: str,
        qualifications: list[str],      # job.qualifications
        criterion: dict,
    ) -> "CriterionScore":
        cv_text    = TextCleaner.clean(cv_text)
        degrees    = CvExtractor.extract_education(cv_text)
        req_level  = JobExtractor.required_degree_level(qualifications)
        req_fields = JobExtractor.required_degree_fields(qualifications)
        req_rank   = DEGREE_LEVEL_RANK.get(req_level, 3)  # default: diploma

        if not degrees:
            return CriterionScore(
                criterion_key=criterion.get("key", "education"),
                criterion_name=criterion.get("name", "Education"),
                weight=float(criterion.get("weight", 0.25)),
                raw_score=1,
                evidence="No educational qualifications found in CV",
            )

        best_score    = 1
        best_evidence = degrees[0].name

        related_pairs = {
            ("accounting", "finance"), ("accounting", "business"),
            ("finance", "business"), ("medicine", "nursing"),
            ("medicine", "public_health"), ("it", "statistics"),
            ("engineering", "it"), ("hr", "business"), ("marketing", "business"),
        }

        for deg in degrees:
            deg_rank = DEGREE_LEVEL_RANK.get(deg.level, 0)

            if req_fields:
                if any(f in deg.fields for f in req_fields):
                    field_match = "exact"
                else:
                    field_match = "none"
                    for rf in req_fields:
                        for df in deg.fields:
                            if (rf, df) in related_pairs or (df, rf) in related_pairs:
                                field_match = "related"
                                break
            else:
                field_match = "exact"   # no specific field required

            if field_match == "exact":
                if deg_rank > req_rank:    score = 5
                elif deg_rank == req_rank: score = 4
                else:                      score = 2
            elif field_match == "related":
                score = 3
            else:
                score = 2 if deg_rank >= req_rank else 1

            if score > best_score:
                best_score    = score
                best_evidence = deg.name

        return CriterionScore(
            criterion_key=criterion.get("key", "education"),
            criterion_name=criterion.get("name", "Education"),
            weight=float(criterion.get("weight", 0.25)),
            raw_score=best_score,
            evidence=best_evidence[:120],
        )

    @staticmethod
    def score_experience(
        cv_text: str,
        qualifications: list[str],      # job.qualifications
        responsibilities: list[str],    # job.responsibilities
        job_title: str,
        criterion: dict,
    ) -> "CriterionScore":
        cv_text     = TextCleaner.clean(cv_text)
        entries     = CvExtractor.extract_experience(cv_text)
        req_years   = JobExtractor.required_experience_years(qualifications)
        req_domains = JobExtractor.job_domains(job_title, qualifications, responsibilities)
        total_years = CvExtractor.total_experience_years(entries)

        # Fallback: scan raw text for explicit year mentions
        if total_years == 0:
            raw = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)', cv_text.lower())
            if raw:
                total_years = min(max(float(y) for y in raw), 40.0)

        if not entries and total_years == 0:
            return CriterionScore(
                criterion_key=criterion.get("key", "experience"),
                criterion_name=criterion.get("name", "Experience"),
                weight=float(criterion.get("weight", 0.35)),
                raw_score=1,
                evidence="No work experience found in CV",
            )

        # Find best domain-matching entry
        domain_match = "none"
        best_entry   = entries[0] if entries else None

        for entry in entries:
            t = entry.title.lower()
            for domain in req_domains:
                if any(a in t for a in FIELD_ALIASES.get(domain, [])):
                    domain_match = "exact"
                    best_entry   = entry
                    break
            if domain_match == "exact":
                break

        # Broader text scan when no entry title matched
        if domain_match == "none" and req_domains:
            cv_lower = cv_text.lower()
            for domain in req_domains:
                aliases = FIELD_ALIASES.get(domain, [])
                if sum(1 for a in aliases if a in cv_lower) >= 2:
                    domain_match = "related"
                    break

        years_met      = (req_years == 0) or (total_years >= req_years)
        years_exceeded = req_years > 0 and total_years >= req_years * 1.5

        if domain_match == "exact":
            score = 5 if years_exceeded else (4 if years_met else 3)
        elif domain_match == "related":
            score = 4 if years_exceeded else (3 if years_met else 2)
        else:
            score = 2 if total_years > 0 else 1

        yrs = f"{total_years:.0f} yrs" if total_years > 0 else "duration unclear"
        if best_entry:
            ev = best_entry.title + (f" at {best_entry.company}" if best_entry.company else "")
            evidence = f"{ev}; {yrs} total"
        else:
            evidence = f"Work history found; {yrs} total"

        return CriterionScore(
            criterion_key=criterion.get("key", "experience"),
            criterion_name=criterion.get("name", "Experience"),
            weight=float(criterion.get("weight", 0.35)),
            raw_score=score,
            evidence=evidence[:120],
        )

    @staticmethod
    def score_technical_skills(
        cv_text: str,
        qualifications: list[str],      # job.qualifications
        responsibilities: list[str],    # job.responsibilities
        criterion: dict,
    ) -> "CriterionScore":
        """Match CV skills against job.qualifications + job.responsibilities.

        Primary:  SKILL_VOCAB items that appear in both job text and CV.
        Fallback: domain keywords extracted from job text matched against CV.
        """
        cv_text    = TextCleaner.clean(cv_text)
        cv_skills  = set(CvExtractor.extract_skills(cv_text))
        req_skills = set(JobExtractor.required_skills(qualifications, responsibilities))

        if req_skills:
            matched = cv_skills & req_skills
            ratio   = len(matched) / len(req_skills)
            if ratio >= 0.75:   score = 5
            elif ratio >= 0.55: score = 4
            elif ratio >= 0.35: score = 3
            elif ratio >= 0.15: score = 2
            else:               score = 1
            evidence = (
                f"Matched {len(matched)}/{len(req_skills)} required: {', '.join(sorted(matched)[:4])}"
                if matched else
                f"None of the {len(req_skills)} required skills found in CV"
            )
        else:
            # SKILL_VOCAB didn't cover this job's domain —
            # extract keywords directly from job.qualifications + job.responsibilities
            domain_kws  = JobExtractor.domain_keywords(qualifications, responsibilities)
            cv_lower    = cv_text.lower()
            matched_kws = [kw for kw in domain_kws if kw in cv_lower]
            ratio       = len(matched_kws) / len(domain_kws) if domain_kws else 0

            if ratio >= 0.5:    score = 4
            elif ratio >= 0.3:  score = 3
            elif ratio >= 0.1:  score = 2
            else:               score = 1
            evidence = (
                f"Domain match: {len(matched_kws)}/{len(domain_kws)} job keywords in CV"
                if domain_kws else "Unable to assess technical skills"
            )

        return CriterionScore(
            criterion_key=criterion.get("key", "technical_skills"),
            criterion_name=criterion.get("name", "Technical Skills"),
            weight=float(criterion.get("weight", 0.25)),
            raw_score=score,
            evidence=evidence[:120],
        )

    @staticmethod
    def score_communication(cv_text: str, criterion: dict) -> "CriterionScore":
        cv_text = TextCleaner.clean(cv_text)

        if not cv_text or len(cv_text.strip()) < 80:
            return CriterionScore(
                criterion_key=criterion.get("key", "communication_skills"),
                criterion_name=criterion.get("name", "Communication Skills"),
                weight=float(criterion.get("weight", 0.15)),
                raw_score=1,
                evidence="Insufficient CV text to assess writing quality",
            )

        words        = re.findall(r"\b[a-zA-Z]{3,}\b", cv_text)
        unique_ratio = len(set(w.lower() for w in words)) / len(words) if words else 0

        action_verbs = [
            "achieved", "improved", "developed", "managed", "led", "designed",
            "implemented", "reduced", "increased", "collaborated", "coordinated",
            "established", "delivered", "executed", "oversaw", "streamlined",
            "spearheaded", "facilitated", "negotiated", "initiated",
            "expanded", "trained", "mentored", "supervised", "optimised",
        ]
        verb_count  = sum(1 for v in action_verbs if v in cv_text.lower())
        quant_count = len(re.findall(
            r'\d+\s*%|\$[\d,]+|\d+\s*(people|staff|team|patients|clients|projects)',
            cv_text.lower()
        ))

        sentences = [s.strip() for s in re.split(r'[.!?]+', cv_text) if len(s.strip()) > 10]
        avg_len   = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        if unique_ratio >= 0.55 and verb_count >= 6 and avg_len >= 10:
            score   = 5
            evidence = f"Excellent: {verb_count} action verbs, {quant_count} quantified results"
        elif unique_ratio >= 0.45 and verb_count >= 3:
            score   = 4
            evidence = f"Good: {verb_count} action verbs, avg {avg_len:.0f} words/sentence"
        elif unique_ratio >= 0.35 and verb_count >= 1:
            score   = 3
            evidence = f"Acceptable: {verb_count} action verbs, vocab {unique_ratio:.0%}"
        elif unique_ratio >= 0.25:
            score   = 2
            evidence = f"Basic: low vocab variety ({unique_ratio:.0%})"
        else:
            score   = 1
            evidence = f"Poor writing quality (vocab ratio {unique_ratio:.0%})"

        return CriterionScore(
            criterion_key=criterion.get("key", "communication_skills"),
            criterion_name=criterion.get("name", "Communication Skills"),
            weight=float(criterion.get("weight", 0.15)),
            raw_score=score,
            evidence=evidence[:120],
        )

    @classmethod
    def score_criterion(
        cls,
        criterion: dict,
        cv_text: str,
        qualifications: list[str],      # job.qualifications
        responsibilities: list[str],    # job.responsibilities
        job_title: str,
    ) -> "CriterionScore":
        key = criterion.get("key", "").lower()

        if "educ" in key or "qualif" in key:
            return cls.score_education(cv_text, qualifications, criterion)
        elif "exp" in key:
            return cls.score_experience(cv_text, qualifications, responsibilities, job_title, criterion)
        elif "tech" in key or "skill" in key:
            return cls.score_technical_skills(cv_text, qualifications, responsibilities, criterion)
        elif "comm" in key:
            return cls.score_communication(cv_text, criterion)
        else:
            return cls._score_generic(criterion, cv_text, qualifications, responsibilities)

    @classmethod
    def score_all(
        cls,
        cv_text: str,
        qualifications: list[str],
        responsibilities: list[str],
        job_title: str,
        criteria: list[dict],
    ) -> list[CriterionScore]:
        """Score every criterion in one pass.

        Cleans the CV text once and extracts education, experience, and skills
        once each, then reuses the results for all criteria of the same type.
        This avoids the per-criterion TextCleaner + extractor overhead.
        """
        cleaned     = TextCleaner.clean(cv_text)
        degrees     = CvExtractor.extract_education(cleaned)
        exp_entries = CvExtractor.extract_experience(cleaned)
        cv_skills   = set(CvExtractor.extract_skills(cleaned))

        results: list[CriterionScore] = []
        for criterion in criteria:
            key = criterion.get("key", "").lower()

            if "educ" in key or "qualif" in key:
                results.append(cls._education_from_data(degrees, qualifications, criterion))
            elif "exp" in key:
                results.append(cls._experience_from_data(
                    exp_entries, cleaned, qualifications, responsibilities, job_title, criterion))
            elif "tech" in key or "skill" in key:
                results.append(cls._skills_from_data(
                    cv_skills, cleaned, qualifications, responsibilities, criterion))
            elif "comm" in key:
                results.append(cls.score_communication(cleaned, criterion))
            else:
                results.append(cls._score_generic(criterion, cleaned, qualifications, responsibilities))

        return results

    # ── Inner helpers used by score_all (accept pre-extracted data) ───────────

    @staticmethod
    def _education_from_data(
        degrees: list,
        qualifications: list[str],
        criterion: dict,
    ) -> CriterionScore:
        """Education scoring that works on pre-extracted degree entries."""
        req_level  = JobExtractor.required_degree_level(qualifications)
        req_fields = JobExtractor.required_degree_fields(qualifications)
        req_rank   = DEGREE_LEVEL_RANK.get(req_level, 3)

        if not degrees:
            return CriterionScore(
                criterion_key=criterion.get("key", "education"),
                criterion_name=criterion.get("name", "Education"),
                weight=float(criterion.get("weight", 0.25)),
                raw_score=1,
                evidence="No educational qualifications found in CV",
            )

        related_pairs = {
            ("accounting", "finance"), ("accounting", "business"),
            ("finance", "business"), ("medicine", "nursing"),
            ("medicine", "public_health"), ("it", "statistics"),
            ("engineering", "it"), ("hr", "business"), ("marketing", "business"),
        }
        best_score, best_evidence = 1, degrees[0].name

        for deg in degrees:
            deg_rank = DEGREE_LEVEL_RANK.get(deg.level, 0)
            if req_fields:
                if any(f in deg.fields for f in req_fields):
                    field_match = "exact"
                else:
                    field_match = "none"
                    for rf in req_fields:
                        for df in deg.fields:
                            if (rf, df) in related_pairs or (df, rf) in related_pairs:
                                field_match = "related"
                                break
            else:
                field_match = "exact"

            if field_match == "exact":
                score = 5 if deg_rank > req_rank else (4 if deg_rank == req_rank else 2)
            elif field_match == "related":
                score = 3
            else:
                score = 2 if deg_rank >= req_rank else 1

            if score > best_score:
                best_score, best_evidence = score, deg.name

        return CriterionScore(
            criterion_key=criterion.get("key", "education"),
            criterion_name=criterion.get("name", "Education"),
            weight=float(criterion.get("weight", 0.25)),
            raw_score=best_score,
            evidence=best_evidence[:120],
        )

    @staticmethod
    def _experience_from_data(
        entries: list,
        cleaned_cv: str,
        qualifications: list[str],
        responsibilities: list[str],
        job_title: str,
        criterion: dict,
    ) -> CriterionScore:
        """Experience scoring that works on pre-extracted experience entries."""
        req_years   = JobExtractor.required_experience_years(qualifications)
        req_domains = JobExtractor.job_domains(job_title, qualifications, responsibilities)
        total_years = CvExtractor.total_experience_years(entries)

        if total_years == 0:
            raw = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)', cleaned_cv.lower())
            if raw:
                total_years = min(max(float(y) for y in raw), 40.0)

        if not entries and total_years == 0:
            return CriterionScore(
                criterion_key=criterion.get("key", "experience"),
                criterion_name=criterion.get("name", "Experience"),
                weight=float(criterion.get("weight", 0.35)),
                raw_score=1,
                evidence="No work experience found in CV",
            )

        domain_match = "none"
        best_entry   = entries[0] if entries else None

        for entry in entries:
            t = entry.title.lower()
            for domain in req_domains:
                if any(a in t for a in FIELD_ALIASES.get(domain, [])):
                    domain_match = "exact"
                    best_entry   = entry
                    break
            if domain_match == "exact":
                break

        if domain_match == "none" and req_domains:
            cv_lower = cleaned_cv.lower()
            for domain in req_domains:
                if sum(1 for a in FIELD_ALIASES.get(domain, []) if a in cv_lower) >= 2:
                    domain_match = "related"
                    break

        years_met      = req_years == 0 or total_years >= req_years
        years_exceeded = req_years > 0 and total_years >= req_years * 1.5

        if domain_match == "exact":
            score = 5 if years_exceeded else (4 if years_met else 3)
        elif domain_match == "related":
            score = 4 if years_exceeded else (3 if years_met else 2)
        else:
            score = 2 if total_years > 0 else 1

        yrs = f"{total_years:.0f} yrs" if total_years > 0 else "duration unclear"
        if best_entry:
            ev = best_entry.title + (f" at {best_entry.company}" if best_entry.company else "")
            evidence = f"{ev}; {yrs} total"
        else:
            evidence = f"Work history found; {yrs} total"

        return CriterionScore(
            criterion_key=criterion.get("key", "experience"),
            criterion_name=criterion.get("name", "Experience"),
            weight=float(criterion.get("weight", 0.35)),
            raw_score=score,
            evidence=evidence[:120],
        )

    @staticmethod
    def _skills_from_data(
        cv_skills: set,
        cleaned_cv: str,
        qualifications: list[str],
        responsibilities: list[str],
        criterion: dict,
    ) -> CriterionScore:
        """Technical skills scoring that works on a pre-extracted skills set."""
        req_skills = set(JobExtractor.required_skills(qualifications, responsibilities))

        if req_skills:
            matched = cv_skills & req_skills
            ratio   = len(matched) / len(req_skills)
            score   = 5 if ratio >= 0.75 else (4 if ratio >= 0.55 else (3 if ratio >= 0.35 else (2 if ratio >= 0.15 else 1)))
            evidence = (
                f"Matched {len(matched)}/{len(req_skills)} required: {', '.join(sorted(matched)[:4])}"
                if matched else f"None of the {len(req_skills)} required skills found in CV"
            )
        else:
            domain_kws  = JobExtractor.domain_keywords(qualifications, responsibilities)
            cv_lower    = cleaned_cv.lower()
            matched_kws = [kw for kw in domain_kws if kw in cv_lower]
            ratio       = len(matched_kws) / len(domain_kws) if domain_kws else 0
            score       = 4 if ratio >= 0.5 else (3 if ratio >= 0.3 else (2 if ratio >= 0.1 else 1))
            evidence    = (
                f"Domain match: {len(matched_kws)}/{len(domain_kws)} job keywords in CV"
                if domain_kws else "Unable to assess technical skills"
            )

        return CriterionScore(
            criterion_key=criterion.get("key", "technical_skills"),
            criterion_name=criterion.get("name", "Technical Skills"),
            weight=float(criterion.get("weight", 0.25)),
            raw_score=score,
            evidence=evidence[:120],
        )

    @staticmethod
    def _score_generic(
        criterion: dict,
        cv_text: str,
        qualifications: list[str],
        responsibilities: list[str],
    ) -> "CriterionScore":
        job_text = (" ".join(qualifications) + " " + " ".join(responsibilities)).lower()
        cv_lower = TextCleaner.clean(cv_text).lower()
        stopwords = {"the", "and", "for", "with", "from", "that", "this", "are",
                     "have", "will", "been", "they", "their", "than", "also",
                     "both", "must", "able", "using", "used", "into", "which"}
        raw      = f"{criterion.get('name', '')} {criterion.get('description', '')} {job_text}"
        words    = re.findall(r"[a-zA-Z]{4,}", raw.lower())
        keywords = list(dict.fromkeys(w for w in words if w not in stopwords))[:15]

        if not keywords:
            return CriterionScore(
                criterion_key=criterion.get("key", ""),
                criterion_name=criterion.get("name", ""),
                weight=float(criterion.get("weight", 0.1)),
                raw_score=2,
                evidence="No keywords to match",
            )

        matched  = [kw for kw in keywords if kw in cv_lower]
        ratio    = len(matched) / len(keywords)
        score    = 4 if ratio >= 0.6 else (3 if ratio >= 0.4 else (2 if ratio >= 0.2 else 1))
        evidence = (
            f"{len(matched)}/{len(keywords)} job keywords matched: {', '.join(matched[:4])}"
            if matched else "No matching keywords found in CV"
        )

        return CriterionScore(
            criterion_key=criterion.get("key", ""),
            criterion_name=criterion.get("name", ""),
            weight=float(criterion.get("weight", 0.1)),
            raw_score=score,
            evidence=evidence[:120],
        )
