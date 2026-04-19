# tools/pdf_reader.py

import io
import logging

import requests

logger = logging.getLogger(__name__)


def read_pdf(url: str) -> str:
    """Fetch a PDF from a URL and extract text.

    Tries pdfplumber first (fast, no extra deps).
    If the page yields little or no text (image-based PDF),
    falls back to Tesseract OCR via pytesseract.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_bytes = io.BytesIO(response.content)
    except requests.exceptions.ConnectionError:
        logger.error("pdf_reader: could not reach URL")
        raise
    except requests.exceptions.Timeout:
        logger.error("pdf_reader: request timed out")
        raise
    except Exception as e:
        logger.error(f"pdf_reader: download failed — {e}")
        raise

    raw_text = _extract_with_pdfplumber(pdf_bytes)

    if len(raw_text.strip()) < 100:
        logger.info("pdf_reader: little text from pdfplumber — trying OCR")
        pdf_bytes.seek(0)
        raw_text = _extract_with_ocr(pdf_bytes) or raw_text

    result = raw_text.strip()[:3000]
    logger.info(f"pdf_reader: extracted {len(result)} characters")
    return result


def _extract_with_pdfplumber(pdf_bytes: io.BytesIO) -> str:
    """Extract text using pdfplumber (text-layer PDFs)."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        logger.warning(f"pdf_reader: pdfplumber failed — {e}")
        return ""


def _extract_with_ocr(pdf_bytes: io.BytesIO) -> str:
    """Extract text from image-based PDFs using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                # Render page to a PIL image (150 DPI is enough for OCR)
                img = page.to_image(resolution=150).original
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                page_text = pytesseract.image_to_string(img, lang="eng")
                if page_text.strip():
                    text_parts.append(page_text)

        result = "\n".join(text_parts)
        logger.info(f"pdf_reader: OCR extracted {len(result)} characters")
        return result

    except ImportError:
        logger.warning("pdf_reader: pytesseract/Pillow not installed — OCR skipped")
        return ""
    except Exception as e:
        logger.warning(f"pdf_reader: OCR failed — {e}")
        return ""
