# tools/pdf_reader.py

import logging
import requests
import io

logger = logging.getLogger(__name__)

def read_pdf(url: str) -> str:
    """
    Fetches PDF from URL and extracts raw text.
    No reasoning. Just reads and returns text.

    Args:
        url: Cloudinary or any PDF URL

    Returns:
        Raw text string from PDF
    """
    try:
        # fetch PDF from URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # extract text from PDF bytes
        import pdfplumber
        pdf_bytes = io.BytesIO(response.content)

        raw_text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                raw_text += (page.extract_text() or "") + "\n"

        if not raw_text.strip():
            logger.warning("pdf_reader: extracted "
                           "empty text from PDF")
            return ""

        logger.info(f"pdf_reader: extracted "
                    f"{len(raw_text)} characters "
                    f"from PDF")

        # truncate to ~3000 chars to stay within llama3 context window
        return raw_text.strip()[:3000]

    except requests.exceptions.ConnectionError:
        logger.error("pdf_reader: could not reach URL")
        raise

    except requests.exceptions.Timeout:
        logger.error("pdf_reader: request timed out")
        raise

    except Exception as e:
        logger.error(f"pdf_reader: failed — {e}")
        raise