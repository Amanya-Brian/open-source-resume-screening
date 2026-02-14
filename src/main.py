"""Main entry point for the Resume Screening API."""

import logging

from src.api.app import create_app
from src.config.logging_config import setup_logging
from src.config.settings import get_settings

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Run the Flask application."""
    settings = get_settings()

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"TalentMatch API: {settings.talentmatch_api_url}")
    logger.info(f"MongoDB: {settings.mongodb_database}")

    app = create_app()

    app.run(
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
    )


if __name__ == "__main__":
    main()
