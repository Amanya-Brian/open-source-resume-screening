"""Health check endpoints."""

from flask import Blueprint, jsonify

from src.config.settings import get_settings
from src.services.talentmatch_client import TalentMatchClient

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint.

    Returns:
        JSON response with health status
    """
    return jsonify({
        "status": "healthy",
        "service": "resume-screening-api",
    })


@health_bp.route("/api/health", methods=["GET"])
def api_health():
    """Detailed API health check.

    Returns:
        JSON response with detailed health status
    """
    settings = get_settings()

    # Check TalentMatch API connectivity
    talentmatch_healthy = False
    try:
        client = TalentMatchClient(settings)
        talentmatch_healthy = client.health_check()
    except Exception:
        pass

    return jsonify({
        "status": "healthy",
        "service": "resume-screening-api",
        "version": settings.app_version,
        "components": {
            "talentmatch_api": "healthy" if talentmatch_healthy else "unavailable",
        },
    })


@health_bp.route("/api/status", methods=["GET"])
def system_status():
    """Get detailed system status.

    Returns:
        JSON response with system status
    """
    settings = get_settings()

    return jsonify({
        "app_name": settings.app_name,
        "version": settings.app_version,
        "config": {
            "talentmatch_api_url": settings.talentmatch_api_url,
            "mongodb_database": settings.mongodb_database,
            "llm_model": settings.llm_model_name,
            "embedding_model": settings.embedding_model_name,
        },
        "thresholds": {
            "fairness_dir": settings.fairness_dir_threshold,
            "validation_agreement": settings.validation_agreement_threshold,
            "min_score": settings.screening_min_score_threshold,
        },
    })


@health_bp.route("/api/llm/status", methods=["GET"])
def llm_status():
    """Get LLM service status.

    Returns:
        JSON response with LLM status
    """
    try:
        from src.services.llm_service import LLMService
        llm = LLMService.get_instance()
        return jsonify(llm.get_model_info())
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        })


@health_bp.route("/api/llm/initialize", methods=["POST"])
def initialize_llm():
    """Initialize the LLM model.

    Returns:
        JSON response with initialization status
    """
    try:
        from src.services.llm_service import LLMService
        llm = LLMService.get_instance()
        llm.initialize()
        return jsonify({
            "status": "success",
            "message": "LLM initialized successfully",
            **llm.get_model_info(),
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500
