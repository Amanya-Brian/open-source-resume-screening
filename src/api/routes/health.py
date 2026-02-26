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


@health_bp.route("/api/llm/test", methods=["POST"])
def test_llm_evaluation():
    """Test LLM evaluation with sample data.

    Returns:
        JSON response with test evaluation results
    """
    try:
        from src.services.llm_service import LLMService

        llm = LLMService.get_instance()

        # Sample candidate text
        sample_text = """
        John Doe
        Bachelor's degree in Computer Science
        5 years of experience in software development
        Skills: Python, JavaScript, React, MongoDB
        Led team of 3 developers on major project
        Excellent written and verbal communication
        """

        # Sample job requirements
        sample_job = {
            "qualifications": [
                "Bachelor's degree in relevant field",
                "3+ years experience",
                "Strong programming skills"
            ],
            "responsibilities": [
                "Develop software applications",
                "Lead development projects"
            ]
        }

        # Sample criteria
        sample_criteria = [
            {"key": "education", "name": "Education", "weight": 0.2, "description": "Degree level"},
            {"key": "experience", "name": "Experience", "weight": 0.3, "description": "Years of work"},
            {"key": "technical_skills", "name": "Technical Skills", "weight": 0.3, "description": "Programming skills"},
            {"key": "leadership", "name": "Leadership", "weight": 0.2, "description": "Team management"}
        ]

        # Run evaluation
        result = llm.evaluate_candidate(sample_text, sample_job, sample_criteria)

        return jsonify({
            "status": "success",
            "message": "LLM test evaluation completed",
            "result": result,
            "scores_count": len(result.get("scores", [])),
            "has_strengths": len(result.get("strengths", [])) > 0,
            "has_concerns": len(result.get("concerns", [])) > 0,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500
