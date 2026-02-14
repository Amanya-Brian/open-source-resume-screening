"""Screening API endpoints."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flask import Blueprint, jsonify, request

from src.agents.orchestrator import AgentOrchestrator
from src.models.schemas import ScreeningOptions

screening_bp = Blueprint("screening", __name__)

# Thread pool for running async code
_executor = ThreadPoolExecutor(max_workers=4)

# Global orchestrator instance
_orchestrator: AgentOrchestrator = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@screening_bp.route("/jobs/<job_id>/start", methods=["POST"])
def start_screening(job_id: str):
    """Start a screening pipeline for a job.

    Args:
        job_id: Job listing ID

    Request Body (optional):
        {
            "top_k": 10,
            "min_score_threshold": 0.5,
            "include_explanations": true,
            "run_fairness_check": true,
            "run_validation": true
        }

    Returns:
        JSON response with session info
    """
    try:
        # Parse options from request (handle missing or empty body)
        data = {}
        if request.is_json:
            data = request.get_json(silent=True) or {}
        options = ScreeningOptions(
            top_k=data.get("top_k", 10),
            min_score_threshold=data.get("min_score_threshold", 0.5),
            include_explanations=data.get("include_explanations", True),
            run_fairness_check=data.get("run_fairness_check", True),
            run_validation=data.get("run_validation", True),
        )

        # Run pipeline
        orchestrator = get_orchestrator()
        result = run_async(orchestrator.run_pipeline(job_id, options))

        return jsonify({
            "session_id": result.session.id,
            "job_id": job_id,
            "status": result.session.status.value,
            "total_candidates": result.session.total_candidates,
            "processed_candidates": result.session.processed_candidates,
            "duration_seconds": result.session.duration_seconds,
            "top_candidates": [
                {
                    "rank": rc.rank,
                    "candidate_id": rc.candidate_id,
                    "score": rc.score,
                    "recommendation": rc.recommendation.value,
                }
                for rc in result.ranking.ranked_candidates[:options.top_k]
            ],
            "fairness": {
                "is_compliant": result.fairness_report.is_compliant if result.fairness_report else None,
                "disparate_impact_ratio": result.fairness_report.metrics.disparate_impact_ratio if result.fairness_report else None,
            } if result.fairness_report else None,
            "validation_agreement": result.validation_agreement,
        })

    except Exception as e:
        return jsonify({
            "error": "Screening failed",
            "message": str(e),
        }), 500


@screening_bp.route("/sessions/<session_id>/status", methods=["GET"])
def get_session_status(session_id: str):
    """Get status of a screening session.

    Args:
        session_id: Session ID

    Returns:
        JSON response with session status
    """
    orchestrator = get_orchestrator()
    status = orchestrator.get_session_status(session_id)

    if not status:
        return jsonify({
            "error": "Session not found",
            "session_id": session_id,
        }), 404

    return jsonify(status)


@screening_bp.route("/sessions/<session_id>/results", methods=["GET"])
def get_session_results(session_id: str):
    """Get results of a completed screening session.

    Args:
        session_id: Session ID

    Returns:
        JSON response with full results
    """
    orchestrator = get_orchestrator()
    status = orchestrator.get_session_status(session_id)

    if not status:
        return jsonify({
            "error": "Session not found",
            "session_id": session_id,
        }), 404

    if status["status"] != "completed":
        return jsonify({
            "error": "Session not completed",
            "status": status["status"],
            "progress": status["progress_percent"],
        }), 400

    # In a real implementation, fetch full results from database
    return jsonify({
        "session_id": session_id,
        "status": status["status"],
        "message": "Use /jobs/{job_id}/rankings for full results",
    })


@screening_bp.route("/sessions/<session_id>/cancel", methods=["POST"])
def cancel_session(session_id: str):
    """Cancel a running screening session.

    Args:
        session_id: Session ID

    Returns:
        JSON response with cancellation status
    """
    orchestrator = get_orchestrator()
    cancelled = run_async(orchestrator.cancel_session(session_id))

    if not cancelled:
        return jsonify({
            "error": "Could not cancel session",
            "message": "Session not found or already completed",
        }), 400

    return jsonify({
        "session_id": session_id,
        "status": "cancelled",
    })


@screening_bp.route("/jobs/<job_id>/rankings", methods=["GET"])
def get_job_rankings(job_id: str):
    """Get rankings for a job.

    Args:
        job_id: Job listing ID

    Query Parameters:
        limit: Maximum results (default 10)
        offset: Results offset (default 0)
        min_score: Minimum score filter

    Returns:
        JSON response with ranked candidates
    """
    limit = request.args.get("limit", 10, type=int)
    offset = request.args.get("offset", 0, type=int)
    min_score = request.args.get("min_score", 0.0, type=float)

    # In a real implementation, fetch from database
    # For now, return placeholder
    return jsonify({
        "job_id": job_id,
        "total_candidates": 0,
        "limit": limit,
        "offset": offset,
        "rankings": [],
        "message": "Run screening first with POST /jobs/{job_id}/start",
    })


@screening_bp.route("/jobs/<job_id>/candidates/<candidate_id>/explanation", methods=["GET"])
def get_candidate_explanation(job_id: str, candidate_id: str):
    """Get explanation for a specific candidate.

    Args:
        job_id: Job listing ID
        candidate_id: Candidate ID

    Returns:
        JSON response with candidate explanation
    """
    # In a real implementation, fetch from database
    return jsonify({
        "job_id": job_id,
        "candidate_id": candidate_id,
        "explanation": None,
        "message": "Run screening first with POST /jobs/{job_id}/start",
    })


@screening_bp.route("/jobs/<job_id>/fairness-report", methods=["GET"])
def get_fairness_report(job_id: str):
    """Get fairness report for a job screening.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with fairness report
    """
    # In a real implementation, fetch from database
    return jsonify({
        "job_id": job_id,
        "fairness_report": None,
        "message": "Run screening first with POST /jobs/{job_id}/start",
    })


@screening_bp.route("/jobs/<job_id>/validation-report", methods=["GET"])
def get_validation_report(job_id: str):
    """Get validation report for a job screening.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with validation report
    """
    # In a real implementation, fetch from database
    return jsonify({
        "job_id": job_id,
        "validation_report": None,
        "message": "Run screening first with POST /jobs/{job_id}/start",
    })
