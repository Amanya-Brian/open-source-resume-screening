"""Screening API endpoints."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, jsonify, request

from src.agents.orchestrator import AgentOrchestrator
from src.models.schemas import ScreeningOptions
from src.services.mongo_service import MongoService

screening_bp = Blueprint("screening", __name__)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)
_orchestrator: AgentOrchestrator = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def get_mongo():
    mongo = MongoService()
    run_async(mongo.connect())
    return mongo


# ── Start screening pipeline ──────────────────────────────────────────────────

@screening_bp.route("/jobs/<job_id>/start", methods=["POST"])
def start_screening(job_id: str):
    try:
        data = request.get_json(silent=True) or {}
        options = ScreeningOptions(
            top_k=data.get("top_k", 10),
            min_score_threshold=data.get("min_score_threshold", 0.5),
            include_explanations=data.get("include_explanations", True),
            run_fairness_check=data.get("run_fairness_check", True),
            run_validation=data.get("run_validation", True),
        )

        orchestrator = get_orchestrator()
        result = run_async(orchestrator.run_pipeline(job_id, options))

        return jsonify({
            "success":              True,
            "session_id":           result.session.id,
            "job_id":               job_id,
            "status":               result.session.status.value,
            "total_candidates":     result.session.total_candidates,
            "processed_candidates": result.session.processed_candidates,
            "duration_seconds":     result.session.duration_seconds,
            "top_candidates": [
                {
                    "rank":           rc.rank,
                    "candidate_id":   rc.candidate_id,
                    "score":          rc.score,
                    "percentage":     round(rc.score * 100, 1),
                    "recommendation": rc.recommendation.value,
                }
                for rc in result.ranking.ranked_candidates[:options.top_k]
            ],
            "validation_agreement": result.validation_agreement,
        })

    except Exception as e:
        logger.error(f"start_screening error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ── Session status ────────────────────────────────────────────────────────────

@screening_bp.route("/sessions/<session_id>/status", methods=["GET"])
def get_session_status(session_id: str):
    status = get_orchestrator().get_session_status(session_id)
    if not status:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(status)


# ── Cancel session ────────────────────────────────────────────────────────────

@screening_bp.route("/sessions/<session_id>/cancel", methods=["POST"])
def cancel_session(session_id: str):
    cancelled = run_async(get_orchestrator().cancel_session(session_id))
    if not cancelled:
        return jsonify({"error": "Session not found or already completed"}), 400
    return jsonify({"session_id": session_id, "status": "cancelled"})


# ── Rankings from MongoDB ─────────────────────────────────────────────────────

@screening_bp.route("/jobs/<job_id>/rankings", methods=["GET"])
def get_job_rankings(job_id: str):
    try:
        limit     = request.args.get("limit", 10, type=int)
        offset    = request.args.get("offset", 0, type=int)
        min_score = request.args.get("min_score", 0.0, type=float)

        mongo = get_mongo()
        results = run_async(mongo.find_many(
            "screening_results",
            {"job_id": job_id},
            sort=[("rank", 1)],
        ))

        filtered = [
            r for r in results
            if r.get("total_weighted_score", 0) / 5.0 >= min_score
        ]
        page = filtered[offset: offset + limit]
        for r in page:
            r.pop("_id", None)

        return jsonify({
            "success":          True,
            "job_id":           job_id,
            "total_candidates": len(filtered),
            "limit":            limit,
            "offset":           offset,
            "rankings":         page,
        })

    except Exception as e:
        logger.error(f"get_job_rankings error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ── Candidate explanation ─────────────────────────────────────────────────────

@screening_bp.route("/jobs/<job_id>/candidates/<candidate_id>/explanation", methods=["GET"])
def get_candidate_explanation(job_id: str, candidate_id: str):
    try:
        mongo = get_mongo()
        doc = run_async(mongo.find_one(
            "screening_results",
            {"job_id": job_id, "candidate_id": candidate_id}
        ))

        if not doc:
            return jsonify({"success": False, "error": "No result found for this candidate"}), 404

        doc.pop("_id", None)
        return jsonify({"success": True, "explanation": doc})

    except Exception as e:
        logger.error(f"get_candidate_explanation error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ── Fairness report ───────────────────────────────────────────────────────────

@screening_bp.route("/jobs/<job_id>/fairness-report", methods=["GET"])
def get_fairness_report(job_id: str):
    try:
        mongo = get_mongo()
        doc = run_async(mongo.find_one("fairness_reports", {"job_id": job_id}))

        if not doc:
            return jsonify({"success": False, "error": "No fairness report found. Run screening first."}), 404

        doc.pop("_id", None)
        return jsonify({"success": True, "fairness_report": doc})

    except Exception as e:
        logger.error(f"get_fairness_report error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ── Validation report ─────────────────────────────────────────────────────────

@screening_bp.route("/jobs/<job_id>/validation-report", methods=["GET"])
def get_validation_report(job_id: str):
    try:
        mongo = get_mongo()
        results = run_async(mongo.find_many("screening_results", {"job_id": job_id}))

        if not results:
            return jsonify({"success": False, "error": "No screening results found. Run screening first."}), 404

        breakdown = {}
        for r in results:
            rec = r.get("recommendation", "unknown")
            breakdown[rec] = breakdown.get(rec, 0) + 1

        return jsonify({
            "success":                  True,
            "job_id":                   job_id,
            "total_screened":           len(results),
            "recommendation_breakdown": breakdown,
        })

    except Exception as e:
        logger.error(f"get_validation_report error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
