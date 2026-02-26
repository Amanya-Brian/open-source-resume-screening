"""Rubric management API endpoints."""

import asyncio
import logging
from flask import Blueprint, jsonify, request

from src.services.mongo_service import MongoService
from src.services.rubric_generator import RubricGenerator
from src.config.settings import get_settings

rubric_bp = Blueprint("rubric", __name__)
logger = logging.getLogger(__name__)


def run_async(coro):
    """Run an async coroutine safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def get_mongo_service() -> MongoService:
    """Get a fresh MongoDB service instance."""
    return MongoService()


@rubric_bp.route("/jobs/<job_id>/rubric/generate", methods=["POST"])
def generate_rubric(job_id: str):
    """Generate a custom rubric for a job.

    Args:
        job_id: Job listing ID

    Request Body (optional):
        {
            "num_criteria": 6,
            "regenerate": false
        }

    Returns:
        JSON response with generated rubric
    """
    try:
        # Get request parameters
        data = request.get_json() if request.is_json else {}
        num_criteria = data.get("num_criteria", 6)
        regenerate = data.get("regenerate", False)

        # Connect to MongoDB
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Get job details
        job = run_async(mongo.find_one("job_listings", {"_id": job_id}))
        if not job:
            return jsonify({
                "error": "Job not found",
                "job_id": job_id,
            }), 404

        # Check if rubric already exists and approved
        existing_rubric = job.get("rubric")
        if existing_rubric and existing_rubric.get("approved") and not regenerate:
            return jsonify({
                "success": True,
                "message": "Rubric already exists and is approved",
                "rubric": existing_rubric,
                "job_id": job_id,
            })

        # Generate new rubric
        generator = RubricGenerator()
        rubric = generator.generate_rubric(
            job_title=job.get("title", ""),
            job_description=job.get("description", ""),
            qualifications=job.get("qualifications", []),
            responsibilities=job.get("responsibilities", []),
            num_criteria=num_criteria,
        )

        # Add metadata
        rubric["approved"] = False
        rubric["version"] = existing_rubric.get("version", 0) + 1 if existing_rubric else 1
        rubric["generated_at"] = None  # Will be set when saved

        # Save draft rubric to job
        run_async(mongo.update_one(
            "job_listings",
            {"_id": job_id},
            {"$set": {"rubric": rubric}}
        ))

        logger.info(f"Generated rubric for job {job_id} with {len(rubric['criteria'])} criteria")

        return jsonify({
            "success": True,
            "message": "Rubric generated successfully. Review and approve before screening.",
            "rubric": rubric,
            "job_id": job_id,
            "job_title": job.get("title"),
        })

    except Exception as e:
        logger.error(f"Rubric generation failed: {e}", exc_info=True)
        return jsonify({
            "error": "Rubric generation failed",
            "message": str(e),
        }), 500


@rubric_bp.route("/jobs/<job_id>/rubric", methods=["GET"])
def get_rubric(job_id: str):
    """Get the rubric for a job.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with rubric
    """
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        job = run_async(mongo.find_one("job_listings", {"_id": job_id}))
        if not job:
            return jsonify({
                "error": "Job not found",
                "job_id": job_id,
            }), 404

        rubric = job.get("rubric")
        if not rubric:
            return jsonify({
                "success": False,
                "message": "No rubric found. Generate one first.",
                "job_id": job_id,
            }), 404

        return jsonify({
            "success": True,
            "rubric": rubric,
            "job_id": job_id,
            "job_title": job.get("title"),
        })

    except Exception as e:
        logger.error(f"Get rubric failed: {e}")
        return jsonify({
            "error": "Failed to get rubric",
            "message": str(e),
        }), 500


@rubric_bp.route("/jobs/<job_id>/rubric/approve", methods=["POST"])
def approve_rubric(job_id: str):
    """Approve or update a rubric for a job.

    Args:
        job_id: Job listing ID

    Request Body:
        {
            "rubric": {
                "criteria": [...],
                "rationale": "..."
            },
            "approved": true
        }

    Returns:
        JSON response with approval status
    """
    try:
        data = request.get_json()
        if not data or "rubric" not in data:
            return jsonify({
                "error": "Missing rubric in request body",
            }), 400

        rubric = data["rubric"]
        approved = data.get("approved", True)

        # Validate rubric structure
        if not isinstance(rubric, dict) or "criteria" not in rubric:
            return jsonify({
                "error": "Invalid rubric structure",
            }), 400

        criteria = rubric["criteria"]
        if not isinstance(criteria, list) or len(criteria) < 3:
            return jsonify({
                "error": "Rubric must have at least 3 criteria",
            }), 400

        # Normalize weights
        generator = RubricGenerator()
        rubric = generator._normalize_weights(rubric)

        # Add approval metadata
        rubric["approved"] = approved
        from datetime import datetime
        rubric["approved_at"] = datetime.utcnow().isoformat() if approved else None

        # Save to MongoDB
        mongo = get_mongo_service()
        run_async(mongo.connect())

        result = run_async(mongo.update_one(
            "job_listings",
            {"_id": job_id},
            {"$set": {"rubric": rubric}}
        ))

        if result.modified_count == 0:
            return jsonify({
                "error": "Job not found or rubric unchanged",
                "job_id": job_id,
            }), 404

        logger.info(f"Rubric {'approved' if approved else 'saved'} for job {job_id}")

        return jsonify({
            "success": True,
            "message": f"Rubric {'approved' if approved else 'saved'} successfully",
            "rubric": rubric,
            "job_id": job_id,
        })

    except Exception as e:
        logger.error(f"Approve rubric failed: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to approve rubric",
            "message": str(e),
        }), 500


@rubric_bp.route("/jobs/<job_id>/rubric/status", methods=["GET"])
def rubric_status(job_id: str):
    """Get rubric status for a job.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with rubric status
    """
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        job = run_async(mongo.find_one("job_listings", {"_id": job_id}))
        if not job:
            return jsonify({
                "error": "Job not found",
                "job_id": job_id,
            }), 404

        rubric = job.get("rubric")

        if not rubric:
            status = "not_generated"
            message = "No rubric exists. Generate one before screening."
            can_screen = False
        elif not rubric.get("approved", False):
            status = "pending_approval"
            message = "Rubric generated but not yet approved. Review and approve before screening."
            can_screen = False
        else:
            status = "approved"
            message = "Rubric approved and ready for screening."
            can_screen = True

        return jsonify({
            "success": True,
            "status": status,
            "message": message,
            "can_screen": can_screen,
            "has_rubric": rubric is not None,
            "approved": rubric.get("approved", False) if rubric else False,
            "criteria_count": len(rubric.get("criteria", [])) if rubric else 0,
            "job_id": job_id,
        })

    except Exception as e:
        logger.error(f"Rubric status check failed: {e}")
        return jsonify({
            "error": "Failed to check rubric status",
            "message": str(e),
        }), 500