"""Rubric management and generation API endpoints."""

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4
from bson import ObjectId

from flask import Blueprint, jsonify, request

from src.agents.base import AgentContext
from src.agents.rubric_generation_agent import (
    RubricGenerationAgent,
    RubricGenerationInput,
)
from src.models.schemas import JobListing, Rubric
from src.services.mongo_service import MongoService

rubrics_bp = Blueprint("rubrics", __name__)
logger = logging.getLogger(__name__)

_rubric_agent: RubricGenerationAgent | None = None


def get_mongo_service() -> MongoService:
    """Get a fresh MongoDB service instance."""
    return MongoService()


def get_rubric_agent() -> RubricGenerationAgent:
    """Get or create rubric generation agent."""
    global _rubric_agent
    if _rubric_agent is None:
        _rubric_agent = RubricGenerationAgent()
    return _rubric_agent


def run_async(coro):
    """Run an async coroutine safely (reuse or create event loop).

    This mirrors the pattern used in other routes to avoid 'Event loop is closed'
    issues when Motor or other async clients are reused.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@rubrics_bp.route("/rubrics/jobs/<job_id>/generate", methods=["POST"])
def generate_rubric(job_id: str):
    """Generate and persist a rubric for a given job.

    The rubric is created using the RubricGenerationAgent (LLM-backed) based on
    the job's responsibilities, requirements, and description. Progress
    milestones are returned in the response for better UI/UX feedback.
    """
    steps: list[dict[str, Any]] = []

    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        steps.append({"step": "load_job", "status": "in_progress", "message": "Loading job details"})
        job_doc = run_async(mongo.find_one("job_listings", {"_id": job_id}))
        if not job_doc:
            steps[-1]["status"] = "failed"
            steps[-1]["message"] = "Job not found"
            return jsonify({
                "success": False,
                "error": "Job not found",
                "steps": steps,
            }), 404
        steps[-1]["status"] = "completed"

        # If job already has a rubric, just return it and update steps
        existing_rubric_id = job_doc.get("rubric_id")
        if existing_rubric_id:
            steps.append({
                "step": "rubric_exists",
                "status": "completed",
                "message": "Rubric already exists for this job",
            })
            existing_rubric = run_async(mongo.find_one("rubrics", {"_id": existing_rubric_id}))
            return jsonify({
                "success": True,
                "rubric_id": existing_rubric_id,
                "job_id": job_id,
                "steps": steps,
                "rubric": existing_rubric,
            })

        job = JobListing.model_validate(job_doc)

        # Prepare and run rubric generation agent
        steps.append({
            "step": "generate_rubric",
            "status": "in_progress",
            "message": "Generating rubric from job responsibilities and description",
        })
        agent = get_rubric_agent()
        context = AgentContext(job_id=job.id, session_id=str(uuid4()))
        input_data = RubricGenerationInput(job=job)
        agent_result = run_async(agent.run(input_data, context))

        if not agent_result.success or not agent_result.data:
            steps[-1]["status"] = "failed"
            steps[-1]["message"] = "Rubric generation failed"
            return jsonify({
                "success": False,
                "error": "Rubric generation failed",
                "details": agent_result.errors if agent_result else None,
                "steps": steps,
            }), 500

        tailored_criteria = agent_result.data.criteria
        steps[-1]["status"] = "completed"

        # Persist rubric
        steps.append({
            "step": "save_rubric",
            "status": "in_progress",
            "message": "Saving rubric to database",
        })
        rubric = Rubric(
            name=f"{job.title} – Rubric",
            description=f"LLM-generated rubric for job '{job.title}' at {job.company}",
            criteria=tailored_criteria,
        )

        rubric_doc = rubric.model_dump(by_alias=True)
        # Let MongoDB generate _id to avoid duplicate '' values
        rubric_doc.pop("_id", None)
        rubric_id = run_async(mongo.insert_one("rubrics", rubric_doc))
        steps[-1]["status"] = "completed"

        # Link rubric to job
        steps.append({
            "step": "link_rubric_to_job",
            "status": "in_progress",
            "message": "Linking rubric to job listing",
        })
        run_async(mongo.update_one(
            "job_listings",
            {"_id": job_id},
            {"$set": {"rubric_id": rubric_id, "updated_at": datetime.now()}},
        ))
        steps[-1]["status"] = "completed"

        return jsonify({
            "success": True,
            "rubric_id": rubric_id,
            "job_id": job_id,
            "steps": steps,
            "rubric": {
                "name": rubric.name,
                "description": rubric.description,
                "criteria": [
                    {
                        "name": c.name,
                        "key": c.key,
                        "weight": c.weight,
                        "description": c.description,
                    }
                    for c in tailored_criteria
                ],
            },
        })

    except Exception as e:
        logger.error(f"Rubric generation error for job {job_id}: {e}")
        if steps:
            steps[-1]["status"] = "failed"
            steps[-1]["message"] = f"Error: {e}"
        return jsonify({
            "success": False,
            "error": "Rubric generation error",
            "message": str(e),
            "steps": steps,
        }), 500


@rubrics_bp.route("/rubrics/<rubric_id>", methods=["GET"])
def get_rubric(rubric_id: str):
    """Retrieve a rubric by its ID."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())
        doc = run_async(mongo.find_one("rubrics", {"_id": rubric_id}))
        if not doc:
            return jsonify({"success": False, "error": "Rubric not found"}), 404
        return jsonify({"success": True, "rubric": doc})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@rubrics_bp.route("/rubrics/jobs/<job_id>", methods=["GET"])
def get_rubric_for_job(job_id: str):
    """Retrieve the rubric associated with a given job (if any)."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())
        job = run_async(mongo.find_one("job_listings", {"_id": job_id}))
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404
        rubric_id = job.get("rubric_id")
        if not rubric_id:
            return jsonify({"success": True, "rubric": None})

        rubric = run_async(mongo.find_one("rubrics", {"_id": ObjectId(rubric_id)}))
        if not rubric:
            return jsonify({"success": False, "error": "Rubric not found"}), 404
        rubric["_id"] = str(rubric["_id"])
        return jsonify({"success": True, "rubric": rubric})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@rubrics_bp.route("/rubrics/<rubric_id>", methods=["PUT", "PATCH"])
def update_rubric(rubric_id: str):
    """Update an existing rubric (name/description/criteria)."""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        data = request.get_json(silent=True) or {}
        update_doc: dict[str, Any] = {}

        allowed_fields = {"name", "description", "criteria"}
        for key in allowed_fields:
            if key in data:
                update_doc[key] = data[key]

        if not update_doc:
            return jsonify({"success": False, "error": "No updatable fields provided"}), 400

        update_doc["updated_at"] = datetime.now()

        mongo = get_mongo_service()
        run_async(mongo.connect())
        modified = run_async(mongo.update_one(
            "rubrics",
            {"_id": rubric_id},
            {"$set": update_doc},
        ))

        if not modified:
            return jsonify({"success": False, "error": "Rubric not found"}), 404

        updated = run_async(mongo.find_one("rubrics", {"_id": rubric_id}))
        return jsonify({"success": True, "rubric": updated})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@rubrics_bp.route("/rubrics/<rubric_id>", methods=["DELETE"])
def delete_rubric(rubric_id: str):
    """Delete a rubric and unlink it from any jobs."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Delete rubric
        deleted = run_async(mongo.delete_one("rubrics", {"_id": rubric_id}))
        if not deleted:
            return jsonify({"success": False, "error": "Rubric not found"}), 404

        # Unlink from any jobs referencing it
        run_async(mongo.update_many(
            "job_listings",
            {"rubric_id": rubric_id},
            {"$unset": {"rubric_id": ""}},
        ))

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

