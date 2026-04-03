"""Rubric management and generation API endpoints."""

import asyncio
import json
import logging
import queue
import threading
from datetime import datetime
from typing import Any
from bson import ObjectId

from flask import Blueprint, Response, jsonify, request, stream_with_context

from src.agents.rubric_generation_mas.coordinator import run_pipeline
from src.services.mongo_service import MongoService

rubrics_bp = Blueprint("rubrics", __name__)
logger = logging.getLogger(__name__)


def get_mongo_service() -> MongoService:
    """Get a fresh MongoDB service instance."""
    return MongoService()


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


# ── SSE helper ──────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── In-flight human input sessions (job_id → {event, response}) ─────────────

_human_input_sessions: dict[str, dict] = {}


# ── Generate rubric — streams every step via SSE ─────────────────────────────

@rubrics_bp.route("/rubrics/jobs/<job_id>/generate", methods=["GET"])
def generate_rubric_stream(job_id: str):
    """
    Stream rubric generation progress via Server-Sent Events.
    Each message is a JSON object with type and data fields.
    When a conflict is detected the stream emits a human_input_required event.
    The frontend submits a response via POST /rubrics/jobs/<job_id>/human-input.
    """
    progress_queue = queue.Queue()

    def on_progress(event: dict):
        progress_queue.put({"type": "step", "data": event})

    def on_human_input(conflict_context: dict, current_rubric: dict) -> dict:
        """Block pipeline thread until human responds via POST endpoint."""
        evt      = threading.Event()
        session  = {"event": evt, "response": None}
        _human_input_sessions[job_id] = session

        # push conflict to the browser
        progress_queue.put({
            "type": "human_input_required",
            "data": {
                "conflict_context": conflict_context,
                "current_rubric":   current_rubric
            }
        })

        # block until POST /human-input sets the response
        evt.wait(timeout=300)  # 5 min timeout
        _human_input_sessions.pop(job_id, None)

        response = session.get("response")
        if not response:
            # timeout — auto-approve to keep pipeline moving
            return {"approved": True, "action": "approve"}
        return response

    def run():
        try:
            result = run_pipeline(
                job_id,
                progress_callback=on_progress,
                human_input_fn=on_human_input
            )
            progress_queue.put({"type": "done", "data": result})
        except Exception as e:
            logger.error(f"generate_rubric_stream error for job {job_id}: {e}")
            progress_queue.put({"type": "error", "data": {"message": str(e)}})

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def event_stream():
        yield _sse({"type": "start", "data": {"job_id": job_id}})
        while True:
            event = progress_queue.get()
            yield _sse(event)
            if event["type"] in ("done", "error"):
                break

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        }
    )


# ── Human input response ─────────────────────────────────────────────────────

@rubrics_bp.route("/rubrics/jobs/<job_id>/human-input", methods=["POST"])
def submit_human_input(job_id: str):
    """
    Receive human decision during a running pipeline.
    Body: { "action": "tweak_weights"|"add_criterion"|"approve", ... }
    """
    session = _human_input_sessions.get(job_id)
    if not session:
        return jsonify({"success": False, "error": "No active pipeline waiting for input"}), 404

    data = request.get_json(silent=True) or {}
    session["response"] = data
    session["event"].set()
    return jsonify({"success": True})


# ── Get rubric by ID ─────────────────────────────────────────────────────────

@rubrics_bp.route("/rubrics/<rubric_id>", methods=["GET"])
def get_rubric(rubric_id: str):
    """Retrieve a rubric by its ID."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())
        doc = run_async(mongo.find_one("rubrics", {"_id": ObjectId(rubric_id)}))
        if not doc:
            return jsonify({"success": False, "error": "Rubric not found"}), 404
        doc["_id"] = str(doc["_id"])
        return jsonify({"success": True, "rubric": doc})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Update a single criterion ─────────────────────────────────────────────────

@rubrics_bp.route("/rubrics/<rubric_id>/criteria/<criterion_key>", methods=["PATCH"])
def update_rubric_criterion(rubric_id: str, criterion_key: str):
    """Update a single criterion's fields (e.g. weight) within a rubric."""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        data = request.get_json(silent=True) or {}
        allowed_fields = {"weight", "name", "description"}
        updates = {k: v for k, v in data.items() if k in allowed_fields}

        if "weight" in updates:
            try:
                updates["weight"] = round(float(updates["weight"]), 4)
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": "weight must be a number"}), 400

        if not updates:
            return jsonify({"success": False, "error": "No updatable fields provided"}), 400

        mongo = get_mongo_service()
        run_async(mongo.connect())

        rubric = run_async(mongo.find_one("rubrics", {"_id": ObjectId(rubric_id)}))
        if not rubric:
            return jsonify({"success": False, "error": "Rubric not found"}), 404

        criteria = rubric.get("criteria") or []
        updated = False
        for c in criteria:
            if c.get("key") == criterion_key:
                for field, value in updates.items():
                    c[field] = value
                updated = True
                break

        if not updated:
            return jsonify({"success": False, "error": "Criterion not found in rubric"}), 404

        total_weight = round(sum(round(float(c.get("weight") or 0.0), 4) for c in criteria), 4)
        if total_weight > 1.0:
            return jsonify({
                "success": False,
                "error": f"Total weight would be {total_weight * 100:.1f}%. It cannot exceed 100%.",
            }), 400

        run_async(mongo.update_one(
            "rubrics",
            {"_id": ObjectId(rubric_id)},
            {"$set": {"criteria": criteria, "updated_at": datetime.now()}},
        ))

        rubric["criteria"] = criteria
        rubric["_id"] = str(rubric["_id"])
        return jsonify({"success": True, "rubric": rubric})
    except Exception as e:
        logger.error(f"update_rubric_criterion error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Get rubric for a job ──────────────────────────────────────────────────────

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


# ── Update rubric ─────────────────────────────────────────────────────────────

@rubrics_bp.route("/rubrics/<rubric_id>", methods=["PUT", "PATCH"])
def update_rubric(rubric_id: str):
    """Update an existing rubric (name/description/criteria)."""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        data = request.get_json(silent=True) or {}
        update_doc: dict[str, Any] = {}

        for key in {"name", "description", "criteria"}:
            if key in data:
                update_doc[key] = data[key]

        if not update_doc:
            return jsonify({"success": False, "error": "No updatable fields provided"}), 400

        update_doc["updated_at"] = datetime.now()

        mongo = get_mongo_service()
        run_async(mongo.connect())
        modified = run_async(mongo.update_one(
            "rubrics",
            {"_id": ObjectId(rubric_id)},
            {"$set": update_doc},
        ))

        if not modified:
            return jsonify({"success": False, "error": "Rubric not found"}), 404

        updated = run_async(mongo.find_one("rubrics", {"_id": ObjectId(rubric_id)}))
        if updated:
            updated["_id"] = str(updated["_id"])
        return jsonify({"success": True, "rubric": updated})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Delete rubric ─────────────────────────────────────────────────────────────

@rubrics_bp.route("/rubrics/<rubric_id>", methods=["DELETE"])
def delete_rubric(rubric_id: str):
    """Delete a rubric and unlink it from any jobs."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())
        deleted = run_async(mongo.delete_one("rubrics", {"_id": ObjectId(rubric_id)}))
        if not deleted:
            return jsonify({"success": False, "error": "Rubric not found"}), 404
        run_async(mongo.update_many(
            "job_listings",
            {"rubric_id": rubric_id},
            {"$unset": {"rubric_id": ""}},
        ))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
