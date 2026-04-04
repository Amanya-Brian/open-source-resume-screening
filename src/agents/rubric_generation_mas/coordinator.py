# coordinator.py

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.services.mongo_service import MongoService
from src.agents.rubric_generation_mas.agents.document_intelligence_agent import document_intelligence_agent
from src.agents.rubric_generation_mas.agents.rubric_builder_agent import rubric_builder_agent
from src.agents.rubric_generation_mas.agents.coverage_agent import coverage_agent
from src.agents.rubric_generation_mas.agents.benchmarking_agent import benchmarking_agent
from src.agents.rubric_generation_mas.agents.refiner_agent import refiner_agent

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2


# ── Async helper ─────────────────────────────────────────────────────────────

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ── DB helpers ────────────────────────────────────────────────────────────────

def fetch_job(job_id: str) -> dict:
    mongo = MongoService()
    _run_async(mongo.connect())
    job = _run_async(mongo.find_one("job_listings", {"_id": job_id}))
    if not job:
        raise ValueError(f"Job not found: {job_id}")
    return job


def save_rubric(rubric: dict, job_id: str, audit_log: list) -> str:
    mongo = MongoService()
    _run_async(mongo.connect())
    rubric_record = {
        **rubric,
        "job_id":     job_id,
        "audit_log":  audit_log,
        "created_at": datetime.now()
    }
    rubric_id = _run_async(mongo.insert_one("rubrics", rubric_record))
    _run_async(mongo.update_one(
        "job_listings",
        {"_id": job_id},
        {"$set": {"rubric_id": rubric_id}}
    ))
    logger.info(f"coordinator: rubric saved — rubric_id={rubric_id}")
    return rubric_id


# ── State ─────────────────────────────────────────────────────────────────────

def _build_state(job_id, document_context, key_responsibilities, progress_callback):
    return {
        "job_id":               job_id,
        "document_context":     document_context,
        "key_responsibilities": key_responsibilities,
        "current_rubric":       None,
        "iteration_count":      0,
        "previous_violations":  [],
        "audit_log":            [],
        "progress_callback":    progress_callback,
    }


def _log(state: dict, message: str) -> None:
    state["audit_log"].append({"iteration": state["iteration_count"], "message": message})
    logger.info(f"coordinator [{state['iteration_count']}]: {message}")
    cb = state.get("progress_callback")
    if cb:
        cb({"iteration": state["iteration_count"], "message": message})


# ── Validation loop ───────────────────────────────────────────────────────────

def _run_validation_loop(state: dict) -> None:
    while state["iteration_count"] < MAX_ITERATIONS:
        _log(state, f"validation — iteration {state['iteration_count'] + 1}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    coverage_agent,
                    state["current_rubric"],
                    state["key_responsibilities"]
                ): "coverage",
                executor.submit(
                    benchmarking_agent,
                    state["current_rubric"],
                    state["document_context"],
                    state["key_responsibilities"],
                    state["previous_violations"]
                ): "benchmarking",
            }
            coverage_verdict = benchmarking_verdict = None
            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                if name == "coverage":
                    coverage_verdict = result
                else:
                    benchmarking_verdict = result
                _log(state, f"{name} check complete")

        state["previous_violations"] = benchmarking_verdict.get("violations", [])

        # Check for high-severity issues
        high_flags = [f for f in coverage_verdict.get("flags", []) if f.get("severity") == "High"]
        high_violations = [v for v in benchmarking_verdict.get("violations", []) if v.get("severity") == "High"]

        if not high_flags and not high_violations:
            _log(state, "rubric passed validation")
            break

        if high_violations:
            _log(state, "weight issue detected — normalizing")
            state["current_rubric"] = refiner_agent(
                current_rubric=state["current_rubric"],
                improvement_brief={"issues_to_fix": [], "weight_changes": {}},
                document_context=state["document_context"],
                key_responsibilities=state["key_responsibilities"],
                human_decisions=[],
                previous_rubric_versions=[]
            )

        state["iteration_count"] += 1


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(job_id: str, progress_callback=None, human_input_fn=None) -> dict:  # human_input_fn kept for API compat
    logger.info(f"coordinator: pipeline started for job_id={job_id}")

    job = fetch_job(job_id)
    key_responsibilities = job.get("responsibilities", [])
    document_url = job.get("raw_data", {}).get("document")

    if not key_responsibilities:
        raise ValueError(f"Job {job_id} has no responsibilities")

    document_context = {}
    if document_url:
        _log({"audit_log": [], "iteration_count": 0, "progress_callback": progress_callback},
             "reading job document")
        try:
            document_context = document_intelligence_agent(document_url)
        except Exception as e:
            logger.warning(f"coordinator: document intelligence failed — {e}")

    state = _build_state(job_id, document_context, key_responsibilities, progress_callback)

    _log(state, f"{len(key_responsibilities)} responsibilities loaded")
    _log(state, "building rubric")

    state["current_rubric"] = rubric_builder_agent(
        document_context=state["document_context"],
        key_responsibilities=state["key_responsibilities"]
    )

    _log(state, f"draft rubric ready — {len(state['current_rubric'].get('criteria', []))} criteria")
    _log(state, "running validation")

    _run_validation_loop(state)

    rubric_id = save_rubric(
        rubric=state["current_rubric"],
        job_id=job_id,
        audit_log=state["audit_log"]
    )

    _log(state, f"done — rubric_id={rubric_id}")

    return {
        "status":    "success",
        "job_id":    job_id,
        "rubric_id": rubric_id,
        "rubric":    state["current_rubric"],
        "audit_log": state["audit_log"]
    }
