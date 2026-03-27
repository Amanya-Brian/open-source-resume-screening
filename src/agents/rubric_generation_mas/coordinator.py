# coordinator.py

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.services.mongo_service import MongoService

from agents.document_intelligence_agent import document_intelligence_agent
from agents.rubric_builder_agent import rubric_builder_agent
from agents.coverage_agent import coverage_agent
from agents.benchmarking_agent import benchmarking_agent
from agents.refiner_agent import refiner_agent
from tools.weight_validator import weight_validator
from tools.rubric_versioner import rubric_versioner
from tools.human_input import request_human_input

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# CONFIG — lives here, no separate file needed
# ═══════════════════════════════════════════════

MAX_ITERATIONS = 3

PRINCIPLES = [
    "Clarity: every criterion must be unambiguous. "
    "Different evaluators must reach same interpretation.",

    "Measurability: each criterion must describe "
    "something observable not a feeling or assumption.",

    "Non-overlap: no two criteria should measure "
    "the same thing from different angles.",

    "Traceability: every criterion must map back "
    "to at least one responsibility.",

    "Balance: weights must reflect actual importance "
    "of responsibilities not be uniform by default.",

    "Completeness: no major responsibility should "
    "be left without a criterion.",

    "Fairness: no criterion should disadvantage "
    "a candidate for reasons unrelated to job performance."
]


# ═══════════════════════════════════════════════
# DB CALLS — lives here, no separate file needed
# ═══════════════════════════════════════════════

def fetch_job(job_id: str) -> dict:
    """
    Fetch job record from database by job_id.
    Returns job dict with responsibilities
    and document URL.
    """
    mongo = MongoService()
    loop  = asyncio.new_event_loop()
    loop.run_until_complete(mongo.connect())
    job   = loop.run_until_complete(mongo.find_one("job_listings", {"_id": job_id}))
    loop.close()

    if not job:
        raise ValueError(f"Job not found: {job_id}")

    return job


def save_rubric(
    rubric: dict,
    job_id: str,
    audit_log: list
) -> str:
    """
    Save approved rubric to rubrics collection.
    Update job record with rubric_id reference.
    Returns rubric_id.
    """
    from src.db.mongodb import get_database
    from datetime import datetime
    from bson import ObjectId

    db = get_database()

    rubric_record = {
        "rubric":      rubric,
        "job_id":      job_id,
        "audit_log":   audit_log,
        "created_at":  datetime.utcnow()
    }

    result    = db["rubrics"].insert_one(rubric_record)
    rubric_id = str(result.inserted_id)

    db["job_listings"].update_one(
        {"_id": job_id},
        {"$set": {"rubric_id": rubric_id}}
    )

    logger.info(f"coordinator: rubric saved — rubric_id={rubric_id} job_id={job_id}")

    return rubric_id


# ═══════════════════════════════════════════════
# STATE — initialized fresh per pipeline run
# ═══════════════════════════════════════════════

def build_state(
    job_id: str,
    document_context: dict,
    key_responsibilities: list
) -> dict:
    return {
        "job_id":                   job_id,
        "document_context":         document_context,
        "key_responsibilities":     key_responsibilities,
        "current_rubric":           None,
        "agent_verdicts":           {},
        "human_decisions":          [],
        "iteration_count":          0,
        "previous_violations":      [],
        "previous_rubric_versions": [],
        "audit_log":                []
    }


# ═══════════════════════════════════════════════
# INTERNAL FUNCTIONS
# ═══════════════════════════════════════════════

def _log(state: dict, message: str) -> None:
    """
    Appends message to audit log
    and logs to logger.
    """
    state["audit_log"].append({
        "iteration": state["iteration_count"],
        "message":   message
    })
    logger.info(f"coordinator [{state['iteration_count']}]: {message}")


def _disagreement_detector(
    coverage_verdict: dict,
    benchmarking_verdict: dict
) -> dict:
    """
    Checks if same criterion flagged by both agents
    or any High severity violation exists.
    Pure logic — no LLM.

    Returns:
        dict with has_conflict bool and conflicts list
    """

    conflicts = []

    coverage_flagged = {
        f.get("responsibility", "")
        for f in coverage_verdict.get("flags", [])
    }

    benchmarking_flagged = {
        v.get("criterion_affected", "")
        for v in benchmarking_verdict.get("violations", [])
    }

    overlap = coverage_flagged & benchmarking_flagged
    if overlap:
        conflicts.append({
            "type":         "both_agents_flagged",
            "items":        list(overlap),
            "coverage":     coverage_verdict.get("flags", []),
            "benchmarking": benchmarking_verdict.get("violations", [])
        })

    high_severity = [
        v for v in benchmarking_verdict.get("violations", [])
        if v.get("severity") == "High"
    ]
    if high_severity:
        conflicts.append({
            "type":       "high_severity",
            "violations": high_severity
        })

    high_coverage = [
        f for f in coverage_verdict.get("flags", [])
        if f.get("severity") == "High"
    ]
    if high_coverage:
        conflicts.append({
            "type":  "high_severity_coverage",
            "flags": high_coverage
        })

    return {
        "has_conflict": len(conflicts) > 0,
        "conflicts":    conflicts
    }


def _evaluation_synthesizer(
    coverage_verdict: dict,
    benchmarking_verdict: dict,
    human_decisions: list
) -> dict:
    """
    Combines all agent findings and human decisions
    into one improvement brief for the Refiner.
    Pure logic — no LLM.

    Returns:
        improvement_brief dict with issues and
        weight_changes. Empty if nothing to fix.
    """

    issues_to_fix  = []
    weight_changes = {}

    for flag in coverage_verdict.get("flags", []):
        issues_to_fix.append({
            "source":         "coverage_agent",
            "responsibility": flag.get("responsibility"),
            "issue":          flag.get("issue"),
            "severity":       flag.get("severity")
        })

    for violation in benchmarking_verdict.get("violations", []):
        issues_to_fix.append({
            "source":             "benchmarking_agent",
            "criterion_affected": violation.get("criterion_affected"),
            "principle_violated": violation.get("principle_violated"),
            "reason":             violation.get("reason"),
            "severity":           violation.get("severity"),
            "suggestion":         violation.get("suggestion")
        })

    for decision in human_decisions:

        if decision.get("action") == "tweak_weights":
            for change in decision.get("changes", []):
                weight_changes[change["criterion_id"]] = change["new_weight"]

        if decision.get("action") == "add_criterion":
            issues_to_fix.append({
                "source":    "human",
                "action":    "add_criterion",
                "criterion": decision.get("criterion")
            })

    return {
        "issues_to_fix":  issues_to_fix,
        "weight_changes": weight_changes
    }


def _apply_human_weight_decision(
    state: dict,
    human_response: dict
) -> None:
    """
    Applies human weight tweaks directly
    to current rubric in STATE.
    Validates weights after applying.
    """
    if human_response.get("action") != "tweak_weights":
        return

    for change in human_response.get("changes", []):
        for criterion in state["current_rubric"]["criteria"]:
            if criterion["id"] == change["criterion_id"]:
                criterion["weight"] = change["new_weight"]
                _log(state, f"human tweaked weight: {change['criterion_id']} → {change['new_weight']}")

    weight_check = weight_validator(state["current_rubric"]["criteria"])
    if not weight_check["valid"]:
        _log(state, f"weight warning after human tweak: {weight_check['issues']}")


def _apply_human_add_criterion(
    state: dict,
    human_response: dict
) -> None:
    """
    Adds human-defined criterion directly
    to current rubric in STATE.
    Validates weights after adding.
    """
    if human_response.get("action") != "add_criterion":
        return

    new_criterion = human_response.get("criterion")
    if new_criterion:
        state["current_rubric"]["criteria"].append(new_criterion)
        _log(state, f"human added criterion: {new_criterion.get('id')} — {new_criterion.get('name')}")

    weight_check = weight_validator(state["current_rubric"]["criteria"])
    if not weight_check["valid"]:
        _log(state, f"weight warning after human added criterion: {weight_check['issues']}")


# ═══════════════════════════════════════════════
# VALIDATION LOOP
# ═══════════════════════════════════════════════

def _run_validation_loop(state: dict) -> None:
    """
    Runs Coverage + Benchmarking agents in parallel.
    Detects conflicts → pauses for human if needed.
    Synthesizes improvement brief.
    Calls Refiner.
    Loops until rubric passes or max iterations hit.
    """

    while state["iteration_count"] < MAX_ITERATIONS:

        _log(state, f"validation loop — iteration {state['iteration_count'] + 1} of {MAX_ITERATIONS}")

        coverage_verdict     = None
        benchmarking_verdict = None

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
                ): "benchmarking"
            }

            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    if agent_name == "coverage":
                        coverage_verdict = result
                    else:
                        benchmarking_verdict = result
                    _log(state, f"{agent_name} agent complete")
                except Exception as e:
                    logger.error(f"coordinator: {agent_name} agent failed — {e}")
                    raise

        state["agent_verdicts"][f"iteration_{state['iteration_count']}"] = {
            "coverage":     coverage_verdict,
            "benchmarking": benchmarking_verdict
        }

        state["previous_violations"] = benchmarking_verdict.get("violations", [])

        conflict_result = _disagreement_detector(coverage_verdict, benchmarking_verdict)

        if conflict_result["has_conflict"]:
            _log(state, "conflict detected — pausing for human input")

            human_response = request_human_input(
                conflict_context=conflict_result,
                current_rubric=state["current_rubric"]
            )

            if human_response["action"] == "tweak_weights":
                _apply_human_weight_decision(state, human_response)
            elif human_response["action"] == "add_criterion":
                _apply_human_add_criterion(state, human_response)

            state["human_decisions"].append(human_response)
            _log(state, f"human decision applied — {human_response['action']}")

        improvement_brief = _evaluation_synthesizer(
            coverage_verdict,
            benchmarking_verdict,
            state["human_decisions"]
        )

        if not improvement_brief["issues_to_fix"] and \
           not improvement_brief["weight_changes"]:
            _log(state, "rubric passed validation — no issues found")
            break

        state["previous_rubric_versions"].append(state["current_rubric"].copy())

        _log(state, "calling refiner agent")

        state["current_rubric"] = refiner_agent(
            current_rubric=state["current_rubric"],
            improvement_brief=improvement_brief,
            document_context=state["document_context"],
            key_responsibilities=state["key_responsibilities"],
            human_decisions=state["human_decisions"],
            previous_rubric_versions=state["previous_rubric_versions"]
        )

        state["iteration_count"] += 1
        _log(state, f"refinement complete — now on rubric v{state['current_rubric'].get('version')}")

    if state["iteration_count"] >= MAX_ITERATIONS:
        _log(state, f"max iterations ({MAX_ITERATIONS}) reached")


# ═══════════════════════════════════════════════
# FINAL GATE
# ═══════════════════════════════════════════════

def _run_final_gate(state: dict) -> bool:
    """
    Presents final rubric to human.
    Human approves or requests changes.
    If changes requested loops back to refiner.
    Returns True if approved, False if stalled.
    """

    while state["iteration_count"] < MAX_ITERATIONS:

        _log(state, "presenting rubric to human at final gate")

        summary = {
            "job_id":           state["job_id"],
            "rubric":           state["current_rubric"],
            "total_iterations": state["iteration_count"],
            "human_decisions":  state["human_decisions"],
            "audit_log":        state["audit_log"],
            "changes_made": [
                v.get("diff", {})
                for v in state["previous_rubric_versions"]
            ]
        }

        human_response = request_human_input(
            conflict_context={"stage": "final_gate", "summary": summary},
            current_rubric=state["current_rubric"]
        )

        if human_response.get("approved") is True:
            _log(state, "human approved rubric at final gate")
            return True

        _log(state, "human requested changes at final gate")

        if human_response["action"] == "tweak_weights":
            _apply_human_weight_decision(state, human_response)
        elif human_response["action"] == "add_criterion":
            _apply_human_add_criterion(state, human_response)

        state["human_decisions"].append(human_response)

        state["previous_rubric_versions"].append(state["current_rubric"].copy())

        state["current_rubric"] = refiner_agent(
            current_rubric=state["current_rubric"],
            improvement_brief={"issues_to_fix": [], "weight_changes": {}},
            document_context=state["document_context"],
            key_responsibilities=state["key_responsibilities"],
            human_decisions=state["human_decisions"],
            previous_rubric_versions=state["previous_rubric_versions"]
        )

        state["iteration_count"] += 1

    _log(state, "max iterations hit at final gate — flagged for manual review")
    return False


# ═══════════════════════════════════════════════
# MAIN PIPELINE — called by parent app
# ═══════════════════════════════════════════════

def run_pipeline(job_id: str) -> dict:
    """
    Main entry point.
    Called by parent app with just a job_id.
    Runs full rubric generation pipeline.

    Args:
        job_id: MongoDB job document _id

    Returns:
        dict with rubric_id and status
    """

    logger.info(f"coordinator: pipeline started for job_id={job_id}")

    job= fetch_job(job_id)
    key_responsibilities = job.get("responsibilities", [])
    document_url= job.get("raw_data", {}).get("document")

    if not key_responsibilities:
        raise ValueError(f"Job {job_id} has no responsibilities")

    logger.info("coordinator: running document intelligence agent")

    document_context = {}
    if document_url:
        document_context = document_intelligence_agent(document_url)
    else:
        logger.warning("coordinator: no document URL found — proceeding with responsibilities only")

    state = build_state(
        job_id=job_id,
        document_context=document_context,
        key_responsibilities=key_responsibilities
    )

    _log(state, f"inputs ready — {len(key_responsibilities)} responsibilities loaded")

    _log(state, "calling rubric builder agent")

    state["current_rubric"] = rubric_builder_agent(
        document_context=state["document_context"],
        key_responsibilities=state["key_responsibilities"]
    )

    _log(state, f"draft rubric built — {len(state['current_rubric'].get('criteria', []))} criteria")

    _log(state, "starting validation loop")
    _run_validation_loop(state)

    approved = _run_final_gate(state)

    if not approved:
        logger.warning(f"coordinator: pipeline stalled for job_id={job_id} — max iterations reached at final gate")
        return {
            "status":    "stalled",
            "job_id":    job_id,
            "rubric":    state["current_rubric"],
            "audit_log": state["audit_log"]
        }

    rubric_id = save_rubric(
        rubric=state["current_rubric"],
        job_id=job_id,
        audit_log=state["audit_log"]
    )

    _log(state, f"pipeline complete — rubric_id={rubric_id}")

    return {
        "status":    "success",
        "job_id":    job_id,
        "rubric_id": rubric_id,
        "rubric":    state["current_rubric"],
        "audit_log": state["audit_log"]
    }
