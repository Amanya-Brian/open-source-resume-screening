"""Web portal routes for the screening dashboard."""

import asyncio
import logging
from typing import Any

from flask import Blueprint, render_template, flash, redirect, url_for

from src.services.mongo_service import MongoService
from src.models.scoring import DefaultCriteria

portal_bp = Blueprint("portal", __name__)
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
    # Create new instance each time to avoid event loop issues
    return MongoService()


@portal_bp.route("/")
def dashboard():
    """Main dashboard view."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Get stats
        jobs = run_async(mongo.find_many("job_listings", {}))
        applications = run_async(mongo.find_many("applications", {}))
        results = run_async(mongo.find_many("screening_results", {}))

        stats = {
            "total_jobs": len(jobs),
            "total_applications": len(applications),
            "total_results": len(results),
        }

        # Get job list with application counts
        job_list = []
        for job in jobs[:10]:  # Limit to 10 most recent
            job_id = job.get("_id")
            job_apps = run_async(mongo.find_many("applications", {"job_id": job_id}))
            job_results = run_async(mongo.find_many("screening_results", {"job_id": job_id}))

            job_list.append({
                "id": job_id,
                "title": job.get("title", "Untitled"),
                "company": job.get("company", "Unknown"),
                "location": job.get("location", ""),
                "application_count": len(job_apps),
                "screened_count": len(job_results),
            })

        return render_template(
            "dashboard.html",
            active_page="dashboard",
            stats=stats,
            jobs=job_list,
        )

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template(
            "dashboard.html",
            active_page="dashboard",
            stats={"total_jobs": 0, "total_applications": 0, "total_results": 0},
            jobs=[],
        )


@portal_bp.route("/jobs")
def jobs_list():
    """List all jobs."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        jobs = run_async(mongo.find_many("job_listings", {}))

        job_list = []
        for job in jobs:
            job_id = job.get("_id")
            job_apps = run_async(mongo.find_many("applications", {"job_id": job_id}))
            job_results = run_async(mongo.find_many("screening_results", {"job_id": job_id}))

            job_list.append({
                "id": job_id,
                "title": job.get("title", "Untitled"),
                "company": job.get("company", "Unknown"),
                "location": job.get("location", ""),
                "application_count": len(job_apps),
                "screened_count": len(job_results),
            })

        return render_template(
            "jobs_list.html",
            active_page="jobs",
            jobs=job_list,
        )

    except Exception as e:
        logger.error(f"Jobs list error: {e}")
        flash(f"Error loading jobs: {e}", "danger")
        return render_template("jobs_list.html", active_page="jobs", jobs=[])


@portal_bp.route("/jobs/<job_id>")
def job_detail(job_id: str):
    """View job details and screening results."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Get job
        job = run_async(mongo.find_one("job_listings", {"_id": job_id}))
        if not job:
            flash("Job not found", "danger")
            return redirect(url_for("portal.jobs_list"))

        # Get applications
        applications = run_async(mongo.find_many("applications", {"job_id": job_id}))

        # Get screening results
        results = run_async(mongo.find_many("screening_results", {"job_id": job_id}))
        results.sort(key=lambda r: r.get("total_weighted_score", 0), reverse=True)

        # Format job data
        job_data = {
            "id": job.get("_id"),
            "title": job.get("title", "Untitled"),
            "company": job.get("company", "Unknown"),
            "location": job.get("location", ""),
            "description": job.get("description", ""),
            "qualifications": job.get("qualifications", []),
            "responsibilities": job.get("responsibilities", []),
        }

        # Format applications
        apps_data = [
            {
                "id": app.get("_id"),
                "student_id": app.get("student_id"),
                "name": f"{app.get('student_firstname', '')} {app.get('student_lastname', '')}".strip() or "Unknown",
                "email": app.get("student_email", ""),
                "has_cover_letter": bool(app.get("cover_letter")),
                "has_resume": bool(app.get("document_url")),
            }
            for app in applications
        ]

        # Format results
        results_data = [
            {
                "candidate_id": r.get("candidate_id"),
                "candidate_name": r.get("candidate_name", "Unknown"),
                "total_weighted_score": r.get("total_weighted_score", 0),
                "percentage": r.get("percentage", 0),
                "recommendation": r.get("recommendation", ""),
                "criteria_scores": r.get("criteria_scores", []),
                "strengths": r.get("strengths", []),
                "concerns": r.get("concerns", []),
            }
            for r in results
        ]

        return render_template(
            "job_detail.html",
            active_page="jobs",
            job=job_data,
            applications=apps_data,
            results=results_data,
        )

    except Exception as e:
        logger.error(f"Job detail error: {e}")
        flash(f"Error loading job: {e}", "danger")
        return redirect(url_for("portal.jobs_list"))


@portal_bp.route("/jobs/<job_id>/candidates/<candidate_id>")
def candidate_detail(job_id: str, candidate_id: str):
    """View detailed candidate evaluation."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Get job
        job = run_async(mongo.find_one("job_listings", {"_id": job_id}))
        if not job:
            flash("Job not found", "danger")
            return redirect(url_for("portal.jobs_list"))

        # Get screening result for this candidate
        result_id = f"{job_id}-{candidate_id}"
        result = run_async(mongo.find_one("screening_results", {"_id": result_id}))

        if not result:
            flash("Candidate screening result not found", "warning")
            return redirect(url_for("portal.job_detail", job_id=job_id))

        # Get all results to determine rank
        all_results = run_async(mongo.find_many("screening_results", {"job_id": job_id}))
        all_results.sort(key=lambda r: r.get("total_weighted_score", 0), reverse=True)

        rank = 1
        for i, r in enumerate(all_results, 1):
            if r.get("candidate_id") == candidate_id:
                rank = i
                break

        job_data = {
            "id": job.get("_id"),
            "title": job.get("title", "Untitled"),
        }

        # Format criteria scores properly
        criteria_scores = []
        for cs in result.get("criteria_scores", []):
            criteria_scores.append({
                "criterion_key": cs.get("criterion_key"),
                "criterion_name": cs.get("criterion_name"),
                "weight": cs.get("weight", 0),
                "raw_score": cs.get("raw_score", 0),
                "weighted_score": cs.get("weighted_score", 0),
                "evidence": cs.get("evidence", ""),
            })

        result_data = {
            "rank": rank,
            "candidate_id": result.get("candidate_id"),
            "candidate_name": result.get("candidate_name", "Unknown"),
            "total_weighted_score": result.get("total_weighted_score", 0),
            "percentage": result.get("percentage", 0),
            "recommendation": result.get("recommendation", ""),
            "criteria_scores": criteria_scores,
            "strengths": result.get("strengths", []),
            "concerns": result.get("concerns", []),
        }

        return render_template(
            "candidate_detail.html",
            active_page="jobs",
            job=job_data,
            result=result_data,
            total_candidates=len(all_results),
        )

    except Exception as e:
        logger.error(f"Candidate detail error: {e}")
        flash(f"Error loading candidate: {e}", "danger")
        return redirect(url_for("portal.job_detail", job_id=job_id))


@portal_bp.route("/scoring-matrix")
def scoring_matrix():
    """View the scoring matrix configuration."""
    criteria = []
    for c in DefaultCriteria.get_all():
        criteria.append({
            "name": c.name,
            "key": c.key,
            "weight_percentage": int(c.weight * 100),
            "description": c.description,
        })

    return render_template(
        "scoring_matrix.html",
        active_page="scoring",
        criteria=criteria,
    )
