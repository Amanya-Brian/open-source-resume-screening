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
        jobs = run_async(mongo.find_many(
            "job_listings", {},
            sort=[("raw_data.created_at", -1)],
        ))
        applications = run_async(mongo.find_many("applications", {}))
        results = run_async(mongo.find_many("screening_results", {}))

        stats = {
            "total_jobs": len(jobs),
            "total_applications": len(applications),
            "total_results": len(results),
        }

        # Get job list with application counts (sorted by created_at descending)
        job_list = []
        for job in jobs[:10]:
            job_id = job.get("_id")
            job_apps = run_async(mongo.find_many("applications", {"job_id": job_id}))
            job_results = run_async(mongo.find_many("screening_results", {"job_id": job_id}))

            raw_data = job.get("raw_data", {})
            created_at = raw_data.get("created_at", "")
            # Format date for display (e.g., "2026-02-07T16:05:24" -> "Feb 07, 2026")
            created_display = ""
            if created_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                    created_display = dt.strftime("%b %d, %Y")
                except (ValueError, TypeError):
                    created_display = str(created_at)[:10]

            job_list.append({
                "id": job_id,
                "title": job.get("title", "Untitled"),
                "company": job.get("company", "Unknown"),
                "location": job.get("location", ""),
                "application_count": len(job_apps),
                "screened_count": len(job_results),
                "created_at": created_display,
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

        jobs = run_async(mongo.find_many(
            "job_listings", {},
            sort=[("raw_data.created_at", -1)],
        ))

        job_list = []
        for job in jobs:
            job_id = job.get("_id")
            job_apps = run_async(mongo.find_many("applications", {"job_id": job_id}))
            job_results = run_async(mongo.find_many("screening_results", {"job_id": job_id}))

            raw_data = job.get("raw_data", {})
            created_at = raw_data.get("created_at", "")
            created_display = ""
            if created_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                    created_display = dt.strftime("%b %d, %Y")
                except (ValueError, TypeError):
                    created_display = str(created_at)[:10]

            job_list.append({
                "id": job_id,
                "title": job.get("title", "Untitled"),
                "company": job.get("company", "Unknown"),
                "location": job.get("location", ""),
                "application_count": len(job_apps),
                "screened_count": len(job_results),
                "created_at": created_display,
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
            "rubric_id": job.get("rubric_id"),
            "has_rubric": bool(job.get("rubric_id")),
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

        # Fetch rubric for the full PDF report (server-side to avoid client fetch issues)
        rubric_data = None
        rubric_id = job.get("rubric_id")
        if rubric_id:
            try:
                from bson import ObjectId
                rubric_doc = run_async(mongo.find_one("rubrics", {"_id": ObjectId(rubric_id)}))
                if rubric_doc:
                    rubric_doc["_id"] = str(rubric_doc["_id"])
                    rubric_data = {
                        "name": rubric_doc.get("name", rubric_doc.get("title", "")),
                        "description": rubric_doc.get("description", ""),
                        "criteria": [
                            {
                                "name": c.get("name", ""),
                                "key": c.get("id", c.get("key", "")),
                                "weight": c.get("weight", 0),
                                "description": c.get("description", ""),
                            }
                            for c in rubric_doc.get("criteria", [])
                        ],
                    }
            except Exception as e:
                logger.warning(f"Could not load rubric for job {job_id}: {e}")

        # Fetch fairness report for the full PDF report (server-side)
        fairness_data = None
        fairness_doc = run_async(mongo.find_one("fairness_reports", {"job_id": job_id}))
        if fairness_doc:
            m = fairness_doc.get("metrics", {})
            fairness_data = {
                "is_compliant": fairness_doc.get("is_compliant"),
                "metrics": {
                    "disparate_impact_ratio": m.get("disparate_impact_ratio"),
                    "demographic_parity": m.get("demographic_parity"),
                    "equal_opportunity": m.get("equal_opportunity"),
                    "attribute_variance": m.get("attribute_variance", {}),
                },
                "violations": fairness_doc.get("violations", []),
                "recommendations": fairness_doc.get("recommendations", []),
                "generated_at": fairness_doc.get("generated_at"),
            }

        return render_template(
            "job_detail.html",
            active_page="jobs",
            job=job_data,
            applications=apps_data,
            results=results_data,
            rubric_data=rubric_data,
            fairness_data=fairness_data,
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

        # Fetch resume URL from applications collection
        application = run_async(mongo.find_one(
            "applications",
            {"job_id": job_id, "student_id": candidate_id},
        ))
        resume_url = None
        if application:
            resume_url = application.get("document_url") or application.get("document") or None

        return render_template(
            "candidate_detail.html",
            active_page="jobs",
            job=job_data,
            result=result_data,
            total_candidates=len(all_results),
            resume_url=resume_url,
        )

    except Exception as e:
        logger.error(f"Candidate detail error: {e}")
        flash(f"Error loading candidate: {e}", "danger")
        return redirect(url_for("portal.job_detail", job_id=job_id))


@portal_bp.route("/scoring-matrix")
def scoring_matrix():
    """View the global scoring matrix and existing rubrics."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Default/global criteria (explanatory)
        criteria = []
        for c in DefaultCriteria.get_all():
            criteria.append({
                "name": c.name,
                "key": c.key,
                "weight_percentage": int(c.weight * 100),
                "description": c.description,
            })

        # Existing rubrics per job
        rubrics_view: list[dict[str, Any]] = []
        rubrics = run_async(mongo.find_many("rubrics", {}, sort=[("created_at", -1)]))

        jobs_with_rubric_ids = set()

        for r in rubrics:
            rubric_id = str(r.get("_id"))
            created_at = r.get("created_at")
            created_display = ""
            if created_at:
                try:
                    from datetime import datetime
                    dt = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(
                        str(created_at).replace("Z", "+00:00")
                    )
                    created_display = dt.strftime("%b %d, %Y")
                except Exception:
                    created_display = str(created_at)[:10]

            job = run_async(mongo.find_one("job_listings", {"rubric_id": rubric_id}))
            job_id = job.get("_id") if job else None
            if job_id:
                jobs_with_rubric_ids.add(str(job_id))

            rubrics_view.append({
                "id": rubric_id,
                "name": r.get("name", "Untitled Rubric"),
                "description": r.get("description", ""),
                "created_at": created_display,
                "job_id": job_id,
                "job_title": job.get("title", "Unlinked") if job else "Unlinked",
                "job_company": job.get("company", "") if job else "",
            })

        # Jobs without a rubric yet
        all_jobs = run_async(mongo.find_many("job_listings", {}, sort=[("raw_data.created_at", -1)]))
        jobs_without_rubric = [
            {
                "id":      j.get("_id"),
                "title":   j.get("title", "Untitled"),
                "company": j.get("company", ""),
            }
            for j in all_jobs
            if str(j.get("_id")) not in jobs_with_rubric_ids
        ]

        return render_template(
            "scoring_matrix.html",
            active_page="scoring",
            criteria=criteria,
            rubrics=rubrics_view,
            jobs_without_rubric=jobs_without_rubric,
        )

    except Exception as e:
        logger.error(f"Scoring matrix error: {e}")
        return render_template(
            "scoring_matrix.html",
            active_page="scoring",
            criteria=[],
            rubrics=[],
        )


@portal_bp.route("/jobs/<job_id>/candidates/<candidate_id>/resume")
def candidate_resume(job_id: str, candidate_id: str):
    """Display resume preview for a candidate."""
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Get job
        job = run_async(mongo.find_one("job_listings", {"_id": job_id}))
        if not job:
            flash("Job not found", "danger")
            return redirect(url_for("portal.jobs_list"))

        # Get candidate name from screening result
        result_id = f"{job_id}-{candidate_id}"
        result = run_async(mongo.find_one("screening_results", {"_id": result_id}))
        candidate_name = result.get("candidate_name", "Unknown") if result else "Unknown"

        # Get resume URL from applications
        application = run_async(mongo.find_one(
            "applications",
            {"job_id": job_id, "student_id": candidate_id},
        ))

        resume_url = None
        if application:
            resume_url = application.get("document_url") or application.get("document") or None

        job_data = {
            "id": job.get("_id"),
            "title": job.get("title", "Untitled"),
        }

        return render_template(
            "resume_preview.html",
            active_page="jobs",
            job=job_data,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            resume_url=resume_url,
        )

    except Exception as e:
        logger.error(f"Resume preview error: {e}")
        flash(f"Error loading resume: {e}", "danger")
        return redirect(url_for("portal.candidate_detail", job_id=job_id, candidate_id=candidate_id))