"""Data sync API endpoints for syncing TalentMatch data to MongoDB."""

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from src.services.mongo_service import MongoService
from src.services.talentmatch_client import TalentMatchClient
from src.services.screening_service import ScreeningService
from src.config.settings import get_settings

data_sync_bp = Blueprint("data_sync", __name__)
logger = logging.getLogger(__name__)

# Global API client (stateless, safe to reuse)
_api_client: TalentMatchClient = None


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


def get_api_client() -> TalentMatchClient:
    """Get or create TalentMatch API client."""
    global _api_client
    if _api_client is None:
        settings = get_settings()
        _api_client = TalentMatchClient(settings)
    return _api_client


@data_sync_bp.route("/sync/jobs", methods=["POST"])
def sync_jobs():
    """Sync all jobs from TalentMatch API to MongoDB.

    Returns:
        JSON response with sync results
    """
    try:
        api_client = get_api_client()
        mongo = get_mongo_service()

        # Connect to MongoDB
        run_async(mongo.connect())

        # Fetch all jobs from TalentMatch
        jobs = api_client.fetch_all_pages(api_client.get_job_listings)

        synced_count = 0
        for job in jobs:
            job_doc = {
                "_id": str(job.get("id", job.get("_id", ""))),
                "title": job.get("title", ""),
                "company": job.get("owner_name", job.get("company_name", "Unknown")),
                "description": job.get("description", ""),
                "qualifications": job.get("qualifications", []),
                "responsibilities": job.get("responsibilities", []),
                "preferred_qualifications": job.get("preferred_qualifications", []),
                "location": job.get("location", ""),
                "work_flexibility": job.get("work_flexibility", ""),
                "status": job.get("status", ""),
                "raw_data": job,
            }

            run_async(mongo.update_one(
                "job_listings",
                {"_id": job_doc["_id"]},
                {"$set": job_doc},
                upsert=True
            ))
            synced_count += 1

        return jsonify({
            "success": True,
            "message": f"Synced {synced_count} jobs",
            "jobs_synced": synced_count,
        })

    except Exception as e:
        logger.error(f"Job sync failed: {e}")
        return jsonify({
            "error": "Job sync failed",
            "message": str(e),
        }), 500


@data_sync_bp.route("/sync/jobs/<job_id>/applications", methods=["POST"])
def sync_job_applications(job_id: str):
    """Sync applications for a specific job to MongoDB.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with sync results
    """
    try:
        api_client = get_api_client()
        mongo = get_mongo_service()

        run_async(mongo.connect())

        # Fetch applications for this job
        applications = api_client.fetch_all_pages(
            api_client.get_job_applications,
            job_id=job_id
        )

        synced_apps = 0
        synced_resumes = 0

        for app in applications:
            # Store application
            app_doc = {
                "_id": str(app.get("id", app.get("_id", ""))),
                "job_id": str(job_id),
                "student_id": str(app.get("student_id", "")),
                "student_firstname": app.get("student_firstname", ""),
                "student_lastname": app.get("student_lastname", ""),
                "student_email": app.get("student_email", ""),
                "status": app.get("status", "pending"),
                "cover_letter": app.get("cover_letter", ""),
                "document_url": app.get("document", ""),
                "raw_data": app,
            }

            run_async(mongo.update_one(
                "applications",
                {"_id": app_doc["_id"]},
                {"$set": app_doc},
                upsert=True
            ))
            synced_apps += 1

            # Store resume reference
            if app.get("cover_letter") or app.get("document"):
                resume_doc = {
                    "_id": f"resume-{app.get('student_id', '')}",
                    "student_id": str(app.get("student_id", "")),
                    "raw_text": app.get("cover_letter", ""),
                    "file_url": app.get("document", ""),
                    "file_type": "pdf" if app.get("document") else "",
                }

                run_async(mongo.update_one(
                    "resumes",
                    {"_id": resume_doc["_id"]},
                    {"$set": resume_doc},
                    upsert=True
                ))
                synced_resumes += 1

        return jsonify({
            "success": True,
            "job_id": job_id,
            "applications_synced": synced_apps,
            "resumes_synced": synced_resumes,
        })

    except Exception as e:
        logger.error(f"Application sync failed: {e}")
        return jsonify({
            "error": "Application sync failed",
            "message": str(e),
        }), 500


@data_sync_bp.route("/sync/all", methods=["POST"])
def sync_all():
    """Full sync: jobs and all their applications.

    Returns:
        JSON response with full sync results
    """
    try:
        api_client = get_api_client()
        mongo = get_mongo_service()

        run_async(mongo.connect())

        # Sync all jobs first
        jobs = api_client.fetch_all_pages(api_client.get_job_listings)
        job_ids = []

        for job in jobs:
            job_id = str(job.get("id", job.get("_id", "")))
            job_ids.append(job_id)

            job_doc = {
                "_id": job_id,
                "title": job.get("title", ""),
                "company": job.get("owner_name", job.get("company_name", "Unknown")),
                "description": job.get("description", ""),
                "qualifications": job.get("qualifications", []),
                "responsibilities": job.get("responsibilities", []),
                "preferred_qualifications": job.get("preferred_qualifications", []),
                "location": job.get("location", ""),
                "work_flexibility": job.get("work_flexibility", ""),
                "status": job.get("status", ""),
                "raw_data": job,
            }

            run_async(mongo.update_one(
                "job_listings",
                {"_id": job_doc["_id"]},
                {"$set": job_doc},
                upsert=True
            ))

        # Sync applications for each job
        total_apps = 0
        total_resumes = 0

        for job_id in job_ids:
            try:
                applications = api_client.fetch_all_pages(
                    api_client.get_job_applications,
                    job_id=job_id
                )

                for app in applications:
                    app_doc = {
                        "_id": str(app.get("id", app.get("_id", ""))),
                        "job_id": job_id,
                        "student_id": str(app.get("student_id", "")),
                        "student_firstname": app.get("student_firstname", ""),
                        "student_lastname": app.get("student_lastname", ""),
                        "student_email": app.get("student_email", ""),
                        "status": app.get("status", "pending"),
                        "cover_letter": app.get("cover_letter", ""),
                        "document_url": app.get("document", ""),
                        "raw_data": app,
                    }

                    run_async(mongo.update_one(
                        "applications",
                        {"_id": app_doc["_id"]},
                        {"$set": app_doc},
                        upsert=True
                    ))
                    total_apps += 1

                    if app.get("cover_letter") or app.get("document"):
                        resume_doc = {
                            "_id": f"resume-{app.get('student_id', '')}",
                            "student_id": str(app.get("student_id", "")),
                            "raw_text": app.get("cover_letter", ""),
                            "file_url": app.get("document", ""),
                        }

                        run_async(mongo.update_one(
                            "resumes",
                            {"_id": resume_doc["_id"]},
                            {"$set": resume_doc},
                            upsert=True
                        ))
                        total_resumes += 1

            except Exception as e:
                logger.warning(f"Failed to sync applications for job {job_id}: {e}")

        return jsonify({
            "success": True,
            "jobs_synced": len(job_ids),
            "applications_synced": total_apps,
            "resumes_synced": total_resumes,
        })

    except Exception as e:
        logger.error(f"Full sync failed: {e}")
        return jsonify({
            "error": "Full sync failed",
            "message": str(e),
        }), 500


@data_sync_bp.route("/data/jobs", methods=["GET"])
def list_jobs():
    """List all jobs from MongoDB.

    Returns:
        JSON response with jobs list
    """
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        jobs = run_async(mongo.find_many("job_listings", {}))

        return jsonify({
            "success": True,
            "count": len(jobs),
            "jobs": [
                {
                    "id": job.get("_id"),
                    "title": job.get("title"),
                    "company": job.get("company"),
                    "location": job.get("location"),
                    "status": job.get("status"),
                }
                for job in jobs
            ],
        })

    except Exception as e:
        logger.error(f"List jobs failed: {e}")
        return jsonify({
            "error": "Failed to list jobs",
            "message": str(e),
        }), 500


@data_sync_bp.route("/data/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    """Get a specific job from MongoDB.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with job details
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

        # Get application count
        apps = run_async(mongo.find_many("applications", {"job_id": job_id}))

        return jsonify({
            "success": True,
            "job": {
                "id": job.get("_id"),
                "title": job.get("title"),
                "company": job.get("company"),
                "description": job.get("description"),
                "qualifications": job.get("qualifications", []),
                "responsibilities": job.get("responsibilities", []),
                "location": job.get("location"),
                "work_flexibility": job.get("work_flexibility"),
                "status": job.get("status"),
                "application_count": len(apps),
            },
        })

    except Exception as e:
        logger.error(f"Get job failed: {e}")
        return jsonify({
            "error": "Failed to get job",
            "message": str(e),
        }), 500


@data_sync_bp.route("/data/jobs/<job_id>/candidates", methods=["GET"])
def list_job_candidates(job_id: str):
    """List all candidates/applications for a job.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with candidates list
    """
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        applications = run_async(mongo.find_many("applications", {"job_id": job_id}))

        return jsonify({
            "success": True,
            "job_id": job_id,
            "count": len(applications),
            "candidates": [
                {
                    "application_id": app.get("_id"),
                    "student_id": app.get("student_id"),
                    "name": f"{app.get('student_firstname', '')} {app.get('student_lastname', '')}".strip(),
                    "email": app.get("student_email"),
                    "status": app.get("status"),
                    "has_cover_letter": bool(app.get("cover_letter")),
                    "has_resume": bool(app.get("document_url")),
                }
                for app in applications
            ],
        })

    except Exception as e:
        logger.error(f"List candidates failed: {e}")
        return jsonify({
            "error": "Failed to list candidates",
            "message": str(e),
        }), 500


@data_sync_bp.route("/screening/jobs/<job_id>/screen", methods=["POST"])
def screen_job_candidates(job_id: str):
    """Run screening on all candidates for a job.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with screening results
    """
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Create screening service
        screening_service = ScreeningService(mongo_service=mongo)

        # Run screening
        evaluations = run_async(screening_service.screen_job_candidates(job_id))

        return jsonify({
            "success": True,
            "job_id": job_id,
            "candidates_screened": len(evaluations),
            "results": [
                {
                    "rank": i + 1,
                    "candidate_id": eval.candidate_id,
                    "candidate_name": eval.candidate_name,
                    "total_score": round(eval.total_weighted_score, 2),
                    "percentage": round(eval.percentage, 1),
                    "recommendation": eval.recommendation,
                    "criteria_scores": [
                        {
                            "criterion": cs.criterion_name,
                            "score": cs.raw_score,
                            "weight": f"{int(cs.weight * 100)}%",
                            "weighted": round(cs.weighted_score, 2),
                        }
                        for cs in eval.criteria_scores
                    ],
                    "strengths": eval.strengths,
                    "concerns": eval.concerns,
                }
                for i, eval in enumerate(evaluations)
            ],
        })

    except ValueError as e:
        return jsonify({
            "error": "Screening failed",
            "message": str(e),
        }), 404
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        return jsonify({
            "error": "Screening failed",
            "message": str(e),
        }), 500


@data_sync_bp.route("/data/jobs/<job_id>/results", methods=["GET"])
def get_screening_results(job_id: str):
    """Get stored screening results for a job.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with screening results
    """
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Get job info
        job = run_async(mongo.find_one("job_listings", {"_id": job_id}))

        if not job:
            return jsonify({
                "error": "Job not found",
                "job_id": job_id,
            }), 404

        # Get screening results
        results = run_async(mongo.find_many("screening_results", {"job_id": job_id}))

        # Sort by score
        results.sort(key=lambda r: r.get("total_weighted_score", 0), reverse=True)

        return jsonify({
            "success": True,
            "job_id": job_id,
            "job_title": job.get("title"),
            "company": job.get("company"),
            "total_candidates": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "candidate_id": r.get("candidate_id"),
                    "candidate_name": r.get("candidate_name"),
                    "total_score": round(r.get("total_weighted_score", 0), 2),
                    "percentage": round(r.get("percentage", 0), 1),
                    "recommendation": r.get("recommendation"),
                    "criteria_scores": r.get("criteria_scores", []),
                    "strengths": r.get("strengths", []),
                    "concerns": r.get("concerns", []),
                }
                for i, r in enumerate(results)
            ],
        })

    except Exception as e:
        logger.error(f"Get results failed: {e}")
        return jsonify({
            "error": "Failed to get results",
            "message": str(e),
        }), 500


@data_sync_bp.route("/data/jobs/<job_id>/fairness", methods=["GET"])
def get_fairness_report(job_id: str):
    """Get fairness report for a job screening.

    Args:
        job_id: Job listing ID

    Returns:
        JSON response with fairness analysis
    """
    try:
        mongo = get_mongo_service()
        run_async(mongo.connect())

        # Get fairness report
        report = run_async(mongo.find_one("fairness_reports", {"job_id": job_id}))

        if not report:
            return jsonify({
                "success": False,
                "message": "No fairness report found. Run screening first.",
                "job_id": job_id,
            }), 404

        return jsonify({
            "success": True,
            "job_id": job_id,
            "fairness_report": {
                "is_compliant": report.get("is_compliant"),
                "metrics": {
                    "disparate_impact_ratio": report.get("metrics", {}).get("disparate_impact_ratio"),
                    "demographic_parity": report.get("metrics", {}).get("demographic_parity"),
                    "equal_opportunity": report.get("metrics", {}).get("equal_opportunity"),
                    "attribute_variance": report.get("metrics", {}).get("attribute_variance", {}),
                },
                "violations": report.get("violations", []),
                "recommendations": report.get("recommendations", []),
                "created_at": report.get("created_at"),
            },
        })

    except Exception as e:
        logger.error(f"Get fairness report failed: {e}")
        return jsonify({
            "error": "Failed to get fairness report",
            "message": str(e),
        }), 500
