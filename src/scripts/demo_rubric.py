"""Demo script to generate a scoring rubric for a real job from the database."""

import asyncio
import sys
from typing import Optional

from src.agents.rubric_generation_agent import (
    RubricGenerationAgent,
    RubricGenerationInput,
)
from src.agents.base import AgentContext
from src.models.schemas import JobListing
from src.services.mongo_service import MongoService


async def create_sample_job() -> JobListing:
    """Create a sample job listing in the database for testing."""
    mongo = MongoService.get_instance()
    try:
        await mongo.connect()
        
        sample_job = JobListing(
            _id="demo-job-001",
            title="Senior Software Engineer",
            company="TechCorp Inc.",
            description="We are seeking a Senior Software Engineer to join our dynamic team. You will be responsible for designing, developing, and maintaining high-quality software solutions.",
            requirements=[
                "Bachelor's degree in Computer Science or related field",
                "5+ years of experience in software development",
                "Strong proficiency in Python, JavaScript, and SQL",
                "Experience with cloud platforms (AWS, Azure, or GCP)",
                "Knowledge of microservices architecture"
            ],
            preferred_qualifications=[
                "Master's degree in Computer Science",
                "Experience with containerization (Docker, Kubernetes)",
                "Familiarity with CI/CD pipelines",
                "Experience with agile development methodologies"
            ],
            required_skills=[
                "Python",
                "JavaScript/TypeScript",
                "SQL",
                "AWS/Azure/GCP",
                "Docker",
                "Kubernetes",
                "REST APIs",
                "Git"
            ],
            location="San Francisco, CA",
            job_type="full-time",
            salary_min=120000,
            salary_max=180000,
            experience_years_min=5,
            is_active=True
        )
        
        # Insert the job into the database
        await mongo.insert_one("job_listings", sample_job.model_dump(by_alias=True))
        print("Sample job created successfully!")
        return sample_job
        
    finally:
        await mongo.disconnect()


async def fetch_job_from_db(job_id: Optional[str] = None) -> Optional[JobListing]:
    """Fetch a job listing from MongoDB.
    
    Args:
        job_id: Optional specific job ID. If not provided, fetches the first active job.
        
    Returns:
        JobListing if found, None otherwise.
    """
    mongo = MongoService.get_instance()
    try:
        await mongo.connect()
        
        # If job_id provided, fetch that specific job
        if job_id:
            job_doc = await mongo.find_one("job_listings", {"_id": job_id})
        else:
            # Otherwise fetch the first active job
            job_doc = await mongo.find_one("job_listings", {"is_active": True})
        
        if job_doc:
            return JobListing(**job_doc)
        return None
    finally:
        await mongo.disconnect()


async def main():
    """Main function to generate and display a rubric for a real job."""
    # First try to fetch an existing job
    print("Fetching job from database...")
    job = await fetch_job_from_db()
    
    # If no job exists, create a sample job
    if not job:
        print("No job found. Creating a sample job for testing...")
        job = await create_sample_job()
    
    print(f"\n{'='*60}")
    print(f"Job Title: {job.title}")
    print(f"Company: {job.company}")
    print(f"Location: {job.location or 'Not specified'}")
    print(f"{'='*60}\n")
    
    print("Generating tailored evaluation rubric...")
    print("-" * 60)
    
    # Create and run the rubric generation agent
    agent = RubricGenerationAgent()  # uses default LLM service
    
    print(f"Testing LLM availability...")
    try:
        # Initialize LLM first
        agent.llm_service.initialize()
        # Then check status
        llm_info = agent.llm_service.get_model_info()
        print(f"LLM Status: {llm_info.get('status', 'unknown')}")
        print(f"Model: {llm_info.get('model', 'unknown')}")
        print(f"Available models: {llm_info.get('available_models', [])}")
    except Exception as e:
        print(f"LLM initialization failed: {e}")
    
    result = await agent.run(
        RubricGenerationInput(job=job),
        AgentContext(job_id=job.id),
    )

    if result.success:
        print("\nGenerated Rubric:\n")
        for i, crit in enumerate(result.data.criteria, 1):
            weight_pct = crit.weight * 100
            print(f"{i}. {crit.name}")
            print(f"   Key: {crit.key}")
            print(f"   Weight: {weight_pct:.1f}%")
            print(f"   Description: {crit.description}")
            print()
    else:
        print("Rubric generation failed:", result.errors)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())