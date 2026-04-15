"""Data Fetching Agent for retrieving data from TalentMatch API."""

import asyncio
import logging
from typing import Any, Optional

from src.agents.base import AgentConfig, AgentContext, AgentResult, BaseAgent
from src.models.schemas import Application, JobListing, Resume, Student
from src.services.document_parser import DocumentParser
from src.services.mongo_service import MongoService
from src.services.talentmatch_client import TalentMatchClient

logger = logging.getLogger(__name__)


class DataFetchingInput:
    """Input for Data Fetching Agent."""

    def __init__(
        self,
        job_id: str,
        fetch_students: bool = True,
        fetch_applications: bool = True,
        fetch_resumes: bool = True,
        sync_to_db: bool = True,
    ):
        self.job_id = job_id
        self.fetch_students = fetch_students
        self.fetch_applications = fetch_applications
        self.fetch_resumes = fetch_resumes
        self.sync_to_db = sync_to_db


class DataFetchingOutput:
    """Output from Data Fetching Agent."""

    def __init__(
        self,
        job: Optional[JobListing] = None,
        students: Optional[list[Student]] = None,
        applications: Optional[list[Application]] = None,
        resumes: Optional[list[Resume]] = None,
        rubric: Optional[dict] = None,
    ):
        self.job = job
        self.students = students or []
        self.applications = applications or []
        self.resumes = resumes or []
        self.rubric = rubric  # Generated rubric for this job, or None

    @property
    def candidate_count(self) -> int:
        """Get number of candidates."""
        return len(self.students)


class DataFetchingAgent(BaseAgent[DataFetchingInput, DataFetchingOutput]):
    """Agent responsible for fetching data from TalentMatch API.

    This agent:
    - Fetches job listing details
    - Retrieves all applications for the job
    - Fetches student profiles for applicants
    - Downloads and parses resumes
    - Optionally syncs data to MongoDB
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        api_client: Optional[TalentMatchClient] = None,
        mongo_service: Optional[MongoService] = None,
        document_parser: Optional[DocumentParser] = None,
    ):
        """Initialize the Data Fetching Agent.

        Args:
            config: Agent configuration
            api_client: TalentMatch API client
            mongo_service: MongoDB service for data persistence
            document_parser: Document parser for resumes
        """
        super().__init__(config or AgentConfig(name="data_fetching"))
        self.api_client = api_client or TalentMatchClient(self.settings)
        self.mongo_service = mongo_service
        self.document_parser = document_parser or DocumentParser()

    async def execute(
        self,
        input_data: DataFetchingInput,
        context: AgentContext,
    ) -> AgentResult[DataFetchingOutput]:
        """Execute data fetching from TalentMatch API.

        Args:
            input_data: Fetch configuration
            context: Pipeline context

        Returns:
            AgentResult with fetched data
        """
        output = DataFetchingOutput()
        errors = []

        try:
            # 1. Fetch job listing
            logger.info(f"Fetching job listing: {input_data.job_id}")
            job_data = await self._fetch_job(input_data.job_id)
            if job_data:
                output.job = self._parse_job(job_data)
                context.job = output.job
            else:
                errors.append(f"Job not found: {input_data.job_id}")
                return AgentResult.failure_result(errors, self.name)

            # 2. Fetch applications for this job
            if input_data.fetch_applications:
                logger.info(f"Fetching applications for job: {input_data.job_id}")
                applications_data = await self._fetch_applications(input_data.job_id)
                output.applications = [
                    self._parse_application(app) for app in applications_data
                ]
                logger.info(f"Found {len(output.applications)} applications")

                # Extract students from application data (TalentMatch includes student info)
                if input_data.fetch_students:
                    output.students = [
                        self._parse_student_from_application(app)
                        for app in applications_data
                    ]
                    context.candidates = output.students
                    logger.info(f"Extracted {len(output.students)} student profiles from applications")

                # Extract resume URLs from applications
                if input_data.fetch_resumes:
                    output.resumes = self._extract_resumes_from_applications(
                        applications_data, output.applications
                    )
                    logger.info(f"Extracted {len(output.resumes)} resume references")

            # 5. Sync to MongoDB
            if input_data.sync_to_db and self.mongo_service:
                logger.info("Syncing data to MongoDB")
                await self._sync_to_db(output)

            # 6. Fetch rubric for this job from MongoDB
            if self.mongo_service:
                output.rubric = await self._fetch_rubric(input_data.job_id)
                if output.rubric:
                    logger.info(f"Rubric loaded: {len(output.rubric.get('criteria', []))} criteria")
                else:
                    logger.info("No rubric found for this job")

            return AgentResult.success_result(output, self.name)

        except Exception as e:
            logger.error(f"Data fetching error: {e}")
            errors.append(str(e))
            return AgentResult.failure_result(errors, self.name)

    async def _fetch_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Fetch job listing from API."""
        try:
            # Run sync API call in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.api_client.get_job_listing,
                job_id,
            )
        except Exception as e:
            logger.error(f"Failed to fetch job {job_id}: {e}")
            return None

    async def _fetch_applications(self, job_id: str) -> list[dict[str, Any]]:
        """Fetch all applications for a job."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.api_client.fetch_all_pages(
                    self.api_client.get_job_applications,
                    job_id=job_id,
                ),
            )
            return result
        except Exception as e:
            logger.error(f"Failed to fetch applications: {e}")
            return []

    async def _fetch_students(
        self,
        student_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch student profiles by IDs."""
        students = []

        # Batch fetch students
        async def fetch_student(sid: str) -> Optional[dict[str, Any]]:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    self.api_client.get_student,
                    sid,
                )
            except Exception as e:
                logger.warning(f"Failed to fetch student {sid}: {e}")
                return None

        # Fetch in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_candidates)

        async def fetch_with_limit(sid: str):
            async with semaphore:
                return await fetch_student(sid)

        results = await asyncio.gather(
            *[fetch_with_limit(sid) for sid in student_ids],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, dict):
                students.append(result)

        return students

    def _extract_resumes_from_applications(
        self,
        applications_data: list[dict[str, Any]],
        applications: list[Application],
    ) -> list[Resume]:
        """Extract resume info from application data."""
        resumes = []

        for app_data, app in zip(applications_data, applications):
            document_url = app_data.get("document", "")
            cover_letter = app_data.get("cover_letter", "")

            # Use cover letter as resume text if available (contains relevant info)
            resume_text = cover_letter or ""

            resumes.append(Resume(
                _id=f"resume-{app.student_id}",
                student_id=app.student_id,
                raw_text=resume_text,
                file_type="pdf" if document_url else "",
                file_url=document_url,
            ))

        return resumes

    async def _fetch_resumes(
        self,
        students: list[Student],
    ) -> list[Resume]:
        """Fetch and parse resumes for students."""
        resumes = []

        # Get resume documents from API
        try:
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                lambda: self.api_client.fetch_all_pages(
                    self.api_client.get_student_documents,
                    document_type="resume",
                ),
            )

            # Create mapping of student_id to document
            student_docs = {
                doc.get("student_id"): doc
                for doc in documents
            }

            for student in students:
                doc = student_docs.get(student.id)
                if doc:
                    resume = await self._parse_resume_document(student.id, doc)
                    if resume:
                        resumes.append(resume)
                else:
                    # Create empty resume placeholder
                    resumes.append(Resume(
                        student_id=student.id,
                        raw_text="",
                        file_type="",
                    ))

        except Exception as e:
            logger.error(f"Failed to fetch resumes: {e}")

        return resumes

    async def _parse_resume_document(
        self,
        student_id: str,
        doc: dict[str, Any],
    ) -> Optional[Resume]:
        """Parse a resume document."""
        try:
            file_url = doc.get("file_url", doc.get("url", ""))
            file_type = doc.get("file_type", "pdf")

            # For now, create resume with metadata
            # Actual file fetching would require downloading
            return Resume(
                id=doc.get("_id", doc.get("id", "")),
                student_id=student_id,
                raw_text=doc.get("content", ""),
                file_type=file_type,
                file_url=file_url,
            )

        except Exception as e:
            logger.warning(f"Failed to parse resume for {student_id}: {e}")
            return None

    async def _fetch_rubric(self, job_id: str) -> Optional[dict]:
        """Fetch the generated rubric for this job from MongoDB."""
        try:
            await self.mongo_service.connect()
            job_record = await self.mongo_service.find_one("job_listings", {"_id": job_id})
            if not job_record or not job_record.get("rubric_id"):
                return None
            from bson import ObjectId
            rubric = await self.mongo_service.find_one(
                "rubrics", {"_id": ObjectId(job_record["rubric_id"])}
            )
            return rubric
        except Exception as e:
            logger.warning(f"Could not fetch rubric for job {job_id}: {e}")
            return None

    async def _sync_to_db(self, output: DataFetchingOutput) -> None:
        """Sync fetched data to MongoDB."""
        if not self.mongo_service:
            return

        try:
            await self.mongo_service.connect()

            # Save job
            if output.job:
                await self.mongo_service.save_model(
                    "job_listings",
                    output.job,
                )

            # Save applications
            for app in output.applications:
                await self.mongo_service.save_model(
                    "applications",
                    app,
                )

            # Save students
            for student in output.students:
                await self.mongo_service.save_model(
                    "students",
                    student,
                )

            # Save resumes
            for resume in output.resumes:
                if resume.raw_text:
                    await self.mongo_service.save_model(
                        "resumes",
                        resume,
                    )

            logger.info("Data synced to MongoDB successfully")

        except Exception as e:
            logger.error(f"Failed to sync to MongoDB: {e}")

    def _parse_job(self, data: dict[str, Any]) -> JobListing:
        """Parse job listing from TalentMatch API response."""
        # TalentMatch uses 'qualifications' and 'responsibilities'
        qualifications = data.get("qualifications", [])
        responsibilities = data.get("responsibilities", [])

        # Build description from responsibilities
        description = data.get("description", "")
        if not description and responsibilities:
            description = "Responsibilities:\n- " + "\n- ".join(responsibilities)

        # Extract skills from qualifications (look for technical terms)
        skills = self._extract_skills_from_qualifications(qualifications)

        return JobListing(
            _id=str(data.get("id", data.get("_id", ""))),
            title=data.get("title", ""),
            company=data.get("owner_name", data.get("company_name", "Unknown Company")),
            description=description,
            requirements=qualifications,
            preferred_qualifications=data.get("preferred_qualifications", []),
            required_skills=skills,
            location=data.get("location", ""),
            job_type=self._map_work_flexibility(data.get("work_flexibility", "full-time")),
        )

    def _extract_skills_from_qualifications(self, qualifications: list[str]) -> list[str]:
        """Extract skill keywords from qualifications list."""
        skill_keywords = [
            "python", "java", "javascript", "sql", "excel", "word", "powerpoint",
            "communication", "marketing", "sales", "leadership", "teamwork",
            "adobe", "photoshop", "illustrator", "figma", "canva",
            "crm", "digital marketing", "social media", "content creation",
            "data analysis", "research", "project management",
        ]

        found_skills = set()
        text = " ".join(qualifications).lower()

        for skill in skill_keywords:
            if skill in text:
                found_skills.add(skill.title())

        return list(found_skills)

    def _map_work_flexibility(self, flexibility: str) -> str:
        """Map TalentMatch work_flexibility to job type."""
        mapping = {
            "ONSITE": "full-time",
            "REMOTE": "remote",
            "HYBRID": "hybrid",
        }
        return mapping.get(flexibility, "full-time")

    def _parse_application(self, data: dict[str, Any]) -> Application:
        """Parse application from TalentMatch API response."""
        return Application(
            _id=str(data.get("id", data.get("_id", ""))),
            student_id=str(data.get("student_id", "")),
            job_id=str(data.get("job_id", "")),
            status=data.get("status", "pending").lower(),
            cover_letter=data.get("cover_letter", ""),
            resume_id=data.get("document", ""),  # URL to resume document
        )

    def _parse_student(self, data: dict[str, Any]) -> Student:
        """Parse student from TalentMatch API response."""
        # TalentMatch uses 'firstname' and 'lastname' (no underscore)
        return Student(
            _id=str(data.get("id", data.get("_id", ""))),
            first_name=data.get("firstname", data.get("first_name", "")),
            last_name=data.get("lastname", data.get("last_name", "")),
            email=data.get("email", ""),
            university=data.get("university_name", data.get("university", "")),
            graduation_year=data.get("graduation_year"),
            major=data.get("major", data.get("field_of_study", "")),
            skills=data.get("skills", []),
            gender=data.get("gender"),
            nationality=data.get("nationality"),
        )

    def _parse_student_from_application(self, app_data: dict[str, Any]) -> Student:
        """Parse student info embedded in application data."""
        return Student(
            _id=str(app_data.get("student_id", "")),
            first_name=app_data.get("student_firstname", ""),
            last_name=app_data.get("student_lastname", ""),
            email=app_data.get("student_email", ""),
            university="",
            skills=[],
        )
