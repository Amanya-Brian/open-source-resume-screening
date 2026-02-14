"""TalentMatch API client for fetching data."""

import logging
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class TalentMatchClient:
    """Client for interacting with TalentMatch API.

    Provides methods to fetch students, jobs, applications, and other data
    from the TalentMatch platform API.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the API client.

        Args:
            settings: Application settings. If None, uses default.
        """
        self.settings = settings or get_settings()
        self.base_url = self.settings.talentmatch_api_url.rstrip("/")
        self.timeout = self.settings.talentmatch_api_timeout

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body for POST/PUT

        Returns:
            Response data as dictionary

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except requests.Timeout:
            logger.error(f"Request timeout for {url}")
            raise
        except requests.ConnectionError:
            logger.error(f"Connection error for {url}")
            raise
        except requests.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            raise

    def health_check(self) -> bool:
        """Check if the API is available.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = self._make_request("GET", "/")
            return response.get("status") == "healthy" or "error" not in response
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def get_students(
        self,
        page: int = 1,
        limit: int = 50,
        university_id: Optional[int] = None,
        graduation_year: Optional[int] = None,
        search: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch students from the API.

        Args:
            page: Page number for pagination
            limit: Number of results per page
            university_id: Filter by university ID
            graduation_year: Filter by graduation year
            search: Search query string

        Returns:
            Response containing students list and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        if university_id:
            params["university_id"] = university_id
        if graduation_year:
            params["graduation_year"] = graduation_year
        if search:
            params["search"] = search

        return self._make_request("GET", "/api/v1/students", params=params)

    def get_student(self, student_id: str) -> dict[str, Any]:
        """Fetch a single student by ID.

        Args:
            student_id: Student ID

        Returns:
            Student data
        """
        return self._make_request("GET", f"/api/v1/students/{student_id}")

    def get_job_listings(
        self,
        page: int = 1,
        limit: int = 50,
        company_id: Optional[int] = None,
        search: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch job listings from the API.

        Args:
            page: Page number for pagination
            limit: Number of results per page
            company_id: Filter by company ID
            search: Search query string

        Returns:
            Response containing job listings and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        if company_id:
            params["company_id"] = company_id
        if search:
            params["search"] = search

        return self._make_request("GET", "/api/v1/job-listings", params=params)

    def get_job_listing(self, job_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single job listing by ID.

        Args:
            job_id: Job listing ID

        Returns:
            Job listing data or None if not found
        """
        # TalentMatch API doesn't have a single-item endpoint
        # Fetch all jobs and filter by ID
        all_jobs = self.fetch_all_pages(self.get_job_listings, limit=100)

        for job in all_jobs:
            if job.get("id") == job_id:
                return job

        logger.warning(f"Job listing not found: {job_id}")
        return None

    def get_job_applications(
        self,
        page: int = 1,
        limit: int = 50,
        job_id: Optional[str] = None,
        student_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch job applications from the API.

        Args:
            page: Page number for pagination
            limit: Number of results per page
            job_id: Filter by job ID
            student_id: Filter by student ID
            status: Filter by application status

        Returns:
            Response containing applications and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        if job_id:
            params["job_id"] = job_id
        if student_id:
            params["student_id"] = student_id
        if status:
            params["status"] = status

        return self._make_request("GET", "/api/v1/job-applications", params=params)

    def get_internships(
        self,
        page: int = 1,
        limit: int = 50,
        company_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Fetch internships from the API.

        Args:
            page: Page number for pagination
            limit: Number of results per page
            company_id: Filter by company ID

        Returns:
            Response containing internships and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        if company_id:
            params["company_id"] = company_id

        return self._make_request("GET", "/api/v1/internships", params=params)

    def get_internship_applications(
        self,
        page: int = 1,
        limit: int = 50,
        internship_id: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch internship applications from the API.

        Args:
            page: Page number for pagination
            limit: Number of results per page
            internship_id: Filter by internship ID
            student_id: Filter by student ID

        Returns:
            Response containing applications and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        if internship_id:
            params["internship_id"] = internship_id
        if student_id:
            params["student_id"] = student_id

        return self._make_request("GET", "/api/v1/internship-applications", params=params)

    def get_student_documents(
        self,
        page: int = 1,
        limit: int = 50,
        student_id: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch student documents from the API.

        Args:
            page: Page number for pagination
            limit: Number of results per page
            student_id: Filter by student ID
            document_type: Filter by document type (resume, transcript, etc.)

        Returns:
            Response containing documents and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        if student_id:
            params["student_id"] = student_id
        if document_type:
            params["document_type"] = document_type

        return self._make_request("GET", "/api/v1/student-documents", params=params)

    def get_universities(
        self,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Fetch universities from the API.

        Args:
            page: Page number for pagination
            limit: Number of results per page

        Returns:
            Response containing universities and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        return self._make_request("GET", "/api/v1/universities", params=params)

    def get_companies(
        self,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Fetch companies from the API.

        Args:
            page: Page number for pagination
            limit: Number of results per page

        Returns:
            Response containing companies and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        return self._make_request("GET", "/api/v1/companies", params=params)

    def fetch_all_pages(
        self,
        fetch_func: callable,
        max_pages: int = 100,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Fetch all pages of results from a paginated endpoint.

        Args:
            fetch_func: Function to call for each page
            max_pages: Maximum number of pages to fetch
            **kwargs: Additional arguments for fetch_func

        Returns:
            Combined list of all results
        """
        all_results = []
        page = 1

        while page <= max_pages:
            response = fetch_func(page=page, **kwargs)

            # Handle different response formats
            if isinstance(response, list):
                results = response
            elif isinstance(response, dict):
                results = response.get("data", response.get("results", []))
            else:
                break

            if not results:
                break

            all_results.extend(results)

            # Check if there are more pages
            if isinstance(response, dict):
                total_pages = response.get("total_pages", response.get("pages", 1))
                if page >= total_pages:
                    break

            page += 1

        return all_results

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self) -> "TalentMatchClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
