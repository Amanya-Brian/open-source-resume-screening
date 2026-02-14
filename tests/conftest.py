"""Pytest configuration and fixtures."""

import pytest
from datetime import datetime

from src.models.schemas import (
    Application,
    JobListing,
    ParsedResume,
    Resume,
    ScreeningScore,
    Student,
)


@pytest.fixture
def sample_job():
    """Create a sample job listing."""
    return JobListing(
        _id="job-123",
        title="Software Engineer",
        company="TechCorp",
        description="We are looking for a skilled software engineer...",
        requirements=["Bachelor's degree in CS", "3+ years experience"],
        required_skills=["Python", "JavaScript", "SQL", "AWS"],
        location="Kigali, Rwanda",
        job_type="full-time",
    )


@pytest.fixture
def sample_students():
    """Create sample students."""
    return [
        Student(
            _id="student-1",
            first_name="John",
            last_name="Doe",
            email="john.doe@example.com",
            university="CMU Africa",
            graduation_year=2024,
            major="Computer Science",
            gpa=3.8,
            skills=["Python", "JavaScript", "React", "AWS"],
            gender="male",
        ),
        Student(
            _id="student-2",
            first_name="Jane",
            last_name="Smith",
            email="jane.smith@example.com",
            university="University of Rwanda",
            graduation_year=2024,
            major="Software Engineering",
            gpa=3.5,
            skills=["Python", "Django", "PostgreSQL"],
            gender="female",
        ),
        Student(
            _id="student-3",
            first_name="Alice",
            last_name="Johnson",
            email="alice.johnson@example.com",
            university="ALU",
            graduation_year=2025,
            major="Data Science",
            gpa=3.9,
            skills=["Python", "Machine Learning", "SQL"],
            gender="female",
        ),
    ]


@pytest.fixture
def sample_resumes(sample_students):
    """Create sample resumes for students."""
    return {
        "student-1": Resume(
            _id="resume-1",
            student_id="student-1",
            raw_text="Experienced software engineer with 4 years of experience...",
            file_type="pdf",
            parsed_data=ParsedResume(
                skills=["Python", "JavaScript", "React", "AWS", "Docker"],
                experience=[],
                education=[],
            ),
        ),
        "student-2": Resume(
            _id="resume-2",
            student_id="student-2",
            raw_text="Full-stack developer specializing in Python and Django...",
            file_type="pdf",
            parsed_data=ParsedResume(
                skills=["Python", "Django", "PostgreSQL", "REST APIs"],
                experience=[],
                education=[],
            ),
        ),
        "student-3": Resume(
            _id="resume-3",
            student_id="student-3",
            raw_text="Data scientist with strong background in ML...",
            file_type="pdf",
            parsed_data=ParsedResume(
                skills=["Python", "Machine Learning", "TensorFlow", "SQL"],
                experience=[],
                education=[],
            ),
        ),
    }


@pytest.fixture
def sample_applications(sample_students):
    """Create sample applications."""
    return [
        Application(
            _id="app-1",
            student_id="student-1",
            job_id="job-123",
            status="pending",
        ),
        Application(
            _id="app-2",
            student_id="student-2",
            job_id="job-123",
            status="pending",
        ),
        Application(
            _id="app-3",
            student_id="student-3",
            job_id="job-123",
            status="pending",
        ),
    ]


@pytest.fixture
def sample_screening_scores():
    """Create sample screening scores."""
    return [
        ScreeningScore(
            candidate_id="student-1",
            job_id="job-123",
            overall_score=0.85,
            matching_skills=["Python", "JavaScript", "AWS"],
            missing_skills=["SQL"],
            experience_match=0.8,
            education_match=0.9,
        ),
        ScreeningScore(
            candidate_id="student-2",
            job_id="job-123",
            overall_score=0.70,
            matching_skills=["Python", "SQL"],
            missing_skills=["JavaScript", "AWS"],
            experience_match=0.6,
            education_match=0.8,
        ),
        ScreeningScore(
            candidate_id="student-3",
            job_id="job-123",
            overall_score=0.65,
            matching_skills=["Python", "SQL"],
            missing_skills=["JavaScript", "AWS"],
            experience_match=0.5,
            education_match=0.9,
        ),
    ]
