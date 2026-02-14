"""Application settings using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Resume Screening System"
    app_version: str = "0.1.0"
    debug: bool = False

    # TalentMatch API
    talentmatch_api_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for TalentMatch API",
    )
    talentmatch_api_timeout: int = Field(
        default=30,
        description="API request timeout in seconds",
    )

    # MongoDB
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI",
    )
    mongodb_database: str = Field(
        default="resume_screening",
        description="MongoDB database name",
    )

    # Flask API
    flask_host: str = "0.0.0.0"
    flask_port: int = 8000
    flask_debug: bool = False

    # Ollama LLM Settings (Local)
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API URL",
    )
    ollama_model: str = Field(
        default="llama3:latest",
        description="Ollama model name (e.g., llama3:latest, mistral:latest)",
    )

    # Legacy LLM Settings (Hugging Face - fallback)
    llm_model_name: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        description="Hugging Face model name for Llama",
    )
    llm_device: str = Field(
        default="auto",
        description="Device for model inference (auto, cuda, cpu)",
    )
    llm_max_tokens: int = 512
    llm_temperature: float = 0.3
    llm_use_8bit: bool = True  # Enable 8-bit quantization for speed

    # Embedding Service
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    embedding_batch_size: int = 16
    embedding_cache_enabled: bool = True

    # Screening Settings
    screening_min_score_threshold: float = 0.5
    screening_top_k: int = 10

    # Ranking Weights
    weight_skills_match: float = 0.35
    weight_experience: float = 0.20
    weight_education: float = 0.15
    weight_projects: float = 0.15
    weight_certifications: float = 0.10
    weight_soft_skills: float = 0.05

    # Fairness Settings
    fairness_dir_threshold: float = 0.8  # Disparate Impact Ratio threshold
    fairness_variance_threshold: float = 0.001  # Max allowed variance
    fairness_protected_attributes: list[str] = Field(
        default=["gender", "age_group", "ethnicity", "nationality"],
        description="Attributes protected from bias",
    )

    # Validation Settings
    validation_agreement_threshold: float = 0.90  # 90% agreement target

    # Performance Settings
    max_concurrent_candidates: int = 4  # Parallel processing limit
    pipeline_timeout_seconds: int = 1800  # 30 minutes
    batch_size: int = 50

    # Experiment Tracking
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "resume-screening"
    wandb_project: Optional[str] = None
    wandb_enabled: bool = False

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def get_weight_config(self) -> dict[str, float]:
        """Return weight configuration as a dictionary."""
        return {
            "skills_match": self.weight_skills_match,
            "experience": self.weight_experience,
            "education": self.weight_education,
            "projects": self.weight_projects,
            "certifications": self.weight_certifications,
            "soft_skills": self.weight_soft_skills,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
