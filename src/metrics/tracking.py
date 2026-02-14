"""Experiment tracking with MLflow and WandB."""

import logging
from datetime import datetime
from typing import Any, Optional

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Unified interface for experiment tracking.

    Supports MLflow and Weights & Biases (WandB) for:
    - Parameter logging
    - Metrics logging
    - Artifact storage
    - Experiment comparison
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize the metrics tracker.

        Args:
            experiment_name: Name for the experiment
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self.experiment_name = experiment_name or self.settings.mlflow_experiment_name

        self._mlflow_enabled = bool(self.settings.mlflow_tracking_uri)
        self._wandb_enabled = self.settings.wandb_enabled

        self._mlflow = None
        self._wandb = None
        self._run_id: Optional[str] = None

        self._initialize_trackers()

    def _initialize_trackers(self) -> None:
        """Initialize tracking backends."""
        if self._mlflow_enabled:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow = mlflow
                logger.info(f"MLflow initialized: {self.settings.mlflow_tracking_uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
                self._mlflow_enabled = False

        if self._wandb_enabled:
            try:
                import wandb
                self._wandb = wandb
                logger.info(f"WandB initialized: {self.settings.wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self._wandb_enabled = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new tracking run.

        Args:
            run_name: Name for the run
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self._mlflow_enabled:
            self._mlflow.start_run(run_name=run_name)
            self._run_id = self._mlflow.active_run().info.run_id
            logger.info(f"Started MLflow run: {self._run_id}")

        if self._wandb_enabled:
            self._wandb.init(
                project=self.settings.wandb_project,
                name=run_name,
            )

    def end_run(self) -> None:
        """End the current tracking run."""
        if self._mlflow_enabled:
            self._mlflow.end_run()

        if self._wandb_enabled and self._wandb.run:
            self._wandb.finish()

        self._run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters/configuration.

        Args:
            params: Dictionary of parameters
        """
        if self._mlflow_enabled:
            # MLflow requires string values
            str_params = {k: str(v) for k, v in params.items()}
            self._mlflow.log_params(str_params)

        if self._wandb_enabled and self._wandb.run:
            self._wandb.config.update(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
        """
        if self._mlflow_enabled:
            self._mlflow.log_metrics(metrics, step=step)

        if self._wandb_enabled and self._wandb.run:
            self._wandb.log(metrics, step=step)

    def log_screening_run(
        self,
        job_id: str,
        total_candidates: int,
        processing_time_seconds: float,
        agreement_rate: Optional[float] = None,
        dir_score: Optional[float] = None,
        is_compliant: Optional[bool] = None,
    ) -> None:
        """Log a complete screening run.

        Args:
            job_id: Job ID
            total_candidates: Number of candidates screened
            processing_time_seconds: Total processing time
            agreement_rate: Validation agreement rate
            dir_score: Disparate Impact Ratio
            is_compliant: Fairness compliance status
        """
        metrics = {
            "total_candidates": total_candidates,
            "processing_time_seconds": processing_time_seconds,
            "candidates_per_second": total_candidates / max(processing_time_seconds, 1),
        }

        if agreement_rate is not None:
            metrics["agreement_rate"] = agreement_rate

        if dir_score is not None:
            metrics["disparate_impact_ratio"] = dir_score

        if is_compliant is not None:
            metrics["is_fairness_compliant"] = int(is_compliant)

        self.log_params({"job_id": job_id})
        self.log_metrics(metrics)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact file.

        Args:
            local_path: Path to local file
            artifact_path: Path in artifact storage
        """
        if self._mlflow_enabled:
            self._mlflow.log_artifact(local_path, artifact_path)

        if self._wandb_enabled and self._wandb.run:
            self._wandb.save(local_path)

    def __enter__(self) -> "MetricsTracker":
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.end_run()
