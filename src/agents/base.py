"""Base agent class for the multi-agent screening system."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings


# Type variables for input and output
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AgentContext(BaseModel):
    """Shared context passed between agents in the pipeline."""

    job_id: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Shared data between agents
    candidates: list[Any] = Field(default_factory=list)
    job: Optional[Any] = None
    scores: dict[str, Any] = Field(default_factory=dict)
    rankings: list[Any] = Field(default_factory=list)

    def set_data(self, key: str, value: Any) -> None:
        """Set data in context metadata."""
        self.metadata[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from context metadata."""
        return self.metadata.get(key, default)


class AgentResult(BaseModel, Generic[OutputT]):
    """Standard result wrapper for all agents."""

    success: bool
    data: Optional[Any] = None  # Will be OutputT at runtime
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    agent_name: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def success_result(
        cls,
        data: Any,
        agent_name: str,
        execution_time_ms: float = 0.0,
    ) -> "AgentResult":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            agent_name=agent_name,
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def failure_result(
        cls,
        errors: list[str],
        agent_name: str,
        execution_time_ms: float = 0.0,
    ) -> "AgentResult":
        """Create a failure result."""
        return cls(
            success=False,
            errors=errors,
            agent_name=agent_name,
            execution_time_ms=execution_time_ms,
        )


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    settings: Optional[Settings] = None

    class Config:
        arbitrary_types_allowed = True


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all screening agents.

    All agents in the multi-agent system inherit from this class.
    It provides:
    - Standardized execution interface
    - Error handling and retry logic
    - Metrics logging
    - Pre/post processing hooks
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        name: Optional[str] = None,
    ):
        """Initialize the agent.

        Args:
            config: Agent configuration. If None, default config is used.
            name: Agent name. Required if config is None.
        """
        if config is None:
            if name is None:
                raise ValueError("Either config or name must be provided")
            config = AgentConfig(name=name)

        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name
        self.settings = config.settings or get_settings()
        self.logger = logging.getLogger(f"agent.{self.name}")
        self._metrics: dict[str, Any] = {}

    @abstractmethod
    async def execute(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> AgentResult[OutputT]:
        """Main execution method - must be implemented by subclasses.

        Args:
            input_data: Input data specific to this agent
            context: Shared pipeline context

        Returns:
            AgentResult containing the output data or errors
        """
        pass

    async def run(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> AgentResult[OutputT]:
        """Run the agent with error handling, retries, and metrics.

        This is the main entry point that wraps execute() with:
        - Input validation
        - Pre-processing
        - Error handling and retries
        - Post-processing
        - Metrics logging

        Args:
            input_data: Input data specific to this agent
            context: Shared pipeline context

        Returns:
            AgentResult containing the output data or errors
        """
        start_time = time.perf_counter()

        self.logger.info(f"Starting agent: {self.name}")

        # Validate input
        try:
            if not self.validate_input(input_data):
                return AgentResult.failure_result(
                    errors=["Input validation failed"],
                    agent_name=self.name,
                )
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return AgentResult.failure_result(
                errors=[f"Validation error: {str(e)}"],
                agent_name=self.name,
            )

        # Pre-process
        try:
            processed_input = self.pre_process(input_data)
        except Exception as e:
            self.logger.error(f"Pre-processing error: {e}")
            return AgentResult.failure_result(
                errors=[f"Pre-processing error: {str(e)}"],
                agent_name=self.name,
            )

        # Execute with retries
        last_error: Optional[Exception] = None
        for attempt in range(self.config.retry_count):
            try:
                result = await asyncio.wait_for(
                    self.execute(processed_input, context),
                    timeout=self.config.timeout_seconds,
                )

                if result.success:
                    # Post-process
                    try:
                        result.data = self.post_process(result.data)
                    except Exception as e:
                        self.logger.error(f"Post-processing error: {e}")
                        result.warnings.append(f"Post-processing warning: {str(e)}")

                    # Calculate execution time
                    execution_time = (time.perf_counter() - start_time) * 1000
                    result.execution_time_ms = execution_time

                    # Log metrics
                    await self.log_metrics({
                        "execution_time_ms": execution_time,
                        "success": True,
                        "attempt": attempt + 1,
                    })

                    self.logger.info(
                        f"Agent {self.name} completed in {execution_time:.2f}ms"
                    )
                    return result

                # If not successful, log and potentially retry
                self.logger.warning(
                    f"Agent {self.name} attempt {attempt + 1} failed: {result.errors}"
                )
                last_error = Exception("; ".join(result.errors))

            except asyncio.TimeoutError:
                self.logger.error(
                    f"Agent {self.name} timed out after {self.config.timeout_seconds}s"
                )
                last_error = asyncio.TimeoutError(
                    f"Timeout after {self.config.timeout_seconds}s"
                )
            except Exception as e:
                self.logger.error(f"Agent {self.name} error: {e}")
                last_error = e

            # Wait before retry
            if attempt < self.config.retry_count - 1:
                await asyncio.sleep(self.config.retry_delay_seconds)

        # All retries failed
        execution_time = (time.perf_counter() - start_time) * 1000

        await self.log_metrics({
            "execution_time_ms": execution_time,
            "success": False,
            "attempts": self.config.retry_count,
        })

        error_msg = str(last_error) if last_error else "Unknown error"
        return AgentResult.failure_result(
            errors=[f"All {self.config.retry_count} attempts failed: {error_msg}"],
            agent_name=self.name,
            execution_time_ms=execution_time,
        )

    def validate_input(self, data: InputT) -> bool:
        """Validate input data before processing.

        Override in subclasses to add custom validation.

        Args:
            data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def pre_process(self, data: InputT) -> InputT:
        """Pre-process input data before execution.

        Override in subclasses to add custom preprocessing.

        Args:
            data: Input data to preprocess

        Returns:
            Preprocessed data
        """
        return data

    def post_process(self, result: OutputT) -> OutputT:
        """Post-process output data after execution.

        Override in subclasses to add custom postprocessing.

        Args:
            result: Output data to postprocess

        Returns:
            Postprocessed data
        """
        return result

    async def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log metrics for this agent execution.

        Override in subclasses to add custom metrics logging
        (e.g., to MLflow or WandB).

        Args:
            metrics: Dictionary of metrics to log
        """
        self._metrics.update(metrics)
        self.logger.debug(f"Metrics: {metrics}")

    def handle_error(self, error: Exception) -> None:
        """Handle an error during execution.

        Override in subclasses for custom error handling.

        Args:
            error: The exception that occurred
        """
        self.logger.error(f"Error in agent {self.name}: {error}")

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dictionary of collected metrics
        """
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset collected metrics."""
        self._metrics = {}

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name={self.name}, id={self.agent_id})"
