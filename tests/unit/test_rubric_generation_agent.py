"""Unit tests for the RubricGenerationAgent."""

import json

import pytest
from unittest.mock import MagicMock

from src.agents.base import AgentContext
from src.agents.rubric_generation_agent import (
    RubricGenerationAgent,
    RubricGenerationInput,
    RubricGenerationOutput,
)
from src.models.scoring import DefaultCriteria


class TestRubricGenerationAgent:
    @pytest.fixture
    def agent(self):
        """Create an agent with a mocked LLM service."""
        mock_llm = MagicMock()

        # default behaviour: echo back the default rubric as JSON
        def fake_generate(prompt, system_prompt=None, max_tokens=None, temperature=None):
            base = DefaultCriteria.get_all()
            data = [
                {"name": c.name, "key": c.key, "weight": c.weight, "description": c.description}
                for c in base
            ]
            return json.dumps(data)

        mock_llm.generate = fake_generate
        mock_llm.initialize = MagicMock()

        agent = RubricGenerationAgent(llm_service=mock_llm)
        return agent

    @pytest.mark.asyncio
    async def test_returns_default_rubric(self, agent, sample_job, sample_rubric):
        input_data = RubricGenerationInput(job=sample_job)
        result = await agent.run(input_data, AgentContext(job_id=sample_job.id))

        assert result.success
        criteria = result.data.criteria
        assert len(criteria) == len(sample_rubric)
        # weights sum to 1
        assert abs(sum(c.weight for c in criteria) - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_custom_weights_from_llm(self, agent, sample_job):
        # override generate to return shifted weights
        def custom_generate(prompt, system_prompt=None, max_tokens=None, temperature=None):
            return json.dumps([
                {"name": "Education & Qualifications", "key": "education", "weight": 0.10, "description": "desc"},
                {"name": "Relevant Experience (Years)", "key": "experience", "weight": 0.30, "description": "desc"},
                {"name": "Technical Skills", "key": "technical_skills", "weight": 0.25, "description": "desc"},
                {"name": "Industry Knowledge", "key": "industry_knowledge", "weight": 0.15, "description": "desc"},
                {"name": "Leadership Experience", "key": "leadership", "weight": 0.10, "description": "desc"},
                {"name": "Communication Skills", "key": "communication", "weight": 0.10, "description": "desc"},
            ])

        agent.llm_service.generate = custom_generate
        result = await agent.run(RubricGenerationInput(job=sample_job), AgentContext(job_id=sample_job.id))
        assert result.success
        criteria = result.data.criteria
        weights = {c.key: c.weight for c in criteria}
        assert weights["education"] == pytest.approx(0.1)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self, agent, sample_job, sample_rubric):
        agent.llm_service.generate.side_effect = Exception("boom")
        result = await agent.run(RubricGenerationInput(job=sample_job), AgentContext(job_id=sample_job.id))
        assert result.success
        criteria = result.data.criteria
        # should equal base rubric when LLM fails
        assert [c.key for c in criteria] == [c.key for c in sample_rubric]