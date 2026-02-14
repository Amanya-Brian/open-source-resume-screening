"""Tests for the Ranking Agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.base import AgentContext
from src.agents.ranking_agent import RankingAgent, RankingInput
from src.models.schemas import Recommendation


class TestRankingAgent:
    """Tests for RankingAgent."""

    @pytest.fixture
    def agent(self):
        """Create a ranking agent instance."""
        return RankingAgent()

    @pytest.fixture
    def context(self):
        """Create a pipeline context."""
        return AgentContext(job_id="job-123")

    @pytest.mark.asyncio
    async def test_ranking_sorts_by_score(self, agent, context, sample_screening_scores):
        """Test that candidates are ranked by score in descending order."""
        input_data = RankingInput(scores=sample_screening_scores)
        result = await agent.run(input_data, context)

        assert result.success
        rankings = result.data.ranking_result.ranked_candidates

        # Verify descending order
        scores = [r.score for r in rankings]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_ranking_assigns_correct_ranks(self, agent, context, sample_screening_scores):
        """Test that ranks are assigned correctly (1-based)."""
        input_data = RankingInput(scores=sample_screening_scores)
        result = await agent.run(input_data, context)

        rankings = result.data.ranking_result.ranked_candidates
        ranks = [r.rank for r in rankings]

        assert ranks == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_ranking_calculates_percentiles(self, agent, context, sample_screening_scores):
        """Test percentile calculation."""
        input_data = RankingInput(scores=sample_screening_scores)
        result = await agent.run(input_data, context)

        rankings = result.data.ranking_result.ranked_candidates

        # Top candidate should have highest percentile
        assert rankings[0].percentile > rankings[-1].percentile

    @pytest.mark.asyncio
    async def test_ranking_assigns_recommendations(self, agent, context, sample_screening_scores):
        """Test recommendation assignment based on score."""
        input_data = RankingInput(scores=sample_screening_scores)
        result = await agent.run(input_data, context)

        rankings = result.data.ranking_result.ranked_candidates

        # Student-1 with 0.85 should be HIGHLY_RECOMMENDED or RECOMMENDED
        top_rec = rankings[0].recommendation
        assert top_rec in [Recommendation.HIGHLY_RECOMMENDED, Recommendation.RECOMMENDED]

    @pytest.mark.asyncio
    async def test_ranking_respects_top_k(self, agent, context, sample_screening_scores):
        """Test that top_k limits results."""
        input_data = RankingInput(scores=sample_screening_scores, top_k=2)
        result = await agent.run(input_data, context)

        rankings = result.data.ranking_result.ranked_candidates
        assert len(rankings) == 2

    @pytest.mark.asyncio
    async def test_ranking_handles_empty_scores(self, agent, context):
        """Test handling of empty scores list."""
        input_data = RankingInput(scores=[])
        result = await agent.run(input_data, context)

        assert result.success
        assert result.data.ranking_result.total_candidates == 0

    def test_get_recommendation_highly_recommended(self, agent):
        """Test recommendation for high scores."""
        rec = agent._get_recommendation(0.90, 95)
        assert rec == Recommendation.HIGHLY_RECOMMENDED

    def test_get_recommendation_recommended(self, agent):
        """Test recommendation for medium-high scores."""
        rec = agent._get_recommendation(0.75, 70)
        assert rec == Recommendation.RECOMMENDED

    def test_get_recommendation_consider(self, agent):
        """Test recommendation for medium scores."""
        rec = agent._get_recommendation(0.55, 50)
        assert rec == Recommendation.CONSIDER

    def test_get_recommendation_not_recommended(self, agent):
        """Test recommendation for low scores."""
        rec = agent._get_recommendation(0.35, 20)
        assert rec == Recommendation.NOT_RECOMMENDED
