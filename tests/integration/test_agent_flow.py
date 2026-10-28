import pytest
from agent.agent import agent


class TestAgentFlow:
    """Integration tests for the full agent flow."""

    @pytest.mark.asyncio
    async def test_eligible_refund_flow(self):
        """Test successful refund for eligible purchase."""
        response = agent.invoke({
            "messages": [{"role": "user", "content": "I am user123 and I want a refund for my book"}]
        })
        
        final_msg = response["messages"][-1].content.lower()
        assert "refund ticket" in final_msg
        assert "23" in final_msg
        assert "issued" in final_msg

    @pytest.mark.asyncio
    async def test_ineligible_refund_flow(self):
        """Test rejection for purchase older than 30 days."""
        response = agent.invoke({
            "messages": [{"role": "user", "content": "I am user125 and I want a refund for my ruler"}]
        })
        
        final_msg = response["messages"][-1].content.lower()
        assert "not eligible" in final_msg or "cannot be issued" in final_msg
        assert "30 days" in final_msg or "more than 30" in final_msg

    @pytest.mark.asyncio
    async def test_missing_username_flow(self):
        """Test agent asks for username when not provided."""
        response = agent.invoke({
            "messages": [{"role": "user", "content": "I want a refund for my mouse"}]
        })
        
        final_msg = response["messages"][-1].content.lower()
        assert "username" in final_msg or "email" in final_msg

    @pytest.mark.asyncio
    async def test_multi_item_refund_flow(self):
        """Test refund for multiple items."""
        response = agent.invoke({
            "messages": [{"role": "user", "content": "I am user131 and I want a refund for my desk lamp and mouse"}]
        })
        
        final_msg = response["messages"][-1].content.lower()
        assert "refund ticket" in final_msg
        assert "48" in final_msg
        assert "issued" in final_msg

    @pytest.mark.asyncio
    async def test_tool_calls_captured(self):
        """Verify agent uses tools correctly."""
        response = agent.invoke({
            "messages": [{"role": "user", "content": "I am user123 and I want a refund for my book"}]
        })
        
        # Check that tools were called
        tool_calls = []
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)
        
        tool_names = [tc["name"] for tc in tool_calls]
        assert "get_user_purchase_history" in tool_names
        assert "get_today_date" in tool_names
        assert "calculate_right_refund" in tool_names
        assert "issue_refund_ticket" in tool_names