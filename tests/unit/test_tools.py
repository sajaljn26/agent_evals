import pytest
from agent.tools import (
    get_user_purchase_history,
    calculate_right_refund,
    issue_refund_ticket,
    get_today_date,
)
from agent.objects import PurchaseItem


class TestTools:
    """Unit tests for refund tools."""

    def test_get_user_purchase_history_valid_user(self):
        result = get_user_purchase_history.invoke({"user": "user123"})
        assert "book" in str(result).lower()
        assert "23" in str(result)

    def test_get_user_purchase_history_invalid_user(self):
        result = get_user_purchase_history.invoke({"user": "nonexistent"})
        assert "no username found" in str(result).lower()

    def test_get_today_date_returns_iso_format(self):
        result = get_today_date.invoke({})
        # Should be YYYY-MM-DD format
        assert len(result) == 10
        assert result.count("-") == 2

    def test_calculate_right_refund_valid_items(self):
        items = [PurchaseItem(name="book")]
        result = calculate_right_refund.invoke({"user": "user123", "items": items})
        assert "23" in str(result)

    def test_calculate_right_refund_multiple_items(self):
        items = [PurchaseItem(name="desk_lamp"), PurchaseItem(name="mouse")]
        result = calculate_right_refund.invoke({"user": "user131", "items": items})
        assert "48" in str(result)

    def test_issue_refund_ticket_returns_confirmation(self):
        result = issue_refund_ticket.invoke({"user": "user123", "refund_amount": 23})
        assert "issued" in str(result).lower()
        assert "23" in str(result)


class TestPurchaseItem:
    """Test Pydantic models."""

    def test_purchase_item_creation(self):
        item = PurchaseItem(name="book")
        assert item.name == "book"

    def test_purchase_item_validation(self):
        with pytest.raises(ValueError):
            PurchaseItem(name="")  # Empty name should fail