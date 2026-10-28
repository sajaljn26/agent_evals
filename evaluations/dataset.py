from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class EvalCase:
    """Represents a single test scenario for the agent."""
    input: str
    expected_output: str
    expected_tools: List[Dict[str, Any]]  # List of {name, input_parameters}
    context: Optional[str] = None

# The Golden Dataset
EVAL_DATASET = [
    EvalCase(
        input="I am user123 and I want a refund for my book.",
        expected_output="a refund ticket with the amount 23 was issued and should be arriving soon to your bank account",
        expected_tools=[
            {"name": "get_user_purchase_history", "input_parameters": {"user": "user123"}},
            {"name": "get_today_date", "input_parameters": {}},
            {"name": "calculate_right_refund", "input_parameters": {"user": "user123", "items": [{"name": "book"}]}},
            {"name": "issue_refund_ticket", "input_parameters": {"user": "user123", "refund_amount": 23}},
        ],
        context="User123 bought a book for 23 on 2026-10-5. This is eligible."
    ),
    EvalCase(
        input="I am user125 and I want a refund for my ruler.",
        expected_output="no refund will be issued because you are not eligible due to the purchase being more than 30 days ago",
        expected_tools=[
            {"name": "get_user_purchase_history", "input_parameters": {"user": "user125"}},
            {"name": "get_today_date", "input_parameters": {}},
        ],
        context="User125 bought a ruler on 2026-9-24. If today is Oct 2026, this is too old."
    ),
    EvalCase(
        input="I want a refund for my mouse.",
        expected_output="please provide your username or email so I can locate your order and process the refund",
        expected_tools=[],
        context="The agent should realize the username is missing and ask for it."
    ),
    EvalCase(
        input="I am user131 and I want a refund for my desk lamp and mouse.",
        expected_output="a refund ticket with the amount 48 was issued and should be arriving soon to your bank account",
        expected_tools=[
            {"name": "get_user_purchase_history", "input_parameters": {"user": "user131"}},
            {"name": "get_today_date", "input_parameters": {}},
            {"name": "calculate_right_refund", "input_parameters": {"user": "user131", "items": [{"name": "desk lamp"}, {"name": "mouse"}]}},
            {"name": "issue_refund_ticket", "input_parameters": {"user": "user131", "refund_amount": 48}},
        ],
        context="User131: desk lamp (28) + mouse (20) = 48."
    ),
]
