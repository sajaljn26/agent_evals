import sys
from pathlib import Path
from uuid import uuid4
from typing import List, Optional

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import CacheConfig, DisplayConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall, ToolCallParams
from langchain_groq import ChatGroq

from .dataset import EVAL_DATASET
from .model import eval_model


class GroqJudgeModel(DeepEvalBaseLLM):
    """Custom DeepEval model that uses Groq directly without logprobs."""
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        self._model = ChatGroq(model=model_name, temperature=0.0)
    
    def load_model(self):
        return self._model
    
    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).content
    
    async def a_generate(self, prompt: str) -> str:
        return (await self._model.ainvoke(prompt)).content
    
    def get_model_name(self) -> str:
        return self.model_name


SUCCESSFUL_TASK_DESCRIPTION = """
If the username is missing, ask the user to provide it.
For a refund request, identify the customer and the requested purchases, and
check that each purchase is no more than 30 days old and is not in the future.
Refund only the items requested by the user. If the customer or requested item
cannot be found, do not issue a refund ticket and explain why.
Once the necessary information is available, tell the user whether a refund
ticket was issued and its amount, or explain why no refund was issued.
For an information-only request, answer the question without issuing a ticket.
"""

def build_metrics():
    """Define the 7 metrics used to evaluate the agent."""
    # Use our custom Groq model to avoid logprobs issues
    eval_model = GroqJudgeModel(model_name="openai/gpt-oss-20b")
    
    trace_metrics = [
        TaskCompletionMetric(
            threshold=0.8,
            model=eval_model,
            task=SUCCESSFUL_TASK_DESCRIPTION,
            include_reason=True,
        ),
        StepEfficiencyMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
        PlanQualityMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
        PlanAdherenceMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
    ]
    single_turn_metrics = [
        ArgumentCorrectnessMetric(
            threshold=None, 
            model=eval_model,
            include_reason=True,
        ),
        ToolCorrectnessMetric(
            threshold=0.9, 
            model=eval_model,
            include_reason=True,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        ),
        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Check that the actual output and expected output have the same "
                "logical conclusion about the user's request.",
            ],
            threshold=0.8,
            model=eval_model,
            verbose_mode=True,
        ),
    ]
    return trace_metrics, single_turn_metrics

def main():
    load_dotenv()
    # Import agent here to avoid circular imports
    from agent.agent import agent

    trace_metrics, single_turn_metrics = build_metrics()
    cache_config = CacheConfig(use_cache=False, write_cache=False)
    display_config = DisplayConfig(inspect_after_run=False)
    test_cases = []

    # We iterate through our Golden Dataset
    for golden in EVAL_DATASET:
        # The CallbackHandler captures the internal trace (thoughts/tool calls)
        # required by the agentic metrics.
        response = agent.invoke(
            {"messages": [{"role": "user", "content": golden.input}]},
            config={
                "configurable": {"thread_id": str(uuid4())},
                "callbacks": [CallbackHandler()],
            },
        )

        tools_called = []
        # Extract the tool calls from the agent's message history
        for message in response["messages"]:
            for tool_call in getattr(message, "tool_calls", []) or []:
                tools_called.append(
                    ToolCall(
                        name=tool_call["name"],
                        input_parameters=tool_call["args"],
                    )
                )

        # Create a DeepEval test case for each scenario
        # Get the final AI message content
        ai_messages = [m for m in response["messages"] if m.type == "ai"]
        final_answer = ai_messages[-1].content if ai_messages else ""
        
        # Convert expected_tools (list of dicts) to ToolCall objects
        expected_tool_calls = [
            ToolCall(name=tool["name"], input_parameters=tool.get("input_parameters", {}))
            for tool in golden.expected_tools
        ]
        
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=final_answer,
                expected_output=golden.expected_output,
                tools_called=tools_called,
                expected_tools=expected_tool_calls,
            )
        )

    # Execute the evaluation using the single-turn metrics
    evaluate(
        test_cases=test_cases,
        metrics=single_turn_metrics,
        cache_config=cache_config,
        display_config=display_config,
    )
    print(f"\nCompleted evaluation for {len(test_cases)} test cases.")

if __name__ == "__main__":
    main()
