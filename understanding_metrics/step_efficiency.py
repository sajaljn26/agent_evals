from langchain_groq import ChatGroq
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import StepEfficiencyMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import CacheConfig
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.agent import AgentExecutor
from langchain.tools import tool

load_dotenv()

@tool
def get_info(topic: str):
    """Get info about a topic."""
    return f"Info about {topic}: it is interesting."

model = ChatGroq(model="llama-3.1-8b-instant")
tools = [get_info]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="Tell me about AI.",
            expected_output="AI is interesting.",
        ),
    ]
)

for golden in dataset.goldens:
    result = agent_executor.invoke({"input": golden.input})
    # In a real scenario, we would capture the trace. 
    # DeepEval's StepEfficiencyMetric requires a trace.
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result["output"],
        expected_output=golden.expected_output,
    )
    dataset.add_test_case(test_case)

metric = StepEfficiencyMetric(threshold=0.8, model=model)

evaluate(test_cases=dataset.test_cases, metrics=[metric], cache_config=CacheConfig(use_cache=False, write_cache=False))
