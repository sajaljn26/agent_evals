from langchain_groq import ChatGroq
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
from deepeval.evaluate import CacheConfig
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.agent import AgentExecutor
from langchain.tools import tool

load_dotenv()

@tool
def get_weather(city: str):
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

model = ChatGroq(model="llama-3.1-8b-instant")
tools = [get_weather]
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
            input="What is the weather in London?",
            expected_output="The weather in London is sunny.",
        ),
    ]
)

for golden in dataset.goldens:
    result = agent_executor.invoke({"input": golden.input})
    
    # Extract tool calls from the intermediate steps
    tools_called = []
    for step in result.get("intermediate_steps", []):
        action = step[0]
        tools_called.append(ToolCall(name=action.tool, input_parameters=action.tool_input))

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result["output"],
        expected_output=golden.expected_output,
        tools_called=tools_called,
        expected_tools=[ToolCall(name="get_weather", input_parameters={"city": "London"})],
    )
    dataset.add_test_case(test_case)

metric = ToolCorrectnessMetric(
    threshold=0.9,
    model=model,
    evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
)

evaluate(test_cases=dataset.test_cases, metrics=[metric], cache_config=CacheConfig(use_cache=False, write_cache=False))
