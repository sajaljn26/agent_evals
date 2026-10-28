from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Using gpt-oss-120b as requested
AGENT_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "openai/gpt-oss-120b"

# Initialize the LLM instance for the agent
model = ChatGroq(model=AGENT_MODEL, temperature=0.0)