from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# The Judge model is used by DeepEval to grade the agent.
# We use the same fast model here, but in production, 
# you would often use a larger model for the judge.
eval_model = ChatGroq(model="llama-3.1-8b-instant")
