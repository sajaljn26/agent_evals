from .models import model
from .tools import calculate_right_refund, get_today_date, issue_refund_ticket, get_user_purchase_history
from dotenv import load_dotenv
from langchain.agents import create_agent
from .prompts import agent_system_prompt

load_dotenv()

tools = [calculate_right_refund, get_user_purchase_history, issue_refund_ticket, get_today_date]

# Use the new create_agent API (LangChain 1.4.5+)
# This returns a compiled graph that handles the tool-calling loop
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=agent_system_prompt,
)

def run_conversation():
    text = "I would like to request a refund"
    print(f"User: {text}")
    
    conversation = True
    current_input = text
    messages = [{"role": "user", "content": text}]

    while conversation:
        response = agent.invoke({"messages": messages})
        
        # Get the last AI message
        ai_messages = [m for m in response["messages"] if m.type == "ai"]
        if ai_messages:
            answer_text = ai_messages[-1].content
            print(f"Agent: {answer_text}")
            messages = response["messages"]
        else:
            break

        if "issued" in answer_text.lower() or "not eligible" in answer_text.lower():
            conversation = False
        else:
            user_input = input("reply: \n")
            messages.append({"role": "user", "content": user_input})

    print("the job is done!")

if __name__ == "__main__":
    run_conversation()