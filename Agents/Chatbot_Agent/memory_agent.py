from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
# used to store secret API keys in .env file and environment configuration
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

# Define the state structure for the agent's memory


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# Create the Node functions


def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}\n")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    # print(result["messages"])
    conversation_history = result["messages"]

    user_input = input("Enter: ")


# Store the conversation history in memory/simple text file for future reference (Recommended is a vector database)
    # Allow the agent to access the memory even afer the conversation has ended(exit command)
with open("logging.txt", "w") as file:
    file.write("Your conversation log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Human: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation.\n")

print("Conversation history saved to logging.txt")


# To save on token costs, write a code to remove the first 2 messages in the conversation history if the length of the conversation history exceeds 10 messages.
if len(conversation_history) > 10:
    # Remove the first 2 messages
    conversation_history = conversation_history[2:]
