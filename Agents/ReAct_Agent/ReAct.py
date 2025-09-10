from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

# The foundational class for all message types in LangGraph
from langchain_core.messages import BaseMessage
# passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import ToolMessage
# Message for providing system-level instructions to the LLM
from langchain_core.messages import SystemMessage

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()  # take environment variables from .env file

# Defination of imports
"""
# # Annotated - provides additional context or metadata about the type without affecting the type itself
# email = Annotated[str, "This has to be a valid email address!"]
# print(email.__metadata__)

# sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history
# Reducer function (add_messages) - Rule that controls/defines how updates from nodes are combined with the existing state/without a reducer, the new state would simply replace the old state
"""

# Define the state structure for the agent


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Create 1st Tool using the @tool decorator


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b


@tool
def subtract(a: int, b: int):
    """This is a subtraction function that subtracts 2 numbers"""
    return a - b


@tool
def multiply(a: int, b: int):
    """This is a multiplication function that multiplies 2 numbers"""
    return a * b


# List of tools available to the agent
tools = [add, subtract, multiply]

# Initialize the LLM and bind the tools to it
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You're my personal assistant, please answer my query to the best of you ability."
                                  )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# define conditional edge


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


# define the graph
graph = StateGraph(AgentState)
graph.add_node("agent_0", model_call)

# create a ToolNode
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent_0")

graph.add_conditional_edges(
    "agent_0",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# add an edge from the tool node back to the model call node
graph.add_edge("tools", "agent_0")

# compile the graph into an agent
app = graph.compile()

# Helper function for printing the stream of messages


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [
    ("user", "add 40 and 6 then multiply the result by 3 then subtract 10")]}
print_stream(app.stream(inputs, stream_mode="values"))
