from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

# Iports for message types
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()  # take environment variables from .env file

# This is a global variable to store document content
document_content = ""

# Define the state structure for the agent


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Create 1st Tool using the @tool decorator


@tool
def update(content: str):
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated succesfully! The current content is:\n{document_content}."


@tool
def save(filename: str) -> str:
    """Saves the current document content to a text file and finish the process.
    Args:
        filename: The name of the text file.
    """
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"\n🗃️Document saved successfully to: {filename}")
        return f"Document saved successfully to '{filename}'."
    except Exception as e:
        return f"Failed to save document: {str(e)}"


tools = [update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def agent_0(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
    You are a Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, use the 'save' tool.
    - Make sure to always show the current document state after modifications.
                                  
    The current socument content is: {document_content}                                                              
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do withthe document?")
        print(f"\n 👨‍💻 USER: {user_input}\n")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    # pretty print the response
    print(f"\n🤖 DRAFTER: {response.content}\n")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(
            f"\n🛠️ USING TOOL: {[tc['name'] for tc in response.tool_calls]}:")

    return {"messages": list(state["messages"]) + [user_message, response]}


# create the conditional edge function
def should_continue(state: AgentState):
    """Determine if we should continue or end the conversation"""
    messages = state["messages"]

    if not messages:
        return "continue"

    # This looks for the most recent tool message ...
    for message in reversed(messages):
        # ... and checks if it is a ToolMessage resulting from the 'save' tool
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
                "document" in message.content.lower()):
            return "end"  # End if document has been saved by going to the end edge

    return "continue"  # Otherwise, continue the conversation


# Pretify the print statemt
def print_messages(messages):
    """Helper function to print the messages in a readable format"""
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULTS: {message.content}")


# CREATE THE GRAPH *******************************
graph = StateGraph(AgentState)

# Add the nodes
graph.add_node("agent_0", agent_0)
graph.add_node("tools", ToolNode(tools))


graph.set_entry_point("agent_0")

# Add the edges
graph.add_edge("agent_0", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent_0",
        "end": END,
    },
)

# compile the graph
app = graph.compile()


def run_document_agent():
    print("\n === DRAFTER AGENT === \n")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])


if __name__ == "__main__":
    run_document_agent()
