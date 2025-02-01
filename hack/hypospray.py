import subprocess
import requests
import sys

from langchain_core.messages import AIMessage
from typing import List

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

#graph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
#from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from typing import Literal

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError

namespace = sys.argv[1]

def kubectl_command(command, namespace="default"):
    cmd = command.split(" ")
    try:
        # Use subprocess.run to execute the command and capture its output.
        result = subprocess.run(
            ["kubectl", "-n", namespace ] + cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        # Decode the output from bytes to a string.
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        # If kubectl returns a non-zero exit code, print the error message and return it as a string.
        return f"Command failed with status {e.returncode}:\n{e.output.decode('utf-8')}"

# Example usage
command = "get all"
k_get_all_output = kubectl_command(command, namespace=namespace)
#print(output)

@tool
def kubectl_get_events() -> str:
    """Get kubernetes events from all pods and deployments in a namespace with kubectl

    """
    print("ran kube get events")
    return kubectl_command("get events", namespace=namespace)

@tool
def kubectl_get(resource_name: str) -> str:
    """Run kubectl get <type>/<name> output

    Examples:
        kubectl_get("pod/my-pod")
        kubectl_get("deployment/mydeploy")
        kubectl_get("service/mysvc")

    Args:
        resource_name (str): The resource type combined with the name
    """
    print(f"ran kube get {resource_name}")
    if ";" in resource_name:
        return "error"
    return kubectl_command(f"get {resource_name}", namespace=namespace)

@tool
def kubectl_describe(resource_name: str) -> str:
    """Run kubectl describe <type>/<name> output

    Examples:
        kubectl_describe("pod/my-pod")
        kubectl_describe("deployment/mydeploy")
        kubectl_describe("service/mysvc")

    Args:
        resource_name (str): The resource type combined with the name
    """
    print(f"ran kube describe {resource_name}")
    if ";" in resource_name:
        return "error"
    return kubectl_command(f"describe {resource_name}", namespace=namespace)

@tool
def kubectl_logs(pod_name: str) -> str:
    """Get kubectl logs <pod_name> --tail=25 output (last 25 lines only)

    Args:
        pod_name (str): The name of the pod
    """
    print(f"ran kube logs {pod_name}")
    if " " in pod_name:
        return "error"
    if ";" in pod_name:
        return "error"
    return kubectl_command(f"logs {pod_name} --tail=25", namespace=namespace)

@tool
def kubectl_describe_pod(pod_name: str) -> str:
    """Get kubectl describe pod <pod_name> output

    Args:
        pod_name (str): The name of the pod
    """
    print(f"ran kube describe pod {pod_name}")
    if " " in pod_name:
        return "error"
    if ";" in pod_name:
        return "error"
    return kubectl_command(f"describe pod {pod_name}", namespace=namespace)

llm_tools = [kubectl_get_events, kubectl_get, kubectl_logs, kubectl_describe]

llm = ChatOllama(
    model="llama3.1:8b-instruct-fp16",
    base_url="http://10.80.50.15:11434",
    temperature=0,
    # other params...
).bind_tools(llm_tools)

tool_node = ToolNode(llm_tools)



#messages = [
#    (
#        "system",
#        "You are a helpful devops infrastructure assistant. Given this kubectl output, can you see anything wrong? What is the root issue? Fully explain all errors that you see."
#    ),
#    ("human", f"{output}"),
#]


#{"role": "user", "content": "what is the weather in sf"}]
messages = {"role": "user", "content": f"As a devops infrastructure & containers expert, read the following kubectl output and determine if anything is wrong. What is the root issue? You can run a function, get the result, and then run another function until you are satisfied. Explain and debug the issue. If any. {k_get_all_output}"}
print("System Prompt:", messages['content'])
#ai_msg = llm.invoke(messages)


#print(ai_msg)
#print(ai_msg.tool_calls)




# Tutorial stuff
#app = create_react_agent(llm, llm_tools, checkpointer=checkpointer)
# Use the agent
#final_state = app.invoke(
#        #    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
#    {"messages": messages},
#    config={"configurable": {"thread_id": 42}}
#)
#print(final_state["messages"][-1].content)
#final_state = app.invoke(
#    {"messages": [{"role": "user", "content": "Why are the pods crashing?"}]},
#    config={"configurable": {"thread_id": 42}}
#)
#print(final_state["messages"][-1].content)





# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the agent (og)
#try:
#    final_state = app.invoke(
#        {"messages": messages},
#        config={"configurable": {"thread_id": 42}, "recursion_limit": 8}
#    )
#except GraphRecursionError:
#    print("Recursion Error")
#print(final_state["messages"][-1].content)

#user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
counter=0
try:
    events = app.stream(
        {"messages": messages},
        config={"configurable": {"thread_id": 42}, "recursion_limit": 20},
        stream_mode="values"
    )
    for event in events:
        print("***")
        tot = len(event["messages"])
        print("Total messages: ", tot)
        for e in event["messages"][counter:]:
            e.pretty_print()
        counter = tot
except GraphRecursionError:
    print("Recursion Error")

