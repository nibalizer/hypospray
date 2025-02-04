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
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
# https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/#define-graph_1


namespace = sys.argv[1]
ollama_url = "http://localhost:11434"

def ping_ollama(url):
    try:
        response = requests.head(url, timeout=5)
        return f"Status Code: {response.status_code}"
    except requests.Timeout:
        return "Request Timed Out"
    except requests.RequestException as e:
        return f"Request Error: {e}"

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
        kubectl_get("pods")
        kubectl_get("pod")
        kubectl_get("pod/my-pod")
        kubectl_get("deployment/mydeploy")
        kubectl_get("service/mysvc")

    Args:
        resource_name (str): The resource type combined with the name
    """
    print(f"ran kube get {resource_name}")
    if ";" in resource_name:
        return "error"
    if "secret" in resource_name:
        return "error"
    if "configmap" in resource_name:
        return "error"
    if "cm" in resource_name:
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



class HyposprayResponse(BaseModel):
    """Respond to the user with this"""

    green: bool = Field(description="The status of the environment, green if everything looks good, red if anything looks wrong")
    erroring_resources: str = Field(description="List of Kubernetes objects that are in causing the issue, examples: deployment/my-app, pod/my-app-1234, svc/my-service, node/node02")


# Inherit 'messages' key from MessagesState, which is a list of chat messages
class AgentState(MessagesState):
    # Final structured response from the agent
    final_response: HyposprayResponse



# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if not last_message.tool_calls:
        return "respond"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function that responds to the user
def respond(state: AgentState):
    # We call the model with structured output in order to return the same format to the user every time
    # state['messages'][-2] is the last ToolMessage in the convo, which we convert to a HumanMessage for the model to use
    # We could also pass the entire chat history, but this saves tokens since all we care to structure is the output of the tool
    response = llm_with_structured_output.invoke(
        [HumanMessage(content=state["messages"][-2].content)]
    )
    # We return the final answer
    return {"final_response": response}


llm_tools = [kubectl_get_events, kubectl_get, kubectl_logs, kubectl_describe]

base_llm = ChatOllama(
    model="llama3.1:8b-instruct-fp16",
    #base_url="http://10.80.50.15:11434",
    #base_url="http://localhost:11434",
    base_url=ollama_url,
    temperature=0,
    # other params...
)



#messages = [
#    (
#        "system",
#        "You are a helpful devops infrastructure assistant. Given this kubectl output, can you see anything wrong? What is the root issue? Fully explain all errors that you see."
#    ),
#    ("human", f"{output}"),
#]


#{"role": "user", "content": "what is the weather in sf"}]
#ai_msg = base_llm.invoke(messages)


#print(ai_msg)
#print(ai_msg.tool_calls)




llm_with_tools =  base_llm.bind_tools(llm_tools)
llm_with_structured_output = base_llm.with_structured_output(HyposprayResponse)

tool_node = ToolNode(llm_tools)

# Define a new graph
#workflow = StateGraph(MessagesState)
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")
#workflow.set_entry_point("agent")
# ^^ probably equivalent?

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",

    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')
workflow.add_edge("respond", END)

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
#print(app.get_graph().draw_mermaid())
import sys
#sys.exit(1)

# Example usage
#print("ping ollama")
ping_ollama(ollama_url)
command = "get all"
k_get_all_output = kubectl_command(command, namespace=namespace)

start_messages = {"role": "user", "content": f"As a devops infrastructure & containers expert, read the following kubectl output and determine if anything is wrong. Remember that you'll have to run kubectl get commands to discover the names of resource that you can then inspect in detail in a later tool call. What is the root issue? You can run a function, get the result, and then run another function until you are satisfied. Explain and debug the issue, if any. Clearly state if there is an issue and provide a list of kubernetes objects that are in an error state. \n{k_get_all_output}"}
#print("Start Messages:", start_messages['content'])
#print(output)
"""
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

"""


answer = app.invoke(input={"messages": start_messages},
                      config={"configurable": {"thread_id": 42}, "recursion_limit": 40},
                      )

print("Total messages:", len(answer['messages']))
print(answer[ "final_response" ])
#from pdb import set_trace; set_trace()




