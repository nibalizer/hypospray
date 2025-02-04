import subprocess
from typing import List, Tuple
from typing_extensions import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

def _kubectl_command(command, namespace="default"):
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

def kubectl_start_data(namespace="default"):
    out = "\n"
    out += "> kubectl get all\n"
    out += _kubectl_command("get all", namespace=namespace)
    out += "> kubectl get events\n"
    out += _kubectl_command("get events", namespace=namespace)
    out += "> kubectl get ingress\n"
    out += _kubectl_command("get ingress", namespace=namespace)
#    out += "kubectl get nodes --show-labels\n"
#    out += _kubectl_command("get nodes --show-labels", namespace=namespace)
#    out += "kubectl version\n"
#    out += _kubectl_command("version", namespace=namespace)
    return out


@tool
def kubectl_get_events(state: Annotated[dict, InjectedState]) -> str:
    """Get kubernetes events from all pods and deployments in a namespace with kubectl

    """
    namespace = state["namespace"]
    print("ran kube get events")
    # TODO raise value error here
    # TODO detect return code and raise value error when kubectl errors
    return _kubectl_command("get events", namespace=namespace)

@tool
def kubectl_get(resource_name: str, state: Annotated[dict, InjectedState]) -> str:
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
    namespace = state["namespace"]
    print(f"ran kube get {resource_name}")
    if ";" in resource_name:
        return "error"
    if "secret" in resource_name:
        return "error"
    if "configmap" in resource_name:
        return "error"
    if "cm" in resource_name:
        return "error"
    return _kubectl_command(f"get {resource_name}", namespace=namespace)

@tool
def kubectl_describe(resource_name: str, state: Annotated[dict, InjectedState]) -> str:
    """Run kubectl describe <type>/<name> output

    Examples:
        kubectl_describe("pod/my-pod")
        kubectl_describe("deployment/mydeploy")
        kubectl_describe("service/mysvc")

    Args:
        resource_name (str): The resource type combined with the name
    """
    namespace = state["namespace"]
    print(f"ran kube describe {resource_name}")
    if ";" in resource_name:
        return "error"
    return _kubectl_command(f"describe {resource_name}", namespace=namespace)

@tool
def kubectl_logs(pod_name: str, state: Annotated[dict, InjectedState]) -> str:
    """Get kubectl logs <pod_name> --tail=25 output (last 25 lines only)

    Args:
        pod_name (str): The name of the pod
    """
    namespace = state["namespace"]
    print(f"ran kube logs {pod_name}")
    if " " in pod_name:
        return "error"
    if ";" in pod_name:
        return "error"
    # TODO raise value error here
    # TODO detect return code and raise value error when kubectl errors
    return _kubectl_command(f"logs {pod_name} --tail=25", namespace=namespace)

