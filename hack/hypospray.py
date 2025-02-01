import subprocess
import requests
import sys
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
command = "get pods"
output = kubectl_command(command, namespace=namespace)
print(output)

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b-instruct-fp16",
    base_url="http://10.80.50.15:11434",
    temperature=0,
    # other params...
)

from langchain_core.messages import AIMessage

messages = [
    (
        "system",
        "You are a helpful devops infrastructure assistant. Given this kubectl output, can you see anything wrong or that need an admin to address? Be helpful but concise."
    ),
    ("human", f"{output}"),
]
ai_msg = llm.invoke(messages)


print(ai_msg)
