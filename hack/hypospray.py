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



url = "http://10.80.50.15:11434/api/generate" # gpu
headers = {
    "Content-Type": "application/json"
}
payload = {
    "model": "llama3.1:8b-instruct-fp16",
    "stream": False,
    "prompt": f"Given this kubectl output, can you see anything wrong or that need an admin to address? Be helpful but concise.\n{output}"
}

#print(payload)

response = requests.post(url, json=payload, headers=headers)
#print(response.text)
print()

print(response.json()["response"])
