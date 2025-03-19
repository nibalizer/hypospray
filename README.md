hypospray
---------

Find and resolve kubernetes issues, automatically.


> A hypospray (colloquially, hypo) was a medical device used to inject liquids into the body. The system used a noninvasive transport mechanism of compressed air to transfer the injectant from the device into the subdermal layer below the skin of the body, or artery.[0]

About
-----

Hypospray is an Agentic kubernetes debugging tool. It gathers information about the kubernetes namespace it is told to inspect and passes that information to the LLM. The LLM has been given kubernetes debugging [tools](https://medium.com/@yashpaddalwar/agents-and-tool-calling-in-agentic-frameworks-the-ultimate-guide-0ec446e89b55) it can use to gather more information and dig into problems. This is a level-up from traditional RAG where the application needs to decide which resources to inspect. Hypospray will always support local/self-hosted LLMs for security.

Example Usage
-------------

```
$ NAMESPACE=hypospray-05-node-selectors
$ hypospray --show-tools -e $NAMESPACE
Welcome to hypospray
ping ollama
ran kube -n hypospray-05-node-selectors get pods
ran kube describe pod/my-deployment-659d54dbd6-q5js2
ran kube get events
ran kube logs my-deployment-659d54dbd6-q5js2
{
    "green": false,
    "erroring_resources": [
        "pod/my-deployment-659d54dbd6-q5js2",
        "deployment.apps/my-deployment"
    ],
    "num_messages": 7,
    "explanation": "Based on the output, it appears that there are several issues with the Kubernetes cluster:\n\n1. The pod `my-deployment-659d54dbd6-q5js2` is in a pending state and has been for 43 days. This suggests that the pod has not been scheduled to run on any node.\n2. The deployment `my-deployment` also shows a status of 0/1, indicating that only one replica has been created but it's not running.\n3. The replicaset `my-deployment-659d54dbd6` shows a status of 0/1, which is consistent with the pod being in a pending state.\n4. The events show a warning message indicating that no nodes are available to schedule the pod due to node affinity/selector mismatch.\n\nThe issue seems to be related to the node affinity/selector configuration of the pod. It appears that the pod's node selector does not match any available nodes, resulting in the scheduling failure.\n\nTo resolve this issue, you can try the following:\n\n1. Check the node affinity/selector configuration of the pod and ensure it matches one or more available nodes.\n2. Verify that the node labels match the node selector specified in the pod configuration.\n3. If necessary, update the node selector to match a different set of nodes.\n\nAdditionally, you may want to investigate why no nodes are available to schedule the pod. This could be due to various reasons such as:\n\n1. Node availability: Check if any nodes are down or not available for scheduling.\n2. Resource constraints: Verify that there are sufficient resources (CPU and memory) available on the nodes to run the pod.\n\nTo get a list of Kubernetes resources in an error state, you can use the following command:\n```\nkubectl get all --field-selector status.phase=Failed\n```\nThis will return a list of pods, deployments, replicaset, etc. that are in a failed state."
}
```

Quickstart
----------

```
git clone git@github.com:nibalizer/hypospray
cd hypospray
poetry install
```

```
# Verify kubectl is working
kubectl get namespaces
```

```
hypospray --help
hypospray -e $NAMESPACE
```

Contributing
------------

Contributions welcome!

1. Fork the repository on GitHub
2. Make changes
3. Run the unit tests (you can skip functests if that's too hard to get working)
3. Push changes and open a pull request

[0] https://memory-alpha.fandom.com/wiki/Hypospray
