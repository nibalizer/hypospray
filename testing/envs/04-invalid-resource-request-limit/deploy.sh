#!/bin/bash
set -eu
set -o pipefail

e=$(basename `pwd`)
echo "Building and deploying test scenario $e"
IMAGE="${TESTING_BASE_IMAGE_URL}/hypospray-images/${e}"
NAMESPACE=hypospray-$e

kubectl -n $NAMESPACE apply -f manifest.yaml --validate=false
