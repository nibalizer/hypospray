#!/bin/bash
set -eu
set -o pipefail

e=$(basename `pwd`)
echo "Building and deploying test scenario $e"
IMAGE="${TESTING_BASE_IMAGE_URL}/hypospray-images/${e}"
NAMESPACE=hypospray-$e

docker build -t $IMAGE .
docker push $IMAGE

rm -f kustomization.yaml out.yaml
kustomize create --resources manifest.yaml
kustomize edit set image replaceme=${IMAGE}
cat kustomization.yaml
kustomize build -o out.yaml
kubectl -n $NAMESPACE apply -f out.yaml --validate=false
