#!/bin/bash
set -eu
set -o pipefail

echo "Deleting and re-creating all broken configuration namespaces"
if [ ! -f ../.env ]; then
    echo "ERROR! No .env file found"
    exit 1
fi
source ../.env

envs=$(cd envs/; echo *)

for e in $envs
do
    echo $e
    pushd envs/$e
    NAMESPACE=hypospray-$e
    kubectl delete ns $NAMESPACE --ignore-not-found
    kubectl create ns $NAMESPACE
    ./deploy.sh
    popd
done

echo "Initial Deployment completed, status:"

for e in $envs
do
    NAMESPACE=hypospray-$e
    echo "inspecting namespace $NAMESPACE"
    kubectl -n $NAMESPACE get all
done




