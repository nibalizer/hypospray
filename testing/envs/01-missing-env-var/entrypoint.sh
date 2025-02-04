#!/usr/bin/bash

# put this in /docker-entrypoint.d/ on an nginx image

if [ -z $MY_EXPECTED_ENV_VAR ]; then
    echo "error: environment variable MY_EXPECTED_ENV_VAR is missing"
    exit 1
fi
