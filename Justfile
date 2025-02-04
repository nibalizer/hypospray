default:
    just --list

test:
    pytest
    ruff --check .
    mypy .

integration:
    for ns in $(cd testing/envs; echo *); do echo hypospray-$ns; python -m hypospray.main hypospray-$ns; done
