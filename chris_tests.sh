#!/usr/bin/env bash

runpy() {
    echo "=== $1"
    PYTHONPATH=$PWD python $1
    echo
}

main() {
    # Delete any stale in-tree pyc files.
    find -name "*.pyc" -not -path "./env/*" -delete

    # Source the virtualenv.
    test -f env/bin/activate && source env/bin/activate

    # Run the "tests".
    runpy programl/proto/protos_test.py
    runpy deeplearning/ml4pl/poj104/dataset_test.py
}
main $@