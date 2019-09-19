#!/bin/bash
export COLLECTION_PY=crane.collect_individual_steps
export ENV_FILENAME=CraneML_0813.x86_64

echo "==> Host name: " $HOSTNAME
echo "==> collect python module: " $COLLECTION_PY 
echo "==> env file: " $ENV_FILENAME

python3 -m \
    $COLLECTION_PY \
    --docker-training=True \
    --env-filename=$ENV_FILENAME
