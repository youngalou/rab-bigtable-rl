#!/bin/bash
git pull
python3 -m \
    crane.collect_to_bigtable \
    --docker-training=True \
    --env-filename=CraneML_0813.x86_64