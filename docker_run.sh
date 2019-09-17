#!/bin/bash
echo "Host name: " $HOSTNAME

echo "=> Pull the newest update from git"
git pull

echo "=> Run crane/collect_to_bigtable..."
python3 -m \
    crane.collect_individual_steps \
    --docker-training=True \
    --env-filename=CraneML_0813.x86_64