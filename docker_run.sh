#!/bin/bash
echo "==> Pull the newest update from git"
git pull

echo "==> run collection"
sh collect_run.sh