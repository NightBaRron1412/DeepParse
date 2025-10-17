#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/artifacts/data"
mkdir -p "$DATA_DIR"

echo "[fetch_datasets] This script attempts to download corrected LogHub datasets."
echo "If offline, manually place datasets under artifacts/data/<DATASET_NAME>/raw.log"

# Placeholder for actual download logic; avoids leaking reviewer identity.
DATASETS=(HDFS Hadoop Spark Zookeeper BGL OpenStack Thunderbird Windows Linux HealthApp Apache HadoopApp SparkApp Android Mac Proxifier)
for ds in "${DATASETS[@]}"; do
  TARGET="$DATA_DIR/$ds"
  mkdir -p "$TARGET"
  if [[ ! -f "$TARGET/raw.log" ]]; then
    echo "Dataset $ds missing. Please download from the public mirror and place raw.log in $TARGET."
  else
    echo "Dataset $ds already present."
  fi
done
