#!/usr/bin/env bash
set -e
CKPT=$1
if [ -z "$CKPT" ]; then echo "Usage: scripts/eval.sh /path/to/checkpoint.pth"; exit 1; fi
python -m src.evaluate --config configs/defaults.yaml --ckpt "$CKPT"
