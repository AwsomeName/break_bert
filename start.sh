#!/usr/bin/env bash
set -euo pipefail
: "${PORT:=8000}"
if [ ! -f "/app/models/bert_model/model.onnx" ]; then
  echo "ONNX model missing: /app/models/bert_model/model.onnx" >&2
  exit 1
fi
python /app/src/server.py
