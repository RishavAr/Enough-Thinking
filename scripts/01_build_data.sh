#!/usr/bin/env bash
set -euo pipefail
python -m enough_thinking.data.build_gsm8k_jsonl --out_dir enough-thinking/data/processed --train_size 2000
echo "✅ data built in enough-thinking/data/processed"
