#!/usr/bin/env bash
set -euo pipefail
python -m enough_thinking.phase1_grpo.run_phase1   --train_jsonl enough-thinking/data/processed/math_train.jsonl   --out_dir enough-thinking/outputs/phase1   --steps 120   --K 4   --batch_size 1   --max_new_tokens 160
echo "✅ phase1 done: enough-thinking/outputs/phase1/lora_adapter"
