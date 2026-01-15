#!/usr/bin/env bash
set -euo pipefail

echo "== Baseline (base model) on easy split =="
python -m enough_thinking.eval.eval_jsonl --data enough-thinking/data/processed/math_eval_easy.jsonl --mode plain --n 50
python -m enough_thinking.eval.eval_jsonl --data enough-thinking/data/processed/math_eval_easy.jsonl --mode tagged_strict --n 50
python -m enough_thinking.eval.eval_jsonl --data enough-thinking/data/processed/math_eval_easy.jsonl --mode easy_short --n 50 --max_new_tokens 128

echo "== Phase1 LoRA on easy/hard split (tagged_strict) =="
python -m enough_thinking.eval.eval_jsonl --data enough-thinking/data/processed/math_eval_easy.jsonl --mode tagged_strict --n 50 --lora enough-thinking/outputs/phase1/lora_adapter
python -m enough_thinking.eval.eval_jsonl --data enough-thinking/data/processed/math_eval_hard.jsonl --mode tagged_strict --n 50 --lora enough-thinking/outputs/phase1/lora_adapter
