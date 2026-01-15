#!/usr/bin/env bash
set -euo pipefail

# Outer loop (non-safety version you already ran earlier and used to generate accepted edits)
python -m enough_thinking.phase2_seal.run_phase2   --phase1_lora enough-thinking/outputs/phase1/lora_adapter   --easy_eval enough-thinking/data/processed/math_eval_easy.jsonl   --out_dir enough-thinking/outputs/phase2   --outer_steps 5   --candidates 2   --keep_top 1

# Train edit-policy LoRA on accepted_edits.jsonl
python -m enough_thinking.phase2_seal.train_edit_policy   --accepted_edits enough-thinking/outputs/phase2/accepted_edits.jsonl   --out_dir enough-thinking/outputs/phase2/edit_policy_lora   --steps 80

echo "✅ phase2 done: outputs/phase2/accepted_edits.jsonl + edit_policy_lora"
