# Enough Thinking (Colab-scale)

Baseline + training pipeline for:
- Phase 1: R1-style structured reasoning (<think>/<answer>)
- Phase 2: token-efficiency (SEAL-style outer loop later)
- Phase 3: MCP agent (later)

SETUP
pip install -r requirements.txt
export HF_TOKEN="hf.."  

BASELINE EVAL ON GSM8K
python -m enough_thinking.eval.eval_gsm8k_baseline --mode plain --n 50
python -m enough_thinking.eval.eval_gsm8k_baseline --mode tagged_strict --n 50
python -m enough_thinking.eval.eval_gsm8k_baseline --mode easy_short --n 50 --max_new_tokens 128
