
import argparse, json, math, random
from pathlib import Path
from statistics import mean

import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from enough_thinking.rewards.r1_rewards import r1_reward

def load_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_prompt(question: str):
    return (
        "Return ONLY the following XML tags and nothing else.\n"
        "<think>...</think>\n"
        "<answer>...</answer>\n\n"
        "Rules:\n"
        "- Put your reasoning inside <think>.\n"
        "- Put ONLY the final integer inside <answer>.\n\n"
        f"Question: {question}\n"
    )

def logprob_of_sequence(model, input_ids, attention_mask):
    """
    Compute sum log-prob of each generated token (excluding prompt tokens).
    input_ids: [B, T] full prompt+gen
    """
    with torch.no_grad():
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", default="outputs/phase1")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--K", type=int, default=4)                 # group size
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--format_bonus", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(args.train_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"]
    )
    model = get_peft_model(base, lora_cfg)
    model.train()

    opt = AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def compute_logprobs(full_ids, prompt_len):
        """
        full_ids: [T]
        Returns sum logprob of generated tokens (prompt_len..T-1) under current model.
        """
        input_ids = full_ids.unsqueeze(0)
        attn = torch.ones_like(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits[0]  # [T, V]
        # logprob for token t is from logits at t-1 predicting token t
        logp = torch.log_softmax(logits, dim=-1)
        gen_logp = 0.0
        for t in range(prompt_len, full_ids.shape[0]):
            tok = int(full_ids[t].item())
            gen_logp += logp[t-1, tok]
        return gen_logp

    losses = []
    rewards_hist = []

    for step in range(args.steps):
        batch = [rows[random.randrange(len(rows))] for _ in range(args.batch_size)]

        step_loss = 0.0
        step_rewards = []

        for ex in batch:
            q = ex["prompt"]
            gold = int(ex["answer"])
            prompt = build_prompt(q)
            prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = prompt_ids["input_ids"].shape[-1]

            # Sample K completions (greedy-ish but with sampling for exploration)
            texts = []
            full_ids_list = []
            for _ in range(args.K):
                with torch.no_grad():
                    out = model.generate(
                        **prompt_ids,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        max_new_tokens=args.max_new_tokens,
                    )
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                texts.append(text)
                full_ids_list.append(out[0])

            # Compute rewards
            rewards = [r1_reward(t, gold, format_bonus=args.format_bonus) for t in texts]
            step_rewards.extend(rewards)

            # Group-relative advantage (normalize within group)
            r_mean = mean(rewards)
            r_std = (mean([(r - r_mean) ** 2 for r in rewards]) + 1e-8) ** 0.5
            adv = [(r - r_mean) / r_std for r in rewards]

            # GRPO-like loss: - E[ adv * logpi(y|x) ]
            # (No critic; advantage from group normalization)
            for a, full_ids in zip(adv, full_ids_list):
                logp = compute_logprobs(full_ids, prompt_len)
                step_loss = step_loss + (-a * logp)

        step_loss = step_loss / (args.batch_size * args.K)
        opt.zero_grad()
        step_loss.backward()
        opt.step()

        losses.append(float(step_loss.detach().cpu()))
        rewards_hist.append(mean(step_rewards))

        if (step + 1) % 20 == 0:
            print({"step": step+1, "loss": losses[-1], "avg_reward": rewards_hist[-1]})

    # Save adapter
    model.save_pretrained(out_dir / "lora_adapter")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    print("✅ Saved Phase1 adapter to:", out_dir / "lora_adapter")

if __name__ == "__main__":
    main()
