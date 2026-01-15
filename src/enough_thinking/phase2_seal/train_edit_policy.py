import argparse, json, random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def load_jsonl(path: str):
    rows=[]
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_edit_prompt(exemplar_question: str):
    return f"""Create a SELF_EDIT for EASY grade-school math to reduce tokens WITHOUT losing correctness.

Return ONLY valid JSON with this schema:
{{
  "rule": "one sentence",
  "pairs": [
    {{"q": "question", "a": "integer"}},
    {{"q": "question", "a": "integer"}}
  ]
}}

Constraints:
- "a" must be an integer string
- keep "rule" short (<= 1 sentence)
- make pairs similar to the target question type

TARGET QUESTION:
{exemplar_question}
""".strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--accepted_edits", required=True)
    ap.add_argument("--out_dir", default="outputs/phase2/edit_policy_lora")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    edits = load_jsonl(args.accepted_edits)
    if not edits:
        raise RuntimeError("accepted_edits.jsonl is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.train()

    opt = AdamW(model.parameters(), lr=args.lr)

    # Build SFT examples: prompt -> JSON
    sft_texts = []
    for row in edits:
        edit = row["edit"]
        # use the first pair's question as exemplar (works even for fallback)
        exemplar_q = edit["pairs"][0][0]
        prompt = build_edit_prompt(exemplar_q)

        # turn edit into strict JSON target
        json_target = {
            "rule": edit["rule"],
            "pairs": [{"q": q, "a": str(a)} for (q, a) in edit["pairs"]]
        }
        full = prompt + "\n\n" + json.dumps(json_target, ensure_ascii=False)
        sft_texts.append(full)

    # Train
    for step in range(1, args.steps + 1):
        text = random.choice(sft_texts)
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=768).to(model.device)
        labels = toks["input_ids"].clone()

        out = model(**toks, labels=labels)
        loss = out.loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 == 0:
            print({"step": step, "loss": float(loss.detach().cpu())})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("✅ Saved edit policy LoRA to:", out_dir)

if __name__ == "__main__":
    main()
