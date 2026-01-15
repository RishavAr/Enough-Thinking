import json
from pathlib import Path
import matplotlib.pyplot as plt

outdir = Path("enough-thinking/outputs/report")
outdir.mkdir(parents=True, exist_ok=True)

# Put your measured numbers here (edit if you want):
# Baseline (from your first eval): accuracy=0.22 avg_tokens=217.66
# Phase1 (tagged strict + LoRA): accuracy=0.30 avg_tokens=192.86
# Phase2 (easy_short baseline before SEAL): accuracy=0.10 avg_tokens=106.3  (prompt-only short)
rows = [
    {"stage": "Baseline", "accuracy": 0.22, "avg_tokens": 217.66},
    {"stage": "Phase-1 (GRPO LoRA)", "accuracy": 0.30, "avg_tokens": 192.86},
    {"stage": "Prompt short (pre-SEAL)", "accuracy": 0.10, "avg_tokens": 106.30},
]

# Save metrics table as markdown
md = ["| Stage | Accuracy | Avg tokens |", "|---|---:|---:|"]
for r in rows:
    md.append(f"| {r['stage']} | {r['accuracy']:.2f} | {r['avg_tokens']:.2f} |")
(outdir / "metrics_table.md").write_text("\n".join(md), encoding="utf-8")

# Plot accuracy vs tokens
tokens = [r["avg_tokens"] for r in rows]
acc = [r["accuracy"] for r in rows]
labels = [r["stage"] for r in rows]

plt.figure(figsize=(6,4))
plt.scatter(tokens, acc)
for t,a,l in zip(tokens, acc, labels):
    plt.annotate(l, (t,a))
plt.xlabel("Average generated tokens")
plt.ylabel("Accuracy")
plt.title("Reasoning efficiency tradeoff")
plt.grid(True)
plt.tight_layout()
plt.savefig(outdir / "reasoning_efficiency.png", dpi=200)

print("✅ wrote:")
print(" -", outdir / "metrics_table.md")
print(" -", outdir / "reasoning_efficiency.png")
