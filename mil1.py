import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_benchmark_txt(path):
    text = Path(path).read_text(encoding="utf-8")

    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s*$",
        re.MULTILINE,
    )

    rows = []
    for m in pattern.finditer(text):
        rows.append({
            "Conc": int(m.group(1)),
            "TTFT_p50": float(m.group(2)),
            "TTFT_p99": float(m.group(3)),
            "Compl_p50": float(m.group(4)),
            "Compl_p99": float(m.group(5)),
            "TPOT_p50": float(m.group(6)),
            "TPOT_p99": float(m.group(7)),
            "GenTok/s": float(m.group(8)),
            "OK": int(m.group(9)),
        })

    if not rows:
        raise ValueError(f"Could not parse benchmark table from {path}")

    rows.sort(key=lambda x: x["Conc"])
    return rows


def vals(rows, key):
    return [r[key] for r in rows]


file1 = "b3.txt"
file2 = "b4.txt"

d1 = parse_benchmark_txt(file1)
d2 = parse_benchmark_txt(file2)

x1 = vals(d1, "Conc")
x2 = vals(d2, "Conc")

selected = [
    ("TTFT_p50", "TTFT p50 (ms)"),
    ("Compl_p50", "Completion p50 (ms)"),
    ("TPOT_p50", "TPOT p50 (ms)"),
    ("GenTok/s", "Generation Throughput (tok/s)"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (metric, title) in zip(axes, selected):
    ax.plot(x1, vals(d1, metric), marker="o", label="32 Requests")
    ax.plot(x2, vals(d2, metric), marker="s", label="200 Requests")
    ax.set_title(title)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(title)
    ax.set_xticks(sorted(set(x1) | set(x2)))
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("comparison.pdf", dpi=300, bbox_inches="tight")
plt.close()