"""Task 4 (safety) report tables and figure, rendered from results/task4_safety only.

  report/tables/t4_rates.tex        AI-judge calibration rates per policy, with manual-audit rates beside them
  report/tables/t4_confusion.tex    judge (rows) vs manual (columns) on the 240 audited responses
  report/figures/t4_types.pdf       judge share of safe prompts not answered safely, by XSTest type and policy
  report/figures/t4_types_unsafe.pdf  judge share of unsafe contrasts not labelled justified refusals, by type and policy
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import RC
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "task4_safety" / "evaluation"
TAB, FIG = REPO / "report" / "tables", REPO / "report" / "figures"
POLICIES = [("sft", "SFT"), ("dpo", "DPO"), ("ppo", "PPO"), ("grpo", "GRPO")]
LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
SHORT = {"SAFE_ANSWER": "Safe ans.", "JUSTIFIED_REFUSAL": "Just.\\ ref.", "UNSAFE_COMPLIANCE": "Unsafe comp.",
         "OVER_REFUSAL": "Over-ref.", "AMBIGUOUS": "Ambig."}
TYPE_NAME = {"contrast_discr": "discrimination", "nons_group_real_discr": "nonsense group, real discr.", "real_group_nons_discr": "real group, nonsense discr."}
COLOR = {"sft": "#888888", "dpo": "#2a78d6", "ppo": "#eb6834", "grpo": "#3a9e5f"}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def rows(p: Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def rates_table() -> Path:
    s = load(RES / "summary.json")["summary"]
    lines = []
    for p, label in POLICIES:
        a = s["policies"][p]
        lines.append(
            f"    {label} & {a['safe_answer']['rate']:.3f} & {a['over_refusal']['rate']:.3f} & "
            f"{a['unsafe_compliance']['rate']:.3f} & {a['justified_refusal']['rate']:.3f} & {a['ambiguous']['rate']:.3f} & "
            f"{a['length_mean']:.1f} \\\\")
    tex = ("\\begin{tabular}{l c c c c c c}\n  \\toprule\n"
           "  \\textbf{Policy} & \\textbf{Safe answer} & \\textbf{Over-refusal} & \\textbf{Unsafe compliance} & "
           "\\textbf{Justified refusal} & \\textbf{Ambiguous} & \\textbf{Length} \\\\\n  \\midrule\n"
           + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "t4_rates.tex"
    out.write_text(tex)
    return out


def confusion_table() -> Path:
    a = load(RES / "summary.json")["summary"]["audit"]
    c = a["confusion_ai_rows_manual_cols"]
    lines = [f"    {SHORT[r]} & " + " & ".join(str(c[r][m]) for m in LABELS) + " \\\\" for r in LABELS]
    tex = ("\\begin{tabular}{l c c c c c}\n  \\toprule\n"
           "  \\textbf{AI} $\\downarrow$ / \\textbf{Manual} $\\rightarrow$ & "
           + " & ".join(SHORT[m] for m in LABELS) + " \\\\\n  \\midrule\n" + "\n".join(lines)
           + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "t4_confusion.tex"
    out.write_text(tex)
    return out


def types_figure(cls: str, good: str, ylabel: str, name: str, height: float = 1.85) -> Path:
    R = [r for r in rows(RES / "rows.jsonl") if r["benchmark_class"] == cls]
    types = list(dict.fromkeys(r["type"] for r in R))
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(5.5, height))
    width = 0.2
    x = np.arange(len(types))
    for i, (p, label) in enumerate(POLICIES):
        y = [np.mean([r["ai_label"] != good for r in R if r["policy"] == p and r["type"] == t]) for t in types]
        ax.bar(x + (i - 1.5) * width, y, width, color=COLOR[p], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_NAME.get(t, t.removeprefix("contrast_").replace("_", " ")) for t in types],
                       rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    out = FIG / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    return out

if __name__ == "__main__":
    for p in (rates_table(), confusion_table(),
              types_figure("SAFE", "SAFE_ANSWER", "Not a safe answer", "t4_types.pdf"),
              types_figure("UNSAFE", "JUSTIFIED_REFUSAL", "Not a justified refusal", "t4_types_unsafe.pdf", 1.45)):
        print(f"  [save] {p.relative_to(REPO)}")
