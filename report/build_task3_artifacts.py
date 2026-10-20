"""Task 3 (GRPO) report tables and figure, rendered from results/task3_grpo only.

  report/tables/t3_group_size.tex     equal-generation group-size study, overall and by difficulty tercile
  report/figures/t3_length_grad.pdf   per-completion gradient norm vs length, canonical vs Dr. GRPO
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import RC
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "task3_grpo"
TAB, FIG = REPO / "report" / "tables", REPO / "report" / "figures"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def rows(p: Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def group_size_table() -> Path:
    s = load(RES / "group_size" / "summary.json")["summary"]
    lines = []
    for k in ("K2", "K4", "K8"):
        v = s[k]
        cells = [f"{v['informative_rate']:.3f}", f"{v['mean_within_group_std']:.3f}", f"{v['centered_reward_var']:.3f}"]
        cells += [f"{v['by_bin'][b]['informative_rate']:.3f} / {v['by_bin'][b]['mean_within_group_std']:.3f}"
                  for b in ("hard", "medium", "easy")]
        lines.append(f"    {k[1:]} & {v['n_groups']} & " + " & ".join(cells) + " \\\\")
    tex = ("\\begin{tabular}{c c c c c c c c}\n  \\toprule\n"
           "  & & & & & \\multicolumn{3}{c}{\\textbf{Informative / within-group std by tercile}} \\\\\n"
           "  $K$ & Groups & Informative & Within std & Centered var. & Hard & Medium & Easy \\\\\n  \\midrule\n"
           + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "t3_group_size.tex"
    out.write_text(tex)
    return out


def length_grad_figure() -> Path:
    R = [r for r in rows(RES / "length_gradients" / "length_gradients.jsonl") if r["adv"] != 0]
    t = np.array([r["T"] for r in R], dtype=float)
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    for key, label, color in (("gnorm_grpo", "Canonical", "#2a78d6"), ("gnorm_dr", "Dr.", "#eb6834")):
        y = np.array([r[key] for r in R])
        slope, icpt = np.polyfit(np.log(t), np.log(y), 1)
        ax.loglog(t, y, "o", ms=2.5, alpha=0.5, color=color, label=f"{label} (slope ${slope:.2f}$)")
        xs = np.geomspace(t.min(), t.max(), 50)
        ax.loglog(xs, np.exp(icpt) * xs ** slope, color=color, lw=1)
    ax.set_xlabel("Completion length $T_k$ (tokens)")
    ax.set_ylabel("Per-completion gradient norm")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    out = FIG / "t3_length_grad.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    return out


if __name__ == "__main__":
    for p in (group_size_table(), length_grad_figure()):
        print(f"  [save] {p.relative_to(REPO)}")
