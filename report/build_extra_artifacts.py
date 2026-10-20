"""Report evidence that no task builder covers, rendered from saved results only.

  report/figures/x_ppo_traj.pdf     standard PPO continuation, all eight per-update trajectories
  report/figures/x_grpo_traj.pdf    standard GRPO continuation, per-update trajectories with dead updates marked
  report/tables/x_rl_summary.tex    held-out metrics of the PPO and GRPO midpoints, standard runs and forks
  report/tables/x_compute.tex       peak GPU memory and wall-clock time of the standard PPO and GRPO runs
  report/tables/x_agreement.tex     verifier-judge agreement on the GSM8K comparisons against SFT
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import RC
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
TAB, FIG = REPO / "report" / "tables", REPO / "report" / "figures"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#999999"
RL_RUNS = {
    "task2_ppo": [("midpoint", "Midpoint", "--"), ("standard", "Standard", "20 upd."),
                  ("eps_0.05", r"$\epsilon = 0.05$", "8 upd."),
                  ("eps_0.2_kl_0.1", r"$\epsilon = 0.20$, $\beta_{\text{KL}} = 0.10$", "8 upd."),
                  ("eps_0.5", r"$\epsilon = 0.50$", "8 upd."), ("kl_0", r"$\beta_{\text{KL}} = 0$", "8 upd."),
                  ("kl_0.2", r"$\beta_{\text{KL}} = 0.20$", "8 upd.")],
    "task3_grpo": [("midpoint", "Midpoint", "--"), ("standard", "Standard", "20 upd."),
                   ("fork_grpo", "Canonical ($1/T_k$)", "8 upd."), ("fork_dr_grpo", "Dr.\\ ($1/T_{\\max}$)", "8 upd.")],
}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def rows(p: Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def style():
    plt.rcParams.update(RC)


def panels(R, specs, shape, size, dead=()):
    style()
    fig, axes = plt.subplots(*shape, figsize=size, sharex=True)
    u = np.array([r["update"] for r in R])
    for i, (ax, (key, ylabel)) in enumerate(zip(axes.flat, specs)):
        y = np.array([r[key] for r in R], dtype=float)
        ax.plot(u, y, "-o", ms=2.5, lw=0.8, color=BLUE)
        for d in dead:
            ax.plot(d, y[d - 1], "o", ms=4, mfc="none", mec=ORANGE, mew=1)
        ax.set_title(f"({chr(97 + i)})")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 10, 20])
    for ax in axes[-1]:
        ax.set_xlabel("Update")
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6)
    return fig


def ppo_figure() -> Path:
    R = rows(RES / "task2_ppo" / "standard" / "train" / "updates.jsonl")
    specs = [("reward_eff", "Reward"), ("kl_seq", "KL to ref."), ("policy_loss", "Policy loss"),
             ("value_loss", "Value loss"), ("entropy_exact", "Entropy"), ("clip_frac", "Clip fraction"),
             ("grad_norm_policy", "Grad. norm"), ("resp_len", "Length")]
    fig = panels(R, specs, (2, 4), (5.5, 2.15))
    fig.axes[6].axhline(1.0, color=GREY, lw=0.8, ls="--")
    fig.axes[5].set_ylim(-0.002, 0.05)
    out = FIG / "x_ppo_traj.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    return out


def grpo_dead(R) -> list:
    return [r["update"] for r in R if not r["informative"] or r["n_masked"] == len(r["len_per_completion"])]


def grpo_figure() -> Path:
    R = rows(RES / "task3_grpo" / "standard" / "train" / "updates.jsonl")
    seen = 0
    for i, r in enumerate(R, 1):
        seen += not r["informative"]
        r["uninformative_so_far"] = seen / i
    specs = [("reward_mean", "Mean reward"), ("reward_std", "Group std"), ("uninformative_so_far", "Uninform. frac."),
             ("kl_seq", "KL to ref."), ("policy_term", "Policy loss"), ("grad_norm", "Grad. norm"),
             ("entropy_exact", "Entropy"), ("len_mean", "Mean length")]
    fig = panels(R, specs, (2, 4), (5.5, 2.15), dead=grpo_dead(R))
    fig.axes[5].axhline(1.0, color=GREY, lw=0.8, ls="--")
    out = FIG / "x_grpo_traj.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    return out


def rl_summary_table() -> Path:
    blocks = []
    for task, name in (("task2_ppo", "PPO"), ("task3_grpo", "GRPO")):
        lines = [f"    \\multicolumn{{7}}{{l}}{{\\textit{{{name}}}}} \\\\"]
        for r, label, budget in RL_RUNS[task]:
            e = load(RES / task / r / "eval" / "summary.json")["summary"]
            train = RES / task / r / "train" / "summary.json"
            t = load(train)["summary"] if task == "task2_ppo" and train.exists() else None
            akl = f"{t['approx_kl_post_max']:.1e}" if t else "--"
            lines.append(f"    {label} & {budget} & {e['rm_score']['mean']:.3f} & {e['kl_seq_mean']:.3f} & "
                         f"{e['length']['mean']:.1f} ({e['length']['iqr']:.0f}) & {e['entropy_exact_tokmean']:.3f} & "
                         f"{akl} \\\\")
        blocks.append("\n".join(lines))
    tex = ("\\begin{tabular}{l c c c c c c}\n  \\toprule\n"
           "  \\textbf{Policy} & \\textbf{Budget} & \\textbf{RM} & \\textbf{KL} & \\textbf{Length (IQR)} & "
           "\\textbf{Entropy} & \\textbf{Max post-update KL} \\\\\n"
           "  \\midrule\n" + "\n    \\midrule\n".join(blocks) + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "x_rl_summary.tex"
    out.write_text(tex)
    return out


def compute_table() -> Path:
    ppo = load(RES / "task2_ppo" / "standard" / "train" / "summary.json")
    grpo = load(RES / "task3_grpo" / "standard" / "train" / "summary.json")
    tex = ("\\begin{tabular}{l c c}\n  \\toprule\n  & \\textbf{PPO} & \\textbf{GRPO} \\\\\n  \\midrule\n"
           f"  Peak GPU memory (GB) & {ppo['peak_vram_gb']:.2f} & {grpo['peak_vram_gb']:.2f} \\\\\n"
           f"  Wall-clock time (s) & {ppo['wall_clock_s']:.0f} & {grpo['wall_clock_s']:.0f} \\\\\n"
           "  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "x_compute.tex"
    out.write_text(tex)
    return out


def agreement_table() -> Path:
    pw = load(RES / "task5_feedback" / "pairwise" / "summary.json")["summary"]
    lines = []
    for k, name in (("gsm|rlvr", "RLVR"), ("gsm|rlaif", "RLAIF")):
        c = pw[k]["confusion_verifier_rows_judge_cols"]
        decisive = sum(c[v][j] for v in ("policy", "sft") for j in c[v])
        agreed = c["policy"]["policy"] + c["sft"]["sft"]
        ties = sum(c["tie"].values())
        judged = c["tie"]["policy"] + c["tie"]["sft"]
        assert abs(agreed / decisive - pw[k]["agreement_when_verifier_decisive"]) < 1e-9
        assert abs(judged / ties - pw[k]["judge_decisive_rate_on_verifier_ties"]) < 1e-9
        lines.append(f"    {name} & {agreed} of {decisive} ({agreed / decisive:.3f}) & {judged} of {ties} ({judged / ties:.3f}) \\\\")
    tex = ("\\begin{tabular}{l c c}\n  \\toprule\n"
           "  \\textbf{Policy} & \\textbf{Judge agrees, verifier decisive} & \\textbf{Judge decisive, verifier ties} \\\\\n"
           "  \\midrule\n" + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "x_agreement.tex"
    out.write_text(tex)
    return out


if __name__ == "__main__":
    for p in (ppo_figure(), grpo_figure(), rl_summary_table(), compute_table(), agreement_table()):
        print(f"  [save] {p.relative_to(REPO)}")
