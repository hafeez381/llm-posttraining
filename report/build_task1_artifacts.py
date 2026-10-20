"""Task 1 (DPO) report tables and figure, rendered from results/task1_dpo only.

  report/tables/t1_summary.tex   held-out pairs + generations for SFT, standard, length-balanced, beta forks
  report/tables/t1_length.tex    per-stratum accuracy, per-token log-ratios, word-limit compliance
  report/figures/t1_length.pdf   margin vs length difference on the stratified pairs
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import RC
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "task1_dpo"
TAB, FIG = REPO / "report" / "tables", REPO / "report" / "figures"
RUNS = [("sft", "SFT (reference)"), ("standard", "Standard DPO"), ("length_balanced", "Length-balanced DPO"),
        ("beta_0.03", r"$\beta = 0.03$"), ("beta_0.1", r"$\beta = 0.10$"), ("beta_0.3", r"$\beta = 0.30$")]
BUDGET = {"sft": "--", "standard": "1500/94", "length_balanced": "1500/94",
          "beta_0.03": "600/38", "beta_0.1": "600/38", "beta_0.3": "600/38"}
STRATA = [("preferred_longer", "Pref.\\ longer"), ("length_matched", "Matched"), ("rejected_longer", "Rej.\\ longer")]
COLOR = {"standard": "#2a78d6", "length_balanced": "#eb6834"}
NAME = {"standard": "Standard", "length_balanced": "Length-balanced"}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def rows(p: Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def summary_table() -> Path:
    gens = {r: load(RES / r / "eval" / "generations_summary.json") for r, _ in RUNS}
    pairs = {r: load(RES / r / "eval" / "pairs_summary.json") for r, _ in RUNS}
    proto = {(json.dumps(g["config"]["generation"]), g["config"]["max_generation_tokens"], g["summary"]["n"])
             for g in gens.values()}
    assert len(proto) == 1, f"runs differ in decoding protocol: {proto}"
    lines = []
    for r, label in RUNS:
        p, g = pairs[r]["summary"], gens[r]["summary"]
        lines.append(
            f"    {label} & {BUDGET[r]} & {p['loss_mean']:.4f} & {p['accuracy']:.3f} & "
            f"{g['kl_seq']['mean']:.3f} & {g['rm_score']['mean']:.3f} & "
            f"{g['length']['mean']:.1f} ({g['length']['iqr']:.0f}) \\\\")
        if r in ("sft", "length_balanced"):
            lines.append("    \\midrule")
    tex = ("\\begin{tabular}{l c c c c c c}\n  \\toprule\n"
           "  \\textbf{Model} & \\textbf{Pairs/steps} & \\textbf{Loss} & \\textbf{Accuracy} & "
           "\\textbf{KL} & \\textbf{RM} & \\textbf{Length (IQR)} \\\\\n  \\midrule\n"
           + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "t1_summary.tex"
    out.write_text(tex)
    return out


def per_token(R: list) -> tuple:
    c = np.mean([(r["logp_c"] - r["ref_c"]) / r["chosen_tokens"] for r in R])
    j = np.mean([(r["logp_r"] - r["ref_r"]) / r["rejected_tokens"] for r in R])
    corr = np.corrcoef([r["margin"] for r in R], [r["length_difference"] for r in R])[0, 1]
    return c, j, corr


def length_table() -> Path:
    strat = load(RES / "length_analysis" / "stratified_summary.json")["summary"]
    wl = load(RES / "length_analysis" / "word_limit_summary.json")["summary"]
    data = load(RES / "length_analysis" / "dataset_summary.json")["summary"]
    R = rows(RES / "length_analysis" / "stratified_pairs.jsonl")
    cols = {}
    for m in ("standard", "length_balanced"):
        s = strat[m]["by_stratum"]
        assert sum(s[k]["n"] for k, _ in STRATA) == strat[m]["n"]
        c, j, corr = per_token([r for r in R if r["model"] == m])
        cols[m] = [f"{data[m]['frac_chosen_longer']:.3f}"] + [f"{s[k]['accuracy']:.3f}" for k, _ in STRATA] + \
                  [f"${c:.4f}$", f"${j:.4f}$", f"${corr:.2f}$", f"{wl[m]['compliance']:.2f}"]
    cols["sft"] = ["--", "0.000", "0.000", "0.000", "$0$", "$0$", "--", f"{wl['sft']['compliance']:.2f}"]
    labels = ["Training pairs, preferred longer", "Accuracy, preferred longer", "Accuracy, length-matched",
              "Accuracy, rejected longer", "Log-ratio per token, preferred", "Log-ratio per token, rejected",
              "$r(m_\\theta, \\Delta T)$", "Word-limit compliance"]
    rules = {1: "\\midrule", 4: "\\midrule", 7: "\\midrule"}
    lines = []
    for i, lab in enumerate(labels):
        if i in rules:
            lines.append("  " + rules[i])
        lines.append(f"    {lab} & {cols['standard'][i]} & {cols['length_balanced'][i]} & {cols['sft'][i]} \\\\")
    tex = ("\\begin{tabular}{l c c c}\n  \\toprule\n"
           "  \\textbf{Metric} & \\textbf{Standard} & \\textbf{Length-balanced} & \\textbf{SFT} \\\\\n  \\midrule\n"
           + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "t1_length.tex"
    out.write_text(tex)
    return out


def length_figure() -> Path:
    R = rows(RES / "length_analysis" / "stratified_pairs.jsonl")
    plt.rcParams.update(RC)
    fig, b = plt.subplots(figsize=(4.0, 2.4))
    for m in ("standard", "length_balanced"):
        Rm = [r for r in R if r["model"] == m]
        dl = np.array([r["length_difference"] for r in Rm])
        mg = np.array([r["margin"] for r in Rm])
        b.scatter(dl, mg, s=6, color=COLOR[m], alpha=0.5, edgecolors="none")
        k, c0 = np.polyfit(dl, mg, 1)
        g = np.linspace(dl.min(), dl.max(), 2)
        b.plot(g, k * g + c0, color=COLOR[m], lw=1.5, label=NAME[m])
    b.axhline(0, color="#999", lw=0.8)
    b.axvline(0, color="#999", lw=0.8)
    b.set_xlabel(r"Length difference $T(y^+) - T(y^-)$ (tokens)")
    b.set_ylabel(r"Margin $m_\theta$")
    b.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    out = FIG / "t1_length.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    return out


if __name__ == "__main__":
    for p in (summary_table(), length_table(), length_figure()):
        print(f"  [save] {p.relative_to(REPO)}")
