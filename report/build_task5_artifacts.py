"""Task 5 (RLVR vs RLAIF) report tables, rendered from results/task5_feedback only.

  report/tables/t5_summary.tex       GSM8K / SVAMP accuracy, format, length, win rate vs SFT, SVAMP failure types
  report/tables/t5_diagnostics.tex   controlled diagnostics: better / tie / wrong per category, verifier vs judge
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "task5_feedback"
TAB = REPO / "report" / "tables"
POLICIES = [("sft", "SFT"), ("rlvr", "RLVR"), ("rlaif", "RLAIF")]
CATEGORIES = [("reasoning", "Corrupt reasoning, correct final"), ("outcome", "Good reasoning, wrong final"),
              ("outcome_distractor", "Gold distractor, wrong final"), ("filler", "Persuasive filler"),
              ("identity_control", "Clean vs clean (control)")]


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def summary_table() -> Path:
    s = load(RES / "comparison" / "summary.json")["summary"]
    lines = []
    for p, label in POLICIES:
        v = s[p]
        f = v["svamp_failure_types"]
        gw = f"{v['gsm_win_rate_vs_sft']:.3f}" if "gsm_win_rate_vs_sft" in v else "--"
        sw = f"{v['transfer_win_rate_vs_sft']:.3f}" if "transfer_win_rate_vs_sft" in v else "--"
        lines.append(f"    {label} & {v['gsm_accuracy']:.3f} & {v['gsm_format']:.2f} & {v['gsm_length_mean']:.1f} & {gw} & "
                     f"{v['svamp_accuracy']:.3f} & {v['svamp_format']:.2f} & {v['svamp_length_mean']:.1f} & {sw} & "
                     f"{f.get('wrong_final', 0)} / {f.get('no_designated_final', 0)} / {f.get('hash_unparsed', 0)} / "
                     f"{f.get('truncated', 0)} \\\\")
    tex = ("\\begin{tabular}{l cccc ccccc}\n  \\toprule\n"
           "  & \\multicolumn{4}{c}{\\textbf{GSM8K}} & \\multicolumn{5}{c}{\\textbf{SVAMP}} \\\\\n"
           "  \\cmidrule(lr){2-5}\\cmidrule(lr){6-10}\n"
           "  \\textbf{Policy} & Acc. & Format & Length & Win & Acc. & Format & Length & Win & Failures \\\\\n  \\midrule\n"
           + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "t5_summary.tex"
    out.write_text(tex)
    return out


def diagnostics_table() -> Path:
    s = load(RES / "diagnostics" / "summary.json")["summary"]
    cell = lambda m, c: f"{s[m][c]['better']:.2f} / {s[m][c]['tie']:.2f} / {s[m][c]['wrong']:.2f}"
    lines = [f"    {label} & {cell('verifier', c)} & {cell('judge', c)} \\\\" for c, label in CATEGORIES]
    tex = ("\\begin{tabular}{l c c}\n  \\toprule\n"
           "  \\textbf{Clean vs \\ldots} & \\textbf{Verifier} better / tie / wrong & "
           "\\textbf{AI judge} better / tie / wrong \\\\\n  \\midrule\n" + "\n".join(lines)
           + f"\n  \\midrule\n  $S_{{\\mathrm{{reason}}}}$ & {s['verifier']['S_reason']:.3f} & {s['judge']['S_reason']:.3f} \\\\\n"
             f"  $S_{{\\mathrm{{outcome}}}}$ & {s['verifier']['S_outcome']:.3f} & {s['judge']['S_outcome']:.3f} \\\\\n"
             "  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "t5_diagnostics.tex"
    out.write_text(tex)
    return out


if __name__ == "__main__":
    for p in (summary_table(), diagnostics_table()):
        print(f"  [save] {p.relative_to(REPO)}")
