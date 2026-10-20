"""Task 2 (PPO) report table, rendered from results/task2_ppo only.

  report/tables/t2_clip.tex          cached-rollout clip / affected-token fractions (common field and per-epsilon)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "task2_ppo"
TAB = REPO / "report" / "tables"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def clip_table() -> Path:
    s = load(RES / "clipping_cache" / "summary.json")["summary"]
    lines = []
    for e in sorted(s["geometry"], key=float):
        g, d = s["geometry"][e], s["dynamics"][e]
        lines.append(f"    {float(e):.2f} & {g['clip_frac']:.4f} & {g['affected_frac']:.4f} & {d['clip_frac']:.4f} & "
                     f"{d['affected_frac']:.4f} \\\\")
    tex = ("\\begin{tabular}{c c c c c}\n  \\toprule\n"
           "  & \\multicolumn{2}{c}{\\textbf{Common field} ($\\epsilon = 0.20$)} & "
           "\\multicolumn{2}{c}{\\textbf{Own field}} \\\\\n"
           "  $\\epsilon$ & Clip & Affected & Clip & Affected \\\\\n  \\midrule\n"
           + "\n".join(lines) + "\n  \\bottomrule\n\\end{tabular}\n")
    out = TAB / "t2_clip.tex"
    out.write_text(tex)
    return out


if __name__ == "__main__":
    print(f"  [save] {clip_table().relative_to(REPO)}")
