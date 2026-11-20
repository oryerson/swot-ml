"""Combine the four figure PDFs into one four-page PDF, one figure per page (figure set version d).

Usage: [FINAL=1] figs_only_c5d.py DEST
  DEST  the folder that holds the four figure PDFs (the kit folder the figure scripts wrote to)

Reads: DEST/c5d_fig1.pdf, c5d_fig2.pdf, c5d_fig3.pdf and c5d_fig4.pdf (each must exist).

Writes: DEST/SWOT-ResOps_figures_c5_o50_20261113d.tex and the .pdf that pdflatex makes from it.

Method: the script writes a LaTeX file that places each figure on its own page, at most 7.48 in wide and
9.0 in high, on 0.5 in margins, with the label "Figure N." below it. There is no title and no caption.
It then runs pdflatex twice in DEST. pdflatex must be on the PATH.

Notes:
  - When the script runs from a folder not named kit_d, the PDF name gets the last "_"-separated part of
    that folder's name as a suffix (from kit_d_ex7: ..._20261113d_ex7).
  - FINAL=1 also copies the PDF to CONUS_storage_fork/eval/grl_figs/figure_sets/ under $SWOTGAT_WORK.
    The script stops instead of overwriting a file that is already there.
  - cmaps_b is imported but not used.
"""
from swotgat.paths import WORK
import os, sys, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import cmaps_b as CB
from pathlib import Path
D = Path(sys.argv[1]); OUTP = Path(f"{WORK}/CONUS_storage_fork/eval/grl_figs/figure_sets")
name = "SWOT-ResOps_figures_c5_o50_20261113d" + ("" if Path(__file__).resolve().parent.name == "kit_d" else "_" + Path(__file__).resolve().parent.name.split("_")[-1])
figs = [("c5d_fig1.pdf", "1"), ("c5d_fig2.pdf", "2"), ("c5d_fig3.pdf", "3"), ("c5d_fig4.pdf", "4")]
tex = (r"\documentclass[10pt]{article}\usepackage[margin=0.5in]{geometry}\usepackage{graphicx}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
       r"\pagestyle{empty}\begin{document}" + "\n")
for f, i in figs:
    assert (D/f).exists(), f
    tex += f"\\begin{{center}}\\includegraphics[width=7.48in,height=9.0in,keepaspectratio]{{{D/f}}}\\end{{center}}\\vspace{{2pt}}\\parbox{{7.48in}}{{\\small \\textbf{{Figure {i}.}}}}\\clearpage\n"
tex += r"\end{document}" + "\n"; (D/f"{name}.tex").write_text(tex)
for _ in range(2): r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"], cwd=D, capture_output=True, text=True)
if r.returncode: print(r.stdout[-1200:]); raise SystemExit("pdflatex failed")
print("wrote", D/f"{name}.pdf")
if os.environ.get("FINAL") == "1":
    assert not (OUTP/f"{name}.pdf").exists(), f"{name}.pdf exists; not overwriting"; shutil.copy2(D/f"{name}.pdf", OUTP/f"{name}.pdf"); print("published", OUTP/f"{name}.pdf")
