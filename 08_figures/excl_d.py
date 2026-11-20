"""The five reservoirs (DAM_ID) left out of every panel that involves the rule models (figure set version d).

Reason: at these reservoirs the published ISTARF parameters pin the simulated storage at full or empty
(in the paper's run, ISTARF sits at capacity or empty on at least 95% of days under the GRADES-hydroDL
inflow). A comparison with the rule models there would not test their rules.

How the list is applied:
  - prepare_kit.py removes these reservoirs' rule-model rows (GDROM, ISTARF, MODROM and the rule-model
    ensemble) from the score tables the figure scripts read.
  - The figure scripts remove them from the reference reservoir sets they check their own sets against.
  - SWOT-GAT's rows stay, so the panels that show SWOT-GAT alone (Figures 2a, 3a and 3c) keep all five.
"""
EXCL = [144, 1095, 1125, 1790, 1916]
