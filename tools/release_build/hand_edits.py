r"""Every hand edit made to a copied file, as exact text replacements.

``build_release.py edits`` applies them after ``copy``; ``provenance`` puts the
reasons in the note column of ``provenance/files.tsv``. Each entry is
``(release_path, old_text, new_text, reason)``; ``old_text`` must occur exactly
once in the file as it stands after the automatic path rewrite. An ``old_text``
that starts with ``re:`` is a regular expression (``new_text`` may use ``\\1``).
"""

_SESSION = r're:(\w+) = requests\.Session\(\); \1\.headers\["User-Agent"\] = "[^"\n]*"'
_POLITE = r"from swotgat.http import PoliteSession; \1 = PoliteSession()"
_WHY_SESSION = "generic client name instead of a personal contact string, and at most one request per 1.1 s per host (swotgat.http)"
_WHY_GET = "at most one request per 1.1 s per host and a generic client name (swotgat.http)"
_AG = "07_evaluation/a_gauge_truth"
_RM = "07_evaluation/b_rule_models"

EDITS = [
    ("07_evaluation/c_scoring/full_record/pinned_check.py",
     'HM = pd.read_csv(B/"kit/sites_2010_g.csv")',
     'HM = pd.read_csv(F/"eval/overlap50/score/sites_2010_g.csv")',
     "reads the High/Medium reservoir ids from the scorer table instead of a superseded figure kit (same ids)"),
    ("07_evaluation/c_scoring/full_record/picks.py",
     'F/"eval/grl_figs/intermediate/a365_vDspan_20261030/figset_a365e/eco_l1_sites.csv"',
     'B/"kit_d/eco_l1_sites.csv"',
     "reads the ecoregion table shipped in 08_figures/inputs (byte-identical file) instead of a test-arm folder"),
    (f"{_AG}/agency_2023_2025/build_catalogs.py", _SESSION, _POLITE, _WHY_SESSION),
    (f"{_AG}/agency_2023_2025/cat_codwr.py", _SESSION, _POLITE, _WHY_SESSION),
    (f"{_AG}/agency_2023_2025/harvest.py", _SESSION, _POLITE, _WHY_SESSION),
    (f"{_AG}/agency_2023_2025/harvest_v2.py", _SESSION, _POLITE, _WHY_SESSION),
    (f"{_AG}/agency_2023_2025/nwis_v2.py", _SESSION, _POLITE, _WHY_SESSION),
    (f"{_AG}/agency_2026/harvest_2026.py", _SESSION, _POLITE, _WHY_SESSION),
    (f"{_AG}/expanded_set/gx_common.py", _SESSION, _POLITE, _WHY_SESSION),
    (f"{_AG}/expanded_set/size_candidates.py", _SESSION, _POLITE, _WHY_SESSION),
    ("04_swot_targets/a_lakes/hc_pull.py", 'import pandas as pd, requests\n', 'import pandas as pd, requests\nfrom swotgat.http import polite_get\n', _WHY_GET),
    ("04_swot_targets/a_lakes/hc_pull.py", "= requests.get(", "= polite_get(", _WHY_GET),
    ("04_swot_targets/b_rivers/hc_pull_reach.py", 'import pandas as pd, requests\n', 'import pandas as pd, requests\nfrom swotgat.http import polite_get\n', _WHY_GET),
    ("04_swot_targets/b_rivers/hc_pull_reach.py", "= requests.get(", "= polite_get(", _WHY_GET),
    (f"{_RM}/grades_cascade_prep.py", 'import sys, re, time, numpy as np, pandas as pd, requests\n', 'import sys, re, time, numpy as np, pandas as pd, requests\nfrom swotgat.http import polite_get\n', _WHY_GET),
    (f"{_RM}/grades_cascade_prep.py", "= requests.get(", "= polite_get(", _WHY_GET),
    (f"{_RM}/grades_prep.py", 'import sys, json, urllib.parse, time, subprocess, numpy as np, pandas as pd, xarray as xr, requests\n', 'import sys, json, urllib.parse, time, subprocess, numpy as np, pandas as pd, xarray as xr, requests\nfrom swotgat.http import polite_get\n', _WHY_GET),
    (f"{_RM}/grades_prep.py", "= requests.get(", "= polite_get(", _WHY_GET),
    (f"{_RM}/grades_prep_2026.py", 'import sys, re, json, time, subprocess, numpy as np, pandas as pd, xarray as xr, requests\n', 'import sys, re, json, time, subprocess, numpy as np, pandas as pd, xarray as xr, requests\nfrom swotgat.http import polite_get\n', _WHY_GET),
    (f"{_RM}/grades_prep_2026.py", "= requests.get(", "= polite_get(", _WHY_GET),
    (f"{_RM}/grades_prep_early.py", 'import sys, re, json, time, subprocess, numpy as np, pandas as pd, requests\n', 'import sys, re, json, time, subprocess, numpy as np, pandas as pd, requests\nfrom swotgat.http import polite_get\n', _WHY_GET),
    (f"{_RM}/grades_prep_early.py", "= requests.get(", "= polite_get(", _WHY_GET),
]

# --- figure kit: drop reads of files from other model configurations --------------------------------------------
_WHY_KIT = ("removed a read of another model configuration's output that no panel of figure set d uses "
            "(the figure numbers and images are unchanged; checked by the smoke test)")
EDITS += [
    ("08_figures/jw_fig_tt_e.py",
     're:AT = F/"eval/compare_weekend/storage_atlas_v10pm"; recs = [^\n]*\n', "", _WHY_KIT),
    ("08_figures/combo_e.py",
     'LM = pd.read_parquet(PA/"a365"/"lm_sites.parquet").reindex(ids9); ', "", _WHY_KIT),
    ("08_figures/combo_e.py", "; y9 = LM.a_nse.values.astype(float)", "", _WHY_KIT),
    ("08_figures/combo_e.py",
     're:ref9 = json\\.load\\(open\\(PA/"a365"/"irreg_groups_numbers\\.json"\\)\\); assert [^\n]*\n', "", _WHY_KIT),
    ("08_figures/combo_e.py",
     're:assert \\{EG\\[g\\]: int\\(\\(\\(ge == g\\) & \\(b9 >= 0\\)\\)\\.sum\\(\\)\\) for g in range\\(5\\)\\} == ref9\\["n_epa"\\]\n', "", _WHY_KIT),
    ("08_figures/combo_e.py",
     'NUM["g"] = dict(edges=[float(e) for e in EDG], n_epa=ref9["n_epa"], max_abs_diff_vs_site_table=float(np.nanmax(np.abs(g482.values - y9))))',
     'y9 = g482.values.astype(float); NUM["g"] = dict(edges=[float(e) for e in EDG])',
     "the unused panel (g) now takes SWOT-GAT's NSE from the site table instead of another configuration's file"),
    ("08_figures/kit_j.py", 'LM = pd.read_parquet(PA/"a365"/"lm_sites.parquet")\n', "", _WHY_KIT),
    ("08_figures/kit_j.py",
     're:(?s)# -+ f Fig 4b: training coverage map \\(fig4_i\\.py\\)\n.*?(?=# -+ explanatory page 8 scatter)', "",
     "removed cov_map(), a panel of an earlier figure set that needs another configuration's output"),
    ("08_figures/kit_j.py", "; _L2 = LM.reindex(IDS)", "", _WHY_KIT),
    ("08_figures/kit_j.py", 're:(?s)def medians\\(x, y, k=5\\):.*\\Z', "",
     "removed medians() and cov_scatter(), panels of an earlier figure set that need another configuration's output"),
]
