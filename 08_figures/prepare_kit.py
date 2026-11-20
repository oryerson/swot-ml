"""Assemble the tables that the four figure scripts read (figure set version d).

The figure scripts (fig1_c5d.py ... fig4_c5d.py) read their score tables from
their own folder, the "kit". This script fills that folder from the scorer
outputs of stage 07:

==============================  =============================================
kit file                        built from
==============================  =============================================
sites_2010_g.csv                overlap50/score/sites_2010_g.csv
sites_2325span_g.csv            overlap50/score/sites_2325span_g.csv
score_j/fullrecord_site_window  overlap50/score/score_j/fullrecord_site_window.csv
score_full_d.csv                full-record scorer, 1981-1990 and 1981-2020 windows
sites_2325span_ens_d.csv        SWOT Period scorer with the rule-model ensemble
irreg_0020.parquet              irreg0020_g.py output (read at import; no panel uses it)
==============================  =============================================

Two filters are applied, exactly as when the paper figures were made:

1. The baseline rows (SWOT seasonal cycle, gauge normal year) are removed from
   the overlap50 tables; the figures treat them separately.
2. For the reservoirs in ``excl_d.EXCL`` every rule-model row (GDROM, ISTARF,
   MODROM, and the rule-model ensemble RULE-ENS) is removed. These are the
   reservoirs where the published ISTARF parameters pin storage at full or
   empty. SWOT-GAT rows are kept, so maps and SWOT-GAT-only panels still show
   them.

Usage: ``bin/run 08_figures/prepare_kit.py`` (after stage 07 scoring).
"""
from swotgat.paths import WORK
import io
import shutil
import sys
from pathlib import Path

import pandas as pd

E = Path(f"{WORK}/CONUS_storage_fork/eval")
O50 = E/"overlap50"
B = E/"grl_figs/intermediate/c5_o50_20261113"
K = B/"kit_d"
IRREG = E/"grl_figs/intermediate/a365_vDspan_20261030/figset_a365e/irreg_0020.parquet"

sys.path.insert(0, str(K))
from excl_d import EXCL  # noqa: E402

BASELINES = ["SWOT-cycle", "normal_year"]
RULE = {"GDROM", "ISTARF", "MODROM-cal", "MODROM-tr", "RULE-ENS"}


def drop_excluded(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Remove the rule-model rows of the excluded reservoirs."""
    keep = ~(df.DAM_ID.isin(EXCL) & df[model_col].isin(RULE))
    print(f"  dropped {int((~keep).sum())} rule-model rows of excluded reservoirs")
    return df[keep]


def without_baselines(path: Path, model_col: str, baselines: list[str]) -> pd.DataFrame:
    """Read a scorer table, drop the baseline rows, and pass it through CSV text once.

    The paper's tables were staged in two steps, each of which wrote the table
    to CSV and read it back. pandas' default number parser can change the last
    digit of a value on such a round trip, so the same two steps are kept here
    to reproduce the paper's tables byte for byte.
    """
    table = pd.read_csv(path)
    table = table[~table[model_col].isin(baselines)]
    return pd.read_csv(io.StringIO(table.to_csv(index=False)))


def main() -> None:
    """Fill the kit folder with the filtered score tables and the irregularity table."""
    (K/"score_j").mkdir(parents=True, exist_ok=True)

    for name in ("sites_2010_g.csv", "sites_2325span_g.csv"):
        print(name)
        table = without_baselines(O50/"score"/name, "MODEL", BASELINES)
        drop_excluded(table, "MODEL").to_csv(K/name, index=False)

    print("score_j/fullrecord_site_window.csv")
    table = without_baselines(O50/"score/score_j/fullrecord_site_window.csv", "model", ["SWOT-cycle"])
    drop_excluded(table, "model").to_csv(K/"score_j/fullrecord_site_window.csv", index=False)

    print("score_full_d.csv")
    table = pd.read_csv(B/"score/score_full/fullrecord_site_window.csv")
    drop_excluded(table, "model").to_csv(K/"score_full_d.csv", index=False)

    print("sites_2325span_ens_d.csv")
    table = pd.read_csv(B/"score/sites_2325span_g.csv")
    drop_excluded(table, "MODEL").to_csv(K/"sites_2325span_ens_d.csv", index=False)

    shutil.copy2(IRREG, K/"irreg_0020.parquet")
    print("kit ready:", K, "| excluded reservoirs:", sorted(EXCL))


if __name__ == "__main__":
    main()
