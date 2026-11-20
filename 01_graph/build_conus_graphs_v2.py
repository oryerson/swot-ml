"""Build the river-network graph of all basins from the sub-basin table.

The graph has one node per sub-basin and one directed link from each node to
the node it drains into (column nextdown). Basins never share a link, so each
basin is one weakly connected component of the graph.

Reads
  --subbasins  sub-basin table (Parquet). The index is the node id; the columns
               used are nextdown (id of the node downstream, empty at a basin
               outlet) and outlet_id (the basin's outlet gauge).
               Default $SWOTGAT_WORK/CONUS/metadata/subbasins_conus_v2.parquet.
               For the paper's graph, pass the graph v3.4 or v3.5 table.

Writes
  --out        the graph as networkx node-link JSON (default
               $SWOTGAT_WORK/CONUS/metadata/graphs_conus_v2.json). The model's
               data loader (CachedBasinGraphDataset) splits it back into one
               sub-graph per basin with weakly_connected_components.

Usage
  python build_conus_graphs_v2.py --subbasins <table.parquet> --out <graph.json> [--dry-run]

Method
  1. A basin whose outlet_id is not itself a node of the table is left out of
     the graph, with a warning (its outlet gauge could not be delineated).
  2. A terminal node (nextdown empty) must be its basin's outlet. Any other
     terminal node is an interior dead end: the script stops with an error
     instead of wiring that node's upstream network straight to the outlet.
  3. Checks: every component has exactly one outlet, and every graph node has
     a row in the sub-basin table.

In the paper's run, 2 of the 102 basins are left out by rule 1, which leaves 100.
"""
from swotgat.paths import WORK

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

# Defaults only. The pipeline always passes --subbasins and --out explicitly.
SUBS = Path(f"{WORK}/CONUS/metadata/subbasins_conus_v2.parquet")
OUT = Path(f"{WORK}/CONUS/metadata/graphs_conus_v2.json")


def parse_args():
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subbasins", type=Path, default=SUBS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_graph(subs: pd.DataFrame, allow_interior_terminals: bool = False) -> nx.DiGraph:
    """Build the directed graph (node -> nextdown) from the sub-basin table.

    subs is indexed by node id and has the columns nextdown and outlet_id. Terminal nodes are named after
    their basin's outlet_id. A basin whose outlet_id is not a node of the table is left out with a warning.
    A terminal node that is not its basin's outlet raises ValueError, unless allow_interior_terminals is
    True (then it is renamed to the outlet, for comparison only). See the module docstring for the rules.
    """
    subs = subs.copy()
    subs.index = subs.index.astype(str)
    subs.index.name = "site_id"
    subs = subs.reset_index()

    # nextdown must be comparable to site_id for edges to resolve. Delineation
    # writes it as a float/object id, so normalise to string and keep NaN as NaN.
    subs["nextdown"] = subs["nextdown"].apply(
        lambda v: pd.NA if pd.isna(v) else str(int(v)) if isinstance(v, float) else str(v)
    )

    # Terminal nodes take their basin's outlet_id; repoint edges that referenced them.
    #
    # Only rename when the outlet_id is itself a delineated subbasin. If a basin's
    # outlet gauge never became a node (delineation dropped it), renaming invents a
    # graph node with no row in the subbasins/static tables. Nothing catches that at
    # build time: the graph validates fine, and training instead dies later with
    # KeyError: "not all values found in index 'subbasin'" on the fraction of samples
    # drawn from that basin. Such basins are excluded here, loudly, so the graph is
    # always a strict subset of the subbasins table.
    known = set(subs["site_id"])
    term = subs["nextdown"].isna()
    orphan_basins = sorted(set(subs.loc[term & ~subs["outlet_id"].isin(known), "outlet_id"]))
    if orphan_basins:
        print(f"WARNING: {len(orphan_basins)} basin(s) whose outlet_id is not a delineated "
              f"subbasin -- excluding them from the graph:")
        for b in orphan_basins:
            n = int((subs["outlet_id"] == b).sum())
            print(f"    {b}  ({n} subbasins)")
        subs = subs[~subs["outlet_id"].isin(orphan_basins)].copy()
        term = subs["nextdown"].isna()

    # An interior terminal is a node with nextdown NaN that is not its basin's outlet. Renaming it to the outlet (below)
    # would silently wire its whole upstream network straight to the basin outlet (in an early graph the upper Columbia
    # skipped The Dalles this way). Fail instead. allow_interior_terminals=True keeps the rename, for comparison only.
    interior = term & (subs["site_id"] != subs["outlet_id"])
    if interior.any() and not allow_interior_terminals:
        bad = subs.loc[interior, ["site_id", "outlet_id"]].itertuples(index=False, name=None)
        raise ValueError(f"{int(interior.sum())} interior terminal(s) (nextdown NaN, not the basin outlet); "
                         f"refusing to rewire them to the outlet: {sorted(bad)[:25]}")
    mapping = subs.loc[term].set_index("site_id")["outlet_id"]
    subs.loc[term, "site_id"] = subs.loc[term, "outlet_id"]
    subs["nextdown"] = subs["nextdown"].replace(mapping.to_dict())

    G = nx.DiGraph()
    G.add_nodes_from(subs["site_id"])
    edges = list(
        subs.loc[subs["nextdown"].notna(), ["site_id", "nextdown"]].itertuples(index=False, name=None)
    )
    G.add_edges_from(edges)
    return G


def validate(G: nx.DiGraph, expected_basins: int):
    """Check the graph and return its weakly connected components.

    Raise ValueError unless every component has exactly one outlet (a node with no outgoing link) and the
    number of components equals expected_basins.
    """
    components = list(nx.weakly_connected_components(G))
    for nodes in components:
        subG = G.subgraph(nodes)
        outlets = [n for n in subG.nodes if subG.out_degree(n) == 0]
        if len(outlets) != 1:
            raise ValueError(f"Expected exactly one outlet per basin, found {outlets}")
    if len(components) != expected_basins:
        raise ValueError(
            f"Graph has {len(components)} connected components but {expected_basins} basins "
            "were supplied -- a basin may be split or two basins may have fused."
        )
    return components


def main():
    """Build the graph from the sub-basin table, check it, print a summary and write the JSON."""
    args = parse_args()
    subs = pd.read_parquet(args.subbasins, columns=["nextdown", "outlet_id"])
    n_basins = subs["outlet_id"].nunique()
    print(f"read {len(subs):,} subbasins across {n_basins} basins")

    G = build_graph(subs)
    # basins actually represented after any exclusions inside build_graph
    n_in_graph = len(list(nx.weakly_connected_components(G)))
    components = validate(G, n_in_graph)
    if n_in_graph != n_basins:
        print(f"note: {n_basins - n_in_graph} basin(s) excluded; graph covers {n_in_graph}")

    print(f"Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    print(f"Components (basins): {len(components)}")
    print(f"Sources (headwaters): {sum(1 for n in G.nodes() if G.in_degree(n) == 0):,}")
    print(f"Sinks (outlets): {sum(1 for n in G.nodes() if G.out_degree(n) == 0):,}")
    for nodes in sorted(components, key=len, reverse=True):
        subG = G.subgraph(nodes)
        outlet = [n for n in subG.nodes if subG.out_degree(n) == 0][0]
        print(f"  {outlet:<18} {len(nodes):>6} nodes")

    # The invariant every downstream consumer assumes: every graph node has a row
    # in the subbasins table (and therefore in the static attributes).
    subs_all = pd.read_parquet(args.subbasins, columns=["outlet_id"])
    known_ids = set(subs_all.index.astype(str)) | set(subs_all["outlet_id"].astype(str))
    phantom = {str(n) for n in G.nodes} - known_ids
    if phantom:
        sys.exit(f"ERROR: {len(phantom)} graph node(s) absent from the subbasins table, "
                 f"e.g. {sorted(phantom)[:5]}")
    print("check: every graph node exists in the subbasins table")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(nx.readwrite.json_graph.node_link_data(G, edges="edges"), f)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
