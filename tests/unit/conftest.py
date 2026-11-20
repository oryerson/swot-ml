"""Shared set-up for the unit tests of the SWOT-GAT model and scoring code.

* JAX runs on the CPU (set before JAX is imported anywhere).
* The vendored model source (vendor/swot_ml/src) is put first on sys.path, so
  ``import models`` etc. load exactly the code the paper's model C5 was trained with.
* Helpers build the C5 model (values from the release's configuration template)
  and small synthetic graph batches.
* ``load_function`` / ``run_statements`` take a function or a few top-level
  statements out of a scoring script's source text and execute only those, so
  scripts that do their work at import time can still be tested piece by piece
  without touching their inputs.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "vendor" / "swot_ml" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO / "src") not in sys.path:
    sys.path.insert(1, str(REPO / "src"))

TEMPLATES = {
    "S1": REPO / "05_training/configs/era5_conus_v35c5_S1_storage.json.template",
    "S2": REPO / "05_training/configs/era5_conus_v35c5_S2_storage_s1.json.template",
}
ORIGINAL_ROOT = "/work/pi_casey_umass_edu/Owen/Reservoirs"   # the project root the paper's runs used

# Paths of the paper's runs on the authors' cluster; tests that need them are skipped elsewhere.
RUN_S1 = Path(ORIGINAL_ROOT) / "CONUS/era5_conus_v35c5_S1_storage_20261028_074428"
RUN_S2 = Path(ORIGINAL_ROOT) / "CONUS/era5_conus_v35c5_S2_storage_s1_20261028_085321"
STAGED_SRC = Path(ORIGINAL_ROOT) / "CONUS_storage_fork/eval/v35_launch/stage_src_v35"
on_cluster = pytest.mark.skipif(not (RUN_S1 / "checkpoints/step_050000/model.eqx").is_file(),
                                reason="needs the paper's checkpoints and data (authors' cluster)")

TARGETS = ["dv_river", "dv_SWOTAE"]


def c5_template(seed: str = "S1") -> dict:
    """Return the C5 training configuration as shipped (placeholders left in)."""
    return json.loads(TEMPLATES[seed].read_text())


def c5_model_args(*, seq_length: int, static_size: int, n_dyn: int = 12, area_scaled=None,
                  return_weights: bool = False, **override):
    """Return SFGRNNArgs filled the way models.set_model_data_args fills them for C5.

    Hidden size 128, k_hops 5, 4 heads, GMM head, seq2seq, dropout 0.3 and static dropout 0.2
    come from the template; data-dependent sizes are given here (C5 itself: 12 ERA5-Land
    inputs incl. two long-memory indices, 300 static columns after one-hot encoding).
    Keyword arguments in ``override`` replace the template's model_args values.
    """
    from config.model_args import SFGRNNArgs
    t = c5_template()
    args = SFGRNNArgs(**{**t["model_args"], **override})
    args.target = list(TARGETS)
    args.seq_length = seq_length
    gaps = t["time_gaps"]
    args.dense_sizes = {k: n_dyn for k, g in gaps.items() if not g}
    args.sparse_sizes = {k: n_dyn for k, g in gaps.items() if g}
    args.static_size = static_size
    args.area_scaled = area_scaled
    args.return_weights = return_weights
    return args


def make_c5_model(*, seq_length: int, static_size: int, n_dyn: int = 12, area_scaled=None,
                  inference: bool = True, return_weights: bool = False, **override):
    """Build the C5 model (SparseFusionGRNN) the way models.make builds it.

    The arguments are those of c5_model_args; ``inference`` sets the model's inference mode
    (True = dropout off).
    """
    import equinox as eqx
    import models
    args = c5_model_args(seq_length=seq_length, static_size=static_size, n_dyn=n_dyn,
                         area_scaled=area_scaled, return_weights=return_weights, **override)
    model = models.MODEL_MAP[args.name](**args.as_kwargs())
    return eqx.nn.inference_mode(model, inference)


# ---------------------------------------------------------------- graphs
def chain_edges(n: int) -> np.ndarray:
    """Return the edges of a chain in which node i drains into node i+1 (node n-1 is the outlet).

    The array has shape (2, n-1); row 0 is src (upstream), row 1 is dest (downstream).
    """
    return np.array([np.arange(n - 1), np.arange(1, n)], dtype=np.int32)


def random_tree_edges(n: int, seed: int = 0) -> np.ndarray:
    """Return the edges of a random river tree, shape (2, n-1), row 0 = src (upstream).

    Every node but the outlet (node n-1) drains into one node with a larger id.
    """
    rng = np.random.default_rng(seed)
    src = np.arange(n - 1)
    dest = np.array([rng.integers(i + 1, n) for i in src])
    return np.array([src, dest], dtype=np.int32)


def undirected_distances(edges: np.ndarray, n: int, start: int) -> np.ndarray:
    """Return each node's hop distance from `start`, ignoring flow direction (inf: unreachable)."""
    adj = [[] for _ in range(n)]
    for s, d in edges.T:
        adj[s].append(d); adj[d].append(s)
    dist = np.full(n, np.inf); dist[start] = 0; frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1; nxt.append(v)
        frontier = nxt
    return dist


def make_batch(*, T: int, edges: np.ndarray, n_nodes: int, static_size: int, n_dyn: int = 12,
               seed: int = 0, node_mask=None, edge_mask=None, y=None, extra_dynamic=None):
    """Build a GraphBatch (JAX arrays) with standard-normal inputs on a given graph.

    The dynamic source "era5" has shape (T, n_nodes, n_dyn) and static (n_nodes, static_size).
    All nodes and edges are real unless node_mask / edge_mask say otherwise (no padding),
    ``y`` defaults to no targets, and ``extra_dynamic`` adds further dynamic sources.
    """
    from data import GraphBatch
    rng = np.random.default_rng(seed)
    dyn = {"era5": rng.standard_normal((T, n_nodes, n_dyn)).astype(np.float32)}
    if extra_dynamic:
        dyn.update(extra_dynamic)
    static = rng.standard_normal((n_nodes, static_size)).astype(np.float32)
    E = edges.shape[1]
    return GraphBatch(
        dynamic=dyn,
        graph_edges=np.asarray(edges, dtype=np.int32),
        static=static,
        node_mask=np.ones(n_nodes, bool) if node_mask is None else np.asarray(node_mask, bool),
        edge_mask=np.ones(E, bool) if edge_mask is None else np.asarray(edge_mask, bool),
        y=y if y is not None else {},
    ).to_jax()


# ---------------------------------------------------------------- source surgery
def _module_ast(path: Path) -> ast.Module:
    return ast.parse(Path(path).read_text(), filename=str(path))


def load_function(path: Path, name: str, namespace: dict | None = None):
    """Compile and return the top-level function `name` of a script WITHOUT running the script.

    The function is executed in `namespace` (its globals), which must provide the
    names it uses (np, pd, module constants...).
    """
    tree = _module_ast(path)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise LookupError(f"{path}: expected one top-level function {name!r}, found {len(nodes)}")
    mod = ast.Module(body=nodes, type_ignores=[])
    ns = {} if namespace is None else namespace
    exec(compile(mod, str(path), "exec"), ns)
    return ns[name]


def run_statements(path: Path, select, namespace: dict) -> dict:
    """Execute only the top-level statements of a script for which select(source_text) is true."""
    src = Path(path).read_text()
    tree = ast.parse(src, filename=str(path))
    body = [n for n in tree.body if select(ast.get_source_segment(src, n) or "")]
    if not body:
        raise LookupError(f"{path}: no top-level statement matched")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="session")
def jax_cpu():
    """Return the jax module after checking that it runs on the CPU."""
    import jax
    assert jax.devices()[0].platform == "cpu", jax.devices()
    return jax
