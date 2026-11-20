"""2. Recurrent cell and 3. observation fusion of the C5 model.

2. Each day the LSTM step runs first, then the graph attention (GAT). The GAT revises only
   the hidden state h, never the cell state c, and the revised h is what the next day's LSTM
   step receives. Checked against a hand-written loop over days.
3. C5 has time_gaps = {"era5": False}: ERA5-Land is a dense input and there is no sparse
   (observation) source, so the fusion module never runs and no observation can reach a
   prediction. Checked on the configuration, on the built model, and by feeding targets and
   an extra observation-like input that must leave the predictions bit-for-bit unchanged.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from conftest import c5_model_args, c5_template, chain_edges, make_batch, make_c5_model, random_tree_edges

S = 6


def _manual_forward(model, batch, revised_h_carries=True):
    """Run a hand-written C5 forward pass: inference mode (no dropout), seq2seq, no area scaling."""
    emb = sum(jax.vmap(jax.vmap(model.dense_embedders[n]))(batch.dynamic[n]) for n in model.dense_sources)
    emb = jax.vmap(jax.vmap(model.fusion_norm))(emb)
    N = batch.static.shape[0]
    h = c = jnp.zeros((N, model.hidden_size))
    cell = model.spat_temp_lstm
    hs = []
    for t in range(emb.shape[0]):
        h_lstm, c_new = jax.vmap(cell.lstm_cell)(emb[t], (h, c))       # 1. LSTM step
        h_lstm = h_lstm * batch.node_mask[:, None]; c_new = c_new * batch.node_mask[:, None]
        h_gat = cell.gat(h_lstm, batch.static, tuple(batch.graph_edges), batch.node_mask,
                         batch.edge_mask, key=jax.random.PRNGKey(0))[0]   # 2. GAT on h only
        hs.append(h_gat)
        h, c = (h_gat if revised_h_carries else h_lstm), c_new
    hs = jnp.stack(hs)
    return {t: jax.vmap(jax.vmap(head))(hs) for t, head in model.head.items()}


def test_lstm_then_gat_and_gat_leaves_cell_state_alone():
    """Check that the cell runs the LSTM step, then the GAT, and that the GAT never changes c.

    c must equal the LSTM's c bit for bit and h must equal GAT(LSTM h); changing a GAT weight
    changes h but not c.
    """
    n, H = 12, 128
    model = make_c5_model(seq_length=1, static_size=S)
    cell = model.spat_temp_lstm
    e = chain_edges(n)
    k = jax.random.split(jax.random.PRNGKey(1), 3)
    x = jax.random.normal(k[0], (n, H)); h0 = jax.random.normal(k[1], (n, H)); c0 = jax.random.normal(k[2], (n, H))
    xs = jax.random.normal(jax.random.PRNGKey(2), (n, S))
    nm, em = jnp.ones(n, bool), jnp.ones(n - 1, bool)
    (h1, c1), _ = cell(x, (h0, c0), xs, tuple(jnp.asarray(e)), nm, em, key=jax.random.PRNGKey(0))
    hl, cl = jax.vmap(cell.lstm_cell)(x, (h0, c0))
    np.testing.assert_array_equal(np.asarray(c1), np.asarray(cl))       # c is the LSTM's, untouched
    h_ref = cell.gat(hl, xs, tuple(jnp.asarray(e)), nm, em, key=None)[0]
    np.testing.assert_allclose(np.asarray(h1), np.asarray(h_ref), rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.asarray(h1), np.asarray(hl))              # the GAT did change h
    # Changing the GAT's weights changes h but never c.
    cell2 = eqx.tree_at(lambda m: m.gat.reset_gate.weight, cell, cell.gat.reset_gate.weight * 2.0)
    (h2, c2), _ = cell2(x, (h0, c0), xs, tuple(jnp.asarray(e)), nm, em, key=jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(c2), np.asarray(c1))
    assert not np.array_equal(np.asarray(h2), np.asarray(h1))


def test_revised_h_carries_to_next_day():
    """Check that the GAT-revised h is what the next day's LSTM step receives.

    The model must equal a hand-written loop over days that carries the revised h, and differ
    from one that carries the pre-GAT h.
    """
    n, T = 15, 6
    model = make_c5_model(seq_length=T, static_size=S)
    b = make_batch(T=T, edges=random_tree_edges(n, 3), n_nodes=n, static_size=S, seed=4)
    got = model(b, jax.random.PRNGKey(0))
    ref = _manual_forward(model, b, revised_h_carries=True)
    alt = _manual_forward(model, b, revised_h_carries=False)
    for t in ("dv_river", "dv_SWOTAE"):
        for p in ("mu", "sigma", "log_pi"):
            np.testing.assert_allclose(np.asarray(got[t][p]), np.asarray(ref[t][p]), rtol=2e-5, atol=2e-6)
    # the test can tell the two designs apart: carrying the pre-GAT h gives other outputs from day 2 on
    g, r_, a_ = (np.asarray(x["dv_river"]["mu"]) for x in (got, ref, alt))
    assert np.max(np.abs(g[1:] - a_[1:])) > 20 * np.max(np.abs(g - r_))
    np.testing.assert_allclose(g[0], a_[0], rtol=2e-5, atol=2e-6)          # day 1 is the same either way


def test_remat_levels_do_not_change_the_forward_pass():
    """Check that the memory settings REMAT_LEVEL 0-3 give the same outputs.

    C5 trained with level 3 (checkpointed scan); inference uses 0.
    """
    import models.sparse_fusion_gann as sfg
    n, T = 10, 12
    model = make_c5_model(seq_length=T, static_size=S)
    b = make_batch(T=T, edges=chain_edges(n), n_nodes=n, static_size=S, seed=5)
    old = sfg.REMAT_LEVEL
    try:
        outs = []
        for lvl in (0, 1, 2, 3):
            sfg.REMAT_LEVEL = lvl
            outs.append(np.asarray(model(b, jax.random.PRNGKey(0))["dv_SWOTAE"]["mu"]))
    finally:
        sfg.REMAT_LEVEL = old
    for o in outs[1:]:
        np.testing.assert_allclose(o, outs[0], rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ 3. fusion
def test_c5_configuration_has_no_observation_source():
    """Check that C5's configuration gives no sparse (observation) source.

    With time_gaps = {"era5": False}, set_model_data_args must give no sparse source and one
    dense source of 12 inputs.
    """
    from models import set_model_data_args
    from config.model_args import SFGRNNArgs
    t = c5_template()
    assert t["time_gaps"] == {"era5": False}
    assert set(t["features"]["dynamic"]) == {"era5"}
    feats = dict(t["features"]["dynamic"]); feats["era5"] = feats["era5"] + t["extra_dynamic_vars"]
    ds = SimpleNamespace(features=SimpleNamespace(dynamic=feats, static=t["features"]["static"]),
                         target=t["features"]["target"])
    cfg = SimpleNamespace(model_args=SFGRNNArgs(**t["model_args"]), sequence_length=t["sequence_length"],
                          time_gaps=t["time_gaps"], area_scaled_targets=None)
    cfg = set_model_data_args(cfg, ds)
    assert cfg.model_args.sparse_sizes == {}
    assert cfg.model_args.dense_sizes == {"era5": 12}          # 10 ERA5-Land variables + 2 long-memory indices


def test_c5_model_has_no_sparse_sources_and_never_fuses():
    """Check that the built C5 model has no sparse source and never fuses.

    The fusion trace must be empty, and the fusion module must return h unchanged when it is
    given no source.
    """
    n, T = 9, 5
    model = make_c5_model(seq_length=T, static_size=S, return_weights=True)
    assert model.sparse_sources == []
    b = make_batch(T=T, edges=chain_edges(n), n_nodes=n, static_size=S)
    _, traces = model(b, jax.random.PRNGKey(0))
    assert traces["fusion"] == {"gate": None, "valid_obs": None, "weights": None}
    h = jnp.ones((n, model.hidden_size))
    h2, info = model.obs_fusion(h, [], [], [])
    assert h2 is h and info["gate"] is None


def test_observations_and_targets_cannot_reach_predictions():
    """Check that targets and observation-like inputs cannot reach the predictions.

    Adding targets, fallback targets and an extra dynamic input to the batch must leave every
    output bit-for-bit unchanged, and the gradient of the outputs with respect to the targets
    must be 0.
    """
    n, T = 11, 8
    model = make_c5_model(seq_length=T, static_size=S)
    e = random_tree_edges(n, 5)
    rng = np.random.default_rng(0)
    b0 = make_batch(T=T, edges=e, n_nodes=n, static_size=S, seed=6)
    y = {k: jnp.asarray(rng.standard_normal((T, n, 1)), jnp.float32)
         for k in ("dv_river", "dv_SWOTAE", "dv_SWOTAE__fb")}
    swot = {"swot": jnp.asarray(rng.standard_normal((T, n, 3)), jnp.float32)}
    b1 = b0._replace(y=y, dynamic={**b0.dynamic, **swot})
    k = jax.random.PRNGKey(0)
    p0, p1 = model(b0, k), model(b1, k)
    for t in p0:
        for q in p0[t]:
            np.testing.assert_array_equal(np.asarray(p0[t][q]), np.asarray(p1[t][q]))
    # and no gradient flows from the targets into the predictions
    g = jax.grad(lambda yy: model(b0._replace(y={"dv_SWOTAE": yy}), k)["dv_SWOTAE"]["mu"].sum())(y["dv_SWOTAE"])
    assert np.all(np.asarray(g) == 0.0)


def test_c5_parameter_count():
    """Check that C5 has 2,239,256 parameters.

    That is with its 300 static columns (after one-hot encoding) and 12 inputs: the number the
    training logs print ("Model contains 2,239,256 parameters") and the release README's "about
    2.2 million". The two output heads start as one shared object but are separate leaves that
    train apart, so both count. A static size of 85 (the columns before one-hot encoding) would
    give 1,688,856 instead.
    """
    import models
    assert models.count_parameters(make_c5_model(seq_length=365, static_size=300)) == 2_239_256
    assert models.count_parameters(make_c5_model(seq_length=365, static_size=85)) == 1_688_856
