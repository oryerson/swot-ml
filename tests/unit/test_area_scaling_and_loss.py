"""4. Area scaling of dv_SWOTAE and 5. the training loss (vendor/swot_ml/src/train/loss.py).

4. The GMM head of dv_SWOTAE is read as a lake-level change (m) and mapped to a volume change
   in the target's z-units with the node's lake area A (km2; km2 x m = MCM):
       mu' = (A*mu - y_offset)/y_scale,   sigma' = A*sigma/y_scale,   log_pi unchanged.
   The NLL of the volume then equals the NLL of the level plus log(A/y_scale).
   A is recovered from the normalised static column (lake_area_km2 is log1p-normalised),
   from the static values BEFORE static dropout.
5. gmm_nll equals a hand computation; only observed days, real nodes and days after the
   90-day burn-in count; the fallback (dv_staticPLD in dv_SWOTAE's z-units) is pooled with
   the primary target per observation; target weights apply as configured.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from types import SimpleNamespace

from conftest import c5_template, chain_edges, make_batch, make_c5_model
from train.loss import compute_loss, gmm_nll

S = 6
# C5's own constants (checkpoint config.json of seed S1): lake_area_km2 is static column 1.
C5_AREA = dict(target="dv_SWOTAE", static_idx=1, area_offset=0.6044181628564236, area_scale=1.0,
               area_log_norm=True, y_offset=-5.7623244434649665, y_scale=54.43990157093096)


def nll_np(y, mu, sigma, log_pi):
    """Return the Gaussian-mixture NLL per observation in float64, residuals capped at 20 sd."""
    z = np.clip((y[..., None] - mu) / sigma, -20, 20)
    log_phi = -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z ** 2
    return -logsumexp(log_pi + log_phi, axis=-1)


def _static_with_area(area, n_cols=S, seed=0):
    rng = np.random.default_rng(seed)
    st = rng.standard_normal((len(area), n_cols)).astype(np.float32)
    st[:, 1] = (np.log1p(area) - C5_AREA["area_offset"]) / C5_AREA["area_scale"]
    return jnp.asarray(st)


# ------------------------------------------------------------------ 4. area scaling
def test_area_scaling_formula():
    """Check the area-scaling formula on a 100-component mixture with C5's constants.

    mu' = (A*mu - y_offset)/y_scale, sigma' = A*sigma/y_scale, log_pi unchanged; dv_river is
    left untouched.
    """
    model = make_c5_model(seq_length=3, static_size=S, area_scaled=[C5_AREA])
    rng = np.random.default_rng(1)
    T, N, K = 3, 5, 100
    area = np.array([0.05, 1.0, 12.5, 300.0, 4000.0])
    p = {"mu": jnp.asarray(rng.standard_normal((T, N, K)), jnp.float32),
         "sigma": jnp.asarray(rng.uniform(0.01, 2, (T, N, K)), jnp.float32),
         "log_pi": jnp.asarray(np.log(rng.dirichlet(np.ones(K), (T, N))), jnp.float32)}
    out = model._apply_area_scaling({"dv_SWOTAE": dict(p), "dv_river": dict(p)}, _static_with_area(area))
    A = area[None, :, None]
    yo, ys = C5_AREA["y_offset"], C5_AREA["y_scale"]
    np.testing.assert_allclose(np.asarray(out["dv_SWOTAE"]["mu"]), (A * np.asarray(p["mu"]) - yo) / ys, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out["dv_SWOTAE"]["sigma"]), A * np.asarray(p["sigma"]) / ys, rtol=1e-4)
    np.testing.assert_array_equal(np.asarray(out["dv_SWOTAE"]["log_pi"]), np.asarray(p["log_pi"]))
    assert out["dv_river"] is not None and np.array_equal(np.asarray(out["dv_river"]["mu"]), np.asarray(p["mu"]))  # river untouched


def test_area_scaling_floor_for_nodes_without_lake_area():
    """Check that nodes with NaN or zero lake area get A = 1e-3, so sigma stays positive."""
    model = make_c5_model(seq_length=1, static_size=S, area_scaled=[C5_AREA])
    st = np.array(_static_with_area(np.array([1.0, 1.0, 1.0])))
    st[0, 1] = np.nan                                      # no lake area
    st[1, 1] = -C5_AREA["area_offset"]                      # area 0
    p = {"mu": jnp.ones((3, 2)), "sigma": jnp.ones((3, 2)), "log_pi": jnp.log(jnp.full((3, 2), .5))}
    out = model._apply_area_scaling({"dv_SWOTAE": p}, jnp.asarray(st))
    sig = np.asarray(out["dv_SWOTAE"]["sigma"])[:, 0]
    np.testing.assert_allclose(sig[:2], 1e-3 / C5_AREA["y_scale"], rtol=1e-4)   # sigma stays > 0
    assert np.all(np.isfinite(np.asarray(out["dv_SWOTAE"]["mu"])))


def test_volume_nll_equals_level_nll_plus_log_area_over_scale():
    """Check NLL(volume) = NLL(level) + log(A/y_scale) for 40 random mixtures of 100 components."""
    model = make_c5_model(seq_length=1, static_size=S, area_scaled=[C5_AREA])
    rng = np.random.default_rng(2)
    N, K = 40, 100
    area = np.exp(rng.uniform(np.log(0.1), np.log(5000), N))
    mu = rng.normal(0, 1.0, (N, K)); sigma = rng.uniform(0.05, 1.5, (N, K))
    log_pi = np.log(rng.dirichlet(np.ones(K), N))
    level = rng.normal(0, 1.0, N)                          # observed level change (m)
    vol = area * level                                     # MCM
    y_z = (vol - C5_AREA["y_offset"]) / C5_AREA["y_scale"]
    out = model._apply_area_scaling({"dv_SWOTAE": {"mu": jnp.asarray(mu, jnp.float32), "sigma": jnp.asarray(sigma, jnp.float32),
                                                   "log_pi": jnp.asarray(log_pi, jnp.float32)}}, _static_with_area(area))["dv_SWOTAE"]
    A = np.exp(np.array(_static_with_area(area))[:, 1].astype(np.float64) + C5_AREA["area_offset"]) - 1  # area as the model sees it (float32 static)
    nll_vol = np.array([float(gmm_nll(jnp.asarray(y_z[i:i + 1], jnp.float32), {k: v[i:i + 1] for k, v in out.items()}, jnp.ones(1), None)) for i in range(N)])
    nll_lvl = nll_np(level, mu, sigma, log_pi)
    np.testing.assert_allclose(nll_vol, nll_lvl + np.log(A / C5_AREA["y_scale"]), rtol=2e-4, atol=2e-3)


def test_area_scaling_uses_static_before_dropout():
    """Check that in training mode (static dropout 0.2) the scaling uses the area before dropout."""
    n, T = 30, 4
    area = np.exp(np.random.default_rng(3).uniform(0, 6, n))
    with_a = make_c5_model(seq_length=T, static_size=S, area_scaled=[C5_AREA], inference=False)
    without = make_c5_model(seq_length=T, static_size=S, area_scaled=None, inference=False)
    b = make_batch(T=T, edges=chain_edges(n), n_nodes=n, static_size=S)
    b = b._replace(static=_static_with_area(area))
    key = jax.random.PRNGKey(7)
    ps, pu = with_a(b, key), without(b, key)               # same seed -> same weights, same dropout masks
    ref = with_a._apply_area_scaling(dict(pu), b.static)
    for q in ("mu", "sigma", "log_pi"):
        np.testing.assert_allclose(np.asarray(ps["dv_SWOTAE"][q]), np.asarray(ref["dv_SWOTAE"][q]), rtol=1e-5, atol=1e-6)
    dropped = with_a.static_dropout(b.static, key=jax.random.split(key, 4)[3])
    assert np.any(np.asarray(dropped)[:, 1] != np.asarray(b.static)[:, 1])  # dropout did hit the area column
    bad = with_a._apply_area_scaling(dict(pu), dropped)
    assert not np.allclose(np.asarray(ps["dv_SWOTAE"]["sigma"]), np.asarray(bad["dv_SWOTAE"]["sigma"]))


def test_area_scaled_args_from_dataset_scalers():
    """Check that the scaling constants come from the dataset's own scalers.

    These are the log1p-normalised lake_area_km2 and the z-scored dv_SWOTAE.
    """
    from models import _area_scaled_args
    t = c5_template()
    assert t["area_scaled_targets"] == {"dv_SWOTAE": "lake_area_km2"}
    assert "lake_area_km2" in t["log_norm_cols"] and "dv_SWOTAE" not in t["log_norm_cols"]
    ds = SimpleNamespace(features=SimpleNamespace(static=t["features"]["static"]),
                         s_scale={"lake_area_km2": {"offset": 0.6, "scale": 1.0, "log_norm": True}},
                         d_scale={"dv_SWOTAE": {"offset": -5.7, "scale": 54.4, "log_norm": False}})
    cfg = SimpleNamespace(area_scaled_targets=t["area_scaled_targets"])
    assert _area_scaled_args(cfg, ds) == [dict(target="dv_SWOTAE", static_idx=1, area_offset=0.6, area_scale=1.0,
                                               area_log_norm=True, y_offset=-5.7, y_scale=54.4)]


# ------------------------------------------------------------------ 5. loss
def test_gmm_nll_matches_hand_computation():
    """Check gmm_nll against a float64 hand computation of the mixture NLL.

    The hand computation caps each standardised residual at 20 sd, as the code does.
    """
    rng = np.random.default_rng(4)
    T, N, K = 7, 6, 5
    y = rng.normal(0, 1, (T, N)); mu = rng.normal(0, 1, (T, N, K)); sigma = rng.uniform(.2, 2, (T, N, K))
    log_pi = np.log(rng.dirichlet(np.ones(K), (T, N)))
    y[0, 0] = 60.0; mu[0, 0] = 0.0; sigma[0, 0] = 0.5        # residual of 120 sd: capped at 20 sd
    mask = rng.random((T, N)) < 0.6
    got = float(gmm_nll(jnp.asarray(y, jnp.float32), {"mu": jnp.asarray(mu, jnp.float32), "sigma": jnp.asarray(sigma, jnp.float32),
                                                       "log_pi": jnp.asarray(log_pi, jnp.float32)}, jnp.asarray(mask), None))
    want = nll_np(y, mu, sigma, log_pi)[mask].mean()
    np.testing.assert_allclose(got, want, rtol=1e-5)


def _loss_setup(T=100, n=8, seed=0):
    model = make_c5_model(seq_length=T, static_size=S, area_scaled=[C5_AREA])
    b = make_batch(T=T, edges=chain_edges(n), n_nodes=n, static_size=S, seed=seed,
                   node_mask=np.r_[np.ones(n - 1, bool), False])   # last node = padding
    key = jax.random.PRNGKey(0)
    y_hat = model(b, key)
    return model, b, key, y_hat


def _y(T, n, frac, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, (T, n, 1)).astype(np.float32)
    v[rng.random((T, n)) > frac] = np.nan
    return v


def _target_loss(y_hat_t, y, node_mask, burn=90):
    T = y.shape[0]
    valid = ~np.isnan(y[..., 0]) & node_mask[None, :] & (np.arange(T)[:, None] >= burn)
    nll = nll_np(np.nan_to_num(y[..., 0]), np.asarray(y_hat_t["mu"], np.float64), np.asarray(y_hat_t["sigma"], np.float64),
                 np.asarray(y_hat_t["log_pi"], np.float64))
    return nll[valid].sum(), valid.sum()


def _compute(model, b, key, y, weights, fb=None, burn=90):
    return float(compute_loss(model, None, b._replace(y={k: jnp.asarray(v) for k, v in y.items()}), lambda x, name: x, key,
                              target_weights=weights, loss_name="gmm_nll", fallback_weights=fb or {}, burn_in_steps=burn))


def test_loss_counts_only_observed_days_real_nodes_after_burn_in():
    """Check that the loss is the hand mean over observed days >= 90 on real nodes only.

    Changing the targets inside the burn-in or at the padding node must leave the loss unchanged.
    """
    model, b, key, y_hat = _loss_setup()
    T, n = 100, 8
    y = {"dv_river": _y(T, n, .3, 1), "dv_SWOTAE": _y(T, n, .2, 2)}
    nm = np.asarray(b.node_mask)
    sr, nr = _target_loss(y_hat["dv_river"], y["dv_river"], nm)
    ss, ns = _target_loss(y_hat["dv_SWOTAE"], y["dv_SWOTAE"], nm)
    assert nr > 0 and ns > 0
    got = _compute(model, b, key, y, {"dv_river": 1.0, "dv_SWOTAE": 1.0})
    np.testing.assert_allclose(got, (sr / nr + ss / ns) / 2, rtol=1e-5)
    # changing observations in the burn-in, at the padding node, or at unobserved days changes nothing
    y2 = {k: v.copy() for k, v in y.items()}
    for v in y2.values():
        v[:90] = np.random.default_rng(9).normal(0, 5, v[:90].shape)
        v[:, -1] = 3.0
    assert _compute(model, b, key, y2, {"dv_river": 1.0, "dv_SWOTAE": 1.0}) == got


def test_burn_in_is_90_days_in_c5():
    """Check that C5's burn-in is 90 days.

    Observations on days 0-88 only give loss 0; one more on day 90 (the 91st day) does not.
    """
    assert c5_template()["step_kwargs"]["burn_in_steps"] == 90
    model, b, key, y_hat = _loss_setup()
    T, n = 100, 8
    y = {"dv_river": _y(T, n, .3, 3), "dv_SWOTAE": _y(T, n, .3, 4)}
    for v in y.values():
        v[89:] = np.nan                                    # observations only on days 0..88
    assert _compute(model, b, key, y, {"dv_river": 1.0, "dv_SWOTAE": 1.0}) == 0.0
    y["dv_river"][90, 0, 0] = 0.3                          # day 90 (91st day) is the first scored day
    assert _compute(model, b, key, y, {"dv_river": 1.0, "dv_SWOTAE": 1.0}) != 0.0


def test_fallback_pooled_with_primary_per_observation():
    """Check that the fallback target is pooled with the primary target per observation.

    The dv_SWOTAE loss must be (n_p L_p + w n_f L_f)/(n_p + w n_f) for fallback weight w = 1
    (C5) and w = 0.25, and the primary loss alone when no fallback weight is given.
    """
    model, b, key, y_hat = _loss_setup()
    T, n = 100, 8
    prim = _y(T, n, .2, 5)
    src = np.random.default_rng(6).normal(0, 1, (T, n, 1)).astype(np.float32)
    fb = np.where(np.isnan(prim), src, np.nan).astype(np.float32)       # the dataset's rule
    fb[np.random.default_rng(7).random((T, n)) > .5] = np.nan
    river = _y(T, n, .3, 8)
    nm = np.asarray(b.node_mask)
    sp, np_ = _target_loss(y_hat["dv_SWOTAE"], prim, nm)
    sf, nf = _target_loss(y_hat["dv_SWOTAE"], fb, nm)
    sr, nr = _target_loss(y_hat["dv_river"], river, nm)
    y = {"dv_river": river, "dv_SWOTAE": prim, "dv_SWOTAE__fb": fb}
    # C5: fallback weight 1 = one pooled mean over primary and fallback observations
    got = _compute(model, b, key, y, {"dv_river": 1.0, "dv_SWOTAE": 1.0}, fb={"dv_SWOTAE": 1.0})
    np.testing.assert_allclose(got, ((sp + sf) / (np_ + nf) + sr / nr) / 2, rtol=1e-5)
    # another weight: (n_p L_p + w n_f L_f) / (n_p + w n_f)
    got = _compute(model, b, key, y, {"dv_river": 1.0, "dv_SWOTAE": 1.0}, fb={"dv_SWOTAE": 0.25})
    np.testing.assert_allclose(got, ((sp + .25 * sf) / (np_ + .25 * nf) + sr / nr) / 2, rtol=1e-5)
    # weight 0 / no fallback configured: the fallback is ignored
    got = _compute(model, b, key, y, {"dv_river": 1.0, "dv_SWOTAE": 1.0}, fb={})
    np.testing.assert_allclose(got, (sp / np_ + sr / nr) / 2, rtol=1e-5)
    assert c5_template()["fallback_targets"] == {"dv_SWOTAE": {"source": "dv_staticPLD", "weight": 1.0, "subsets": ["train"]}}


def test_target_weights():
    """Check that the loss is the weighted mean over targets, sum(w_t L_t)/sum(w_t).

    C5 uses weights 1 and 1.
    """
    model, b, key, y_hat = _loss_setup()
    T, n = 100, 8
    y = {"dv_river": _y(T, n, .3, 10), "dv_SWOTAE": _y(T, n, .3, 11)}
    nm = np.asarray(b.node_mask)
    Lr = np.divide(*_target_loss(y_hat["dv_river"], y["dv_river"], nm))
    Ls = np.divide(*_target_loss(y_hat["dv_SWOTAE"], y["dv_SWOTAE"], nm))
    np.testing.assert_allclose(_compute(model, b, key, y, {"dv_river": 2.0, "dv_SWOTAE": 1.0}), (2 * Lr + Ls) / 3, rtol=1e-5)
    np.testing.assert_allclose(_compute(model, b, key, y, {"dv_river": 0.0, "dv_SWOTAE": 1.0}), Ls, rtol=1e-5)
    assert c5_template()["step_kwargs"]["target_weights"] == {"dv_river": 1.0, "dv_SWOTAE": 1.0}


def test_target_without_observations_still_counts_in_the_weight_total():
    """Check that a target with no observation in the batch keeps its weight in the denominator.

    Documents the normalisation: such a target adds 0 to the sum, so the batch loss is the other
    target's NLL divided by 2 (C5: weights 1 and 1).
    """
    model, b, key, y_hat = _loss_setup()
    T, n = 100, 8
    y = {"dv_river": np.full((T, n, 1), np.nan, np.float32), "dv_SWOTAE": _y(T, n, .3, 12)}
    Ls = np.divide(*_target_loss(y_hat["dv_SWOTAE"], y["dv_SWOTAE"], np.asarray(b.node_mask)))
    np.testing.assert_allclose(_compute(model, b, key, y, {"dv_river": 1.0, "dv_SWOTAE": 1.0}), Ls / 2, rtol=1e-5)
