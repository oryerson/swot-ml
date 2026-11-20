from typing import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from data import GraphBatch


def compute_loss(
    diff_model: PyTree,
    static_model: PyTree | None,
    data: GraphBatch,
    denorm_fn: Callable,
    key: PRNGKeyArray,
    *,

    target_weights: dict[str, int],
    corr_keys: dict = {},
    multiscale_keys: dict = {},
    varcal_keys: dict = {},
    obs_weight_keys: dict = {},
    loss_name: str = "mse",
    fallback_weights: dict[str, float] = {},
    burn_in_steps: int = 0,
    beta_nll: float = 0.0,
) -> float:
    """Compute the loss between the predicted and true values.

    Parameters
    ----------
    diff_model: PyTree
        Differential components of the model.
    static_model: PyTree
        Static components of the model.
    data: GraphBatch
        GraphBatch of data to use for training.
    keys: list[PRNGKeyArray]
        Batch of keys to use for the random dropout in the model.
    loss_name: str, optional
        Name of the loss function to use.
    target_weights: dict[str: float]
        Weights to use for the target variables.
    multiscale_keys: dict, optional
        Off when empty. {"target", "lambda", "windows" (days, default [30, 90]), "min_obs" (2),
        "use_fallback" (True), "standardize" (True)}: adds lambda * multiscale_mean_loss.
    obs_weight_keys: dict, optional
        Off when empty. {"target", "source" ("era5"), "cols" (input-column indices of standardised indices),
        "threshold" (1.5), "weight" (w)}: in that target's NLL (primary and fallback), an observation on a day
        when |x| > threshold for ANY of the listed input columns counts w times; the loss is the weighted mean.
    varcal_keys: dict, optional
        Off when empty. {"target", "lambda", "mode" ("calib" -> (alpha - r)^2, "kge" ->
        (alpha - 1)^2), "min_obs" (4), "alpha_max" (None), "use_fallback" (True)}: adds
        lambda * variability_penalty.
    fallback_weights: dict[str: float]
        For a target with a lower-tier source in data.y["<target>__fb"] (see
        Config.fallback_targets): the weight of one fallback observation relative to one
        primary observation. The target's loss is the pooled per-observation mean
        (n_p*L_p + w*n_f*L_f) / (n_p + w*n_f), so for gmm_nll it is exactly the
        weighted mean NLL over both sources, and with no fallback present it reduces
        to the original L_p.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    loss : float
    """
    model = eqx.combine(diff_model, static_model) if static_model is not None else diff_model
    y_hat = model(data, key)

    # Burn-in (R2D2-style): every window starts from a zero recurrent state, so the first
    # days are predicted with no context and the best a model can do there is regress to
    # the mean. Scoring them trains that shrinkage into every node; inference only ever
    # reads the fully warmed final day. Targets in the first `burn_in_steps` days of a
    # seq2seq window are masked out of the loss (the forward pass is unchanged).
    if burn_in_steps:
        ys = {}
        for k, v in data.y.items():
            if v.ndim == 3 and v.shape[0] > burn_in_steps:
                t = jnp.arange(v.shape[0])[:, None, None]
                v = jnp.where(t >= burn_in_steps, v, jnp.nan)
            ys[k] = v
        data = data._replace(y=ys)

    if loss_name in ["nse", "spin_up_nse"]:
        # Expand the mask with a time dimension for broadcasting.
        node_mask = data.node_mask[jnp.newaxis, :]
    else:
        node_mask = data.node_mask

    # Loop through the targets and calculate loss
    weighted_losses = []
    total_weight = 0.0
    loss_fn = get_loss_fn(loss_name)
    for target_name, y_target in data.y.items():
        # Skip targets the model didn't predict (or aren't in target_weights)
        if target_name not in y_hat:
            continue
        weight = target_weights.get(target_name, 0.0)
        if weight == 0.0:
            continue

        y_hat_target = y_hat[target_name]
        target_denorm_fn = partial(denorm_fn, name=target_name)
        # Reported observation noise (Config.obs_noise), present only when configured.
        sig = data.y.get(f"{target_name}__sigma")
        w_obs = _obs_weights(data, obs_weight_keys) if obs_weight_keys.get("target") == target_name else None
        loss, n_obs = _masked_target_loss(
            loss_fn, y_target[..., 0], y_hat_target, node_mask, target_denorm_fn,
            sigma_obs=None if sig is None else sig[..., 0], w_obs=w_obs, beta_nll=beta_nll,
        )

        fb_key = f"{target_name}__fb"
        fb_weight = fallback_weights.get(target_name, 0.0)
        if fb_key in data.y and fb_weight > 0.0:
            # Same head, same normalisation (the dataset converted the source into the
            # target's z-score); the dataset only fills it where the target is NaN.
            fb_sig = data.y.get(f"{target_name}__fb__sigma")
            fb_loss, n_fb = _masked_target_loss(
                loss_fn, data.y[fb_key][..., 0], y_hat_target, node_mask, target_denorm_fn,
                sigma_obs=None if fb_sig is None else fb_sig[..., 0], w_obs=w_obs, beta_nll=beta_nll,
            )
            denom = n_obs + fb_weight * n_fb
            loss = (n_obs * loss + fb_weight * n_fb * fb_loss) / jnp.maximum(denom, 1.0)
            n_obs = denom

        loss = jnp.where(n_obs > 0, loss, 0.0)

        weighted_losses.append(loss * weight)
        total_weight += weight

    # Normalize by total weight so overall loss scale is invariant to weight choices
    loss = jnp.sum(jnp.stack(weighted_losses)) / total_weight

    # Check for Attention Supervision
    if "attention_weights" in y_hat:
        # Re-use the node_mask for attention
        attn_loss_val = attention_leakage_loss(
            y_hat["attention_weights"],
            y_hat["valid_obs"],
            node_mask,
        )
        loss += attn_loss_val

    if corr_keys:
        x = y_hat[corr_keys['x']]
        y = data.y[corr_keys['y']][..., 0]

        # Dict output (gmm/cmal heads): reduce mixture to a point estimate before correlating.
        if isinstance(x, dict):
            weights = jnp.exp(x["log_pi"]) if "log_pi" in x else x["pi"]
            x = jnp.sum(weights * x["mu"], axis=-1)
        else:
            x = x[..., 0]

        valid_mask = ~jnp.isnan(y) & node_mask
        masked_x = jnp.where(valid_mask, x, 0)
        masked_y = jnp.where(valid_mask, y, 0)

        reg_term = correlation_penalty(
            masked_x,
            masked_y,
            valid_mask,
            method=corr_keys.get("method", "pearson"),
            tau_rel=corr_keys.get("tau_rel", 0.1),
        )
        loss = loss + (corr_keys['lambda'] * reg_term)

    if multiscale_keys:
        k = multiscale_keys
        p, y, m = _point_and_obs(y_hat, data, k["target"], node_mask, k.get("use_fallback", True))
        scale = _mixture_sd(y_hat[k["target"]]) if k.get("standardize", True) else None
        loss = loss + k["lambda"] * multiscale_mean_loss(
            y, p, m, windows=tuple(k.get("windows", (30, 90))), min_obs=k.get("min_obs", 2), scale=scale
        )

    if varcal_keys:
        k = varcal_keys
        p, y, m = _point_and_obs(y_hat, data, k["target"], node_mask, k.get("use_fallback", True))
        loss = loss + k["lambda"] * variability_penalty(
            y, p, m, mode=k.get("mode", "calib"), min_obs=k.get("min_obs", 4), alpha_max=k.get("alpha_max")
        )

    return loss


def _mixture_mean(x: dict) -> Array:
    """Point estimate of a GMM head: sum_k pi_k mu_k (log_pi renormalised, so exact either way)."""
    return jnp.sum(jax.nn.softmax(x["log_pi"], axis=-1) * x["mu"], axis=-1)


def _mixture_sd(x: dict) -> Array:
    """Predictive sd of a GMM head, gradient-stopped: a per-node, per-day scale only."""
    w = jax.nn.softmax(x["log_pi"], axis=-1)
    m = jnp.sum(w * x["mu"], axis=-1)
    var = jnp.sum(w * (jnp.square(x["sigma"]) + jnp.square(x["mu"])), axis=-1) - jnp.square(m)
    return jax.lax.stop_gradient(jnp.sqrt(jnp.maximum(var, 1e-12)))


def _point_and_obs(y_hat, data, target, node_mask, use_fallback=True):
    """(mixture mean, observations, mask), each (time, node), for one seq2seq GMM target.

    Observations are the target where present, else its fallback source (y["<t>__fb"],
    already in the target's z-units and only filled where the target is NaN), which is
    what the NLL is trained on when fallback_weights is 1.0.
    """
    x = y_hat[target]
    if not isinstance(x, dict):
        raise ValueError(f"variability terms need a gmm head; {target} is not one")
    p = _mixture_mean(x)
    y = data.y[target][..., 0]
    fb = data.y.get(f"{target}__fb")
    if use_fallback and fb is not None:
        y = jnp.where(jnp.isnan(y), fb[..., 0], y)
    if y.ndim != 2 or p.shape != y.shape:
        raise ValueError(f"variability terms need seq2seq (time, node) outputs, got {p.shape} vs {y.shape}")
    mask = ~jnp.isnan(y) & node_mask
    return p, jnp.where(mask, y, 0.0), mask


def multiscale_mean_loss(
    y: Array, y_hat: Array, mask: Array, windows=(30, 90), min_obs: int = 2, scale: Array | None = None
) -> Array:
    """Squared error of window means, averaged over windows and then over timescales.

    For each timescale w the sequence is cut into non-overlapping w-day windows aligned
    to its LAST day (any leading remainder is dropped; inference reads the last day).
    Per node and window, both the prediction and the observations are averaged over the
    SAME observed days, so the term is the window-mean residual and has no sampling
    mismatch from SWOT's sparse repeat. Windows with fewer than `min_obs` observations
    are ignored. Only the low-frequency part of the error counts: a residual that
    alternates sign within a window contributes ~0 here, while a sustained offset (the
    damped monthly and seasonal swings) contributes its full square.

    scale (time, node), optional: each window error is divided by the mean of `scale`
    over its observed days. Passing the gradient-stopped mixture sd standardises the
    term the way the NLL standardises each residual, so nodes are weighted alike instead
    of by (reservoir size)^2, which z-units of a single global y_scale would do.
    The standardised error is capped at +-20 like gmm_nll's.
    """
    T = y.shape[0]
    m_all = mask.astype(y.dtype)
    terms = []
    for w in windows:
        nw = T // w
        if nw == 0:
            continue
        sl = slice(T - nw * w, T)
        shape = (nw, w) + y.shape[1:]
        m = m_all[sl].reshape(shape)
        n = jnp.sum(m, axis=1)
        ok = n >= min_obs
        n_safe = jnp.where(ok, n, 1.0)
        err = (jnp.sum(y_hat[sl].reshape(shape) * m, axis=1) - jnp.sum(y[sl].reshape(shape) * m, axis=1)) / n_safe
        if scale is not None:
            s = jnp.sum(jnp.where(mask, scale, 0.0)[sl].reshape(shape) * m, axis=1) / n_safe
            err = jnp.clip(err / jnp.where(ok, jnp.maximum(s, 1e-6), 1.0), -20.0, 20.0)
        terms.append(jnp.sum(jnp.where(ok, jnp.square(err), 0.0)) / jnp.maximum(jnp.sum(ok), 1.0))
    if not terms:
        return jnp.asarray(0.0, y.dtype)
    return jnp.mean(jnp.stack(terms))


def variability_penalty(
    y: Array, y_hat: Array, mask: Array, mode: str = "calib", min_obs: int = 4, alpha_max: float | None = None
) -> Array:
    """Per-node variability penalty from observed days pooled over the sequence.

    alpha = sd(y_hat)/sd(y) and r = corr(y_hat, y) over each node's observed days.
    mode="calib": (alpha - r)^2. alpha = r is the condition an MSE-optimal (conditional
        mean) prediction satisfies, so this pushes the swing toward what the timing skill
        supports. r is gradient-stopped: the term moves alpha only, and can never lower
        r to meet a too-small alpha.
    mode="kge":   (alpha - 1)^2, the KGE variability term; it inflates the swing whatever
        r is, which costs MSE/NSE wherever r < 1 (NSE = 2 alpha r - alpha^2 peaks at alpha = r).
    Nodes with fewer than `min_obs` observations or no variance in either series are
    skipped; the result is the mean over the rest.
    alpha_max (optional): alpha is capped there before the penalty. Inside one ~90-day
    window a node whose observations are nearly flat gets alpha >> 1 from a handful of
    points; uncapped, those nodes dominate the mean and its gradient (the term then
    mostly flattens predictions). Above the cap a node adds a constant and no gradient. Inputs are sanitised before every sqrt
    so empty or flat columns cannot put a NaN in the gradient (see _masked_pearson).
    """
    if mode not in ("calib", "kge"):
        raise ValueError(f"unknown variability mode: {mode!r}")
    EPS = 1e-12
    mf = mask.astype(y.dtype)
    n = jnp.sum(mf, axis=0)
    n_safe = jnp.where(n > 0, n, 1.0)
    pc = (y_hat - jnp.sum(y_hat * mf, axis=0) / n_safe) * mf
    oc = (y - jnp.sum(y * mf, axis=0) / n_safe) * mf
    ss_p = jnp.sum(jnp.square(pc), axis=0)
    ss_o = jnp.sum(jnp.square(oc), axis=0)
    valid = (n >= min_obs) & (ss_p > EPS) & (ss_o > EPS)
    ss_p = jnp.where(valid, ss_p, 1.0)
    ss_o = jnp.where(valid, ss_o, 1.0)
    alpha = jnp.sqrt(ss_p) / jnp.sqrt(ss_o)
    if alpha_max is not None:
        alpha = jnp.minimum(alpha, alpha_max)
    if mode == "calib":
        r = jnp.sum(pc * oc, axis=0) / (jnp.sqrt(ss_p) * jnp.sqrt(ss_o))
        goal = jax.lax.stop_gradient(r)
    else:
        goal = 1.0
    node = jnp.where(valid, jnp.square(alpha - goal), 0.0)
    return jnp.sum(node) / jnp.maximum(jnp.sum(valid), 1.0)


def _obs_weights(data, k):
    """(time, node) weights: k["weight"] where any listed input column exceeds |threshold|, else 1."""
    x = data.dynamic[k.get("source", "era5")][..., jnp.asarray(list(k["cols"]))]
    hit = jnp.any(jnp.abs(jnp.nan_to_num(x)) > k.get("threshold", 1.5), axis=-1)
    return jnp.where(hit, float(k["weight"]), 1.0)


def _masked_target_loss(loss_fn, y_true, y_hat_target, node_mask, denorm_fn, sigma_obs=None, w_obs=None, beta_nll=0.0):
    """Loss of one target against one set of observations, and how many were valid.

    The loss is zeroed when nothing is valid (mean-type losses return NaN on an empty
    selection), and inputs are sanitised before the loss so no NaN reaches the gradient.

    sigma_obs (GMM heads only): each observation's reported 1-sigma noise in the target's
    z-units. Every mixture component is widened to sqrt(sigma_k^2 + sigma_obs^2), i.e. the
    likelihood of a noisy observation of the modelled quantity. None = unchanged.
    """
    valid_mask = ~jnp.isnan(y_true) & node_mask
    masked_y = jnp.where(valid_mask, y_true, 0)

    # Handle GMM (dict output) vs Standard (Array output)
    if isinstance(y_hat_target, dict):
        expanded_mask = jnp.expand_dims(valid_mask, axis=-1)
        sigma = y_hat_target["sigma"]
        if sigma_obs is not None:
            so = jnp.where(valid_mask, jnp.nan_to_num(sigma_obs), 0.0)[..., None]
            sigma = jnp.sqrt(jnp.square(sigma) + jnp.square(so))
        masked_y_hat = {
            "mu": jnp.where(expanded_mask, y_hat_target["mu"], 0.0),
            "sigma": jnp.where(expanded_mask, sigma, 1.0),  # sigma=1 is safe
            "log_pi": jnp.where(
                expanded_mask, y_hat_target["log_pi"], 1.0 / y_hat_target["log_pi"].shape[-1]
            ),  # uniform is safe
        }
    else:
        masked_y_hat = jnp.where(valid_mask, y_hat_target[..., 0], 0)

    if beta_nll and isinstance(y_hat_target, dict):
        # beta-NLL (bnll stage): weight = stop_gradient(mixture variance)^beta, mean 1 over the valid observations; combined
        # multiplicatively with any obs_weight_keys weights. Uses the observation-widened sigma when obs noise is configured.
        pi_ = jnp.exp(masked_y_hat["log_pi"]); mu_, sg_ = masked_y_hat["mu"], masked_y_hat["sigma"]
        m1_ = jnp.sum(pi_ * mu_, axis=-1)
        var_ = jnp.sum(pi_ * (jnp.square(sg_) + jnp.square(mu_)), axis=-1) - jnp.square(m1_)
        wb = jax.lax.stop_gradient(jnp.power(jnp.maximum(var_, 1e-6), beta_nll))
        wb = jnp.where(valid_mask, wb, 0.0)
        wb = wb / jnp.maximum(jnp.sum(wb) / jnp.maximum(jnp.sum(valid_mask), 1.0), 1e-12)
        w_obs = wb if w_obs is None else wb * jnp.where(valid_mask, w_obs, 0.0)
    if w_obs is None:
        loss = loss_fn(masked_y, masked_y_hat, valid_mask, denorm_fn)
        n_obs = jnp.sum(valid_mask).astype(jnp.float32)
    else:
        # Weighted mean: the mean-type losses use the mask as a multiplier, so a float weight mask gives
        # sum(w * l) / sum(w); n_obs becomes the weight total so fallback pooling stays exact.
        if loss_fn is not gmm_nll:
            raise ValueError("obs_weight_keys is implemented for gmm_nll only")
        wm = jnp.where(valid_mask, w_obs, 0.0)
        loss = loss_fn(masked_y, masked_y_hat, wm, denorm_fn)
        n_obs = jnp.sum(wm).astype(jnp.float32)
    return jnp.where(n_obs > 0, loss, 0.0), n_obs


def get_loss_fn(loss_name):
    match loss_name:
        case "mse":
            return mse_loss
        case "mae":
            return mae_loss
        case "huber":
            return huber_loss
        case "nse":
            return nse_loss
        case "spin_up_nse":
            return spin_up_nse_loss
        case "gmm_nll":
            return gmm_nll
        case _:
            raise ValueError(f"Config'd loss ({loss_name}) not found.")


def mse_loss(y: Array, y_hat: Array, mask: Array, denorm_fn: Callable):
    """Calculates Mean Squared Error (MSE) loss."""
    mse = jnp.mean(jnp.square(y - y_hat), where=mask)
    return mse


def mae_loss(y: Array, y_hat: Array, mask: Array, denorm_fn: Callable):
    """Calculates Mean Absolute Error (MAE) loss."""
    mae = jnp.mean(jnp.abs(y - y_hat), where=mask)
    return mae


def huber_loss(
    y: Array, y_hat: Array, mask: Array, denorm_fn: Callable, *, huber_delta: float = 1.0
):
    """Calculates Huber loss."""
    # Passing delta through make_step is not yet implemented.
    residual = y - y_hat
    condition = jnp.abs(residual) <= huber_delta
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = huber_delta * (jnp.abs(residual) - 0.5 * huber_delta)
    return jnp.mean(jnp.where(condition, squared_loss, linear_loss), where=mask)


def nse_loss(y: Array, y_hat: Array, mask: Array, denorm_fn: Callable):
    """
    Calculates the smooth-joint NSE from Kratzert et al., 2019
    https://doi.org/10.5194/hess-23-5089-2019
    """
    sq_error = jnp.square(y - y_hat) * mask
    mse = jnp.mean(sq_error, axis=0)  # mean squared errors per basin

    std_y = jnp.std(y, axis=0, where=mask.astype(bool))  # Per-basin standard deviation
    stable_std_y = jnp.nan_to_num(std_y) + 0.1
    denom = jnp.square(stable_std_y)

    node_nse = mse / denom
    jnp.mean(node_nse, where=node_nse != 0)


def spin_up_nse_loss(y: Array, y_hat: Array, mask: Array, denorm_fn: Callable):
    """Equivalent to nse_loss but with weights that favor the final ~1/3 of the sequence"""
    y = denorm_fn(y)
    y_hat = denorm_fn(y_hat)

    seq_len = y.shape[0]
    weights = jax.nn.sigmoid(jnp.linspace(-10, 10, seq_len))

    sq_error = jnp.square(y - y_hat) * mask
    mse = jnp.average(sq_error, axis=0, weights=weights)  # mean squared errors per basin

    std_y = jnp.std(y, axis=0, where=mask.astype(bool))  # Per-basin standard deviation
    stable_std_y = jnp.nan_to_num(std_y) + 0.1
    denom = jnp.square(stable_std_y)

    node_nse = mse / denom
    return jnp.mean(node_nse, where=node_nse != 0)


def gmm_nll(y: Array, y_hat: dict[str, Array], mask: Array, denorm_fn: Callable) -> Array:
    y_obs = jnp.expand_dims(y, axis=-1)
    mu, sigma, log_pi = y_hat["mu"], y_hat["sigma"], y_hat["log_pi"]

    log_sigma = jnp.log(sigma)
    log_2pi = jnp.log(2.0 * jnp.pi)

    # Calculate z-score: (obs - pred) / std
    scaled_diff = (y_obs - mu) / sigma
    # Cap the squared residual to prevent a single outlier from exploding gradients
    sq_diff = jnp.square(jnp.clip(scaled_diff, -20.0, 20.0))

    log_phi = -log_sigma - 0.5 * log_2pi - 0.5 * sq_diff

    # Numerical stability: logsumexp avoids pi * exp(log_phi) overflow
    log_likelihood = jax.scipy.special.logsumexp(log_pi + log_phi, axis=-1)

    return -jnp.sum(log_likelihood * mask) / (jnp.sum(mask) + 1e-8)


def attention_leakage_loss(attn_weights: Array, obs_mask: Array, node_mask: Array) -> Array:
    """
    Penalizes attention to masked (invalid) time steps AND encourages
    attention to valid time steps when they exist.
    """
    # Original penalty: don't attend to null token when real data exists
    real_data_exists = jnp.any(obs_mask[..., :-1], axis=-1)
    null_attn_weights = attn_weights[..., -1]
    leakage_penalty = null_attn_weights * real_data_exists

    # Sum of attention weights on valid time steps
    valid_attn_weights = attn_weights[..., :-1]  # Exclude null token
    valid_attn_sum = jnp.sum(valid_attn_weights * obs_mask[..., :-1], axis=-1)

    # Penalize low attention to valid data (when valid data exists)
    underutilization_penalty = (1.0 - valid_attn_sum) * real_data_exists

    # Combine both penalties
    total_penalty = leakage_penalty + underutilization_penalty

    return jnp.mean(total_penalty, where=real_data_exists)

def _masked_pearson(a: Array, b: Array, mask: Array) -> tuple[Array, Array]:
    """Masked Pearson r along axis 0, plus a per-column validity flag.

    NaN-gradient safety: a finite forward value is not enough here. Reverse-mode AD
    differentiates BOTH branches of a jnp.where and d/dx sqrt(x) is infinite at x=0,
    so columns with no observations (all-zero centred series, zero variance) poisoned
    the whole gradient tree even though the returned value looked fine. That is why
    training once reported "Global norm of grads: nan" at every checkpoint while the
    loss stayed finite. Every dangerous op below has its INPUT sanitised first (the
    "double where" pattern), and reductions use sum/count rather than mean(where=...)
    so an empty selection can never produce a NaN to differentiate through. Both
    correlation methods route through here so that fix cannot be lost again.
    """
    EPS = 1e-12
    valid_count = jnp.sum(mask, axis=0)
    safe_count = jnp.where(valid_count > 0, valid_count, 1.0)

    mean_a = jnp.sum(a * mask, axis=0) / safe_count
    mean_b = jnp.sum(b * mask, axis=0) / safe_count
    a_centered = (a - mean_a) * mask
    b_centered = (b - mean_b) * mask

    cov = jnp.sum(a_centered * b_centered, axis=0)
    ss_a = jnp.sum(jnp.square(a_centered), axis=0)
    ss_b = jnp.sum(jnp.square(b_centered), axis=0)

    has_var = (ss_a > EPS) & (ss_b > EPS)
    ss_a_safe = jnp.where(has_var, ss_a, 1.0)
    ss_b_safe = jnp.where(has_var, ss_b, 1.0)
    r = jnp.where(has_var, cov / (jnp.sqrt(ss_a_safe) * jnp.sqrt(ss_b_safe)), 0.0)
    valid = (valid_count > 1) & has_var
    return r, valid


def _soft_rank(x: Array, mask: Array, tau_rel: float) -> Array:
    """Differentiable approximation of within-column ranks along axis 0.

    rank_i = sum_j sigmoid((x_i - x_j) / tau) over VALID j only, converging to the
    true rank as tau -> 0. Hard ranks (argsort) have zero gradient almost everywhere
    and cannot be trained through.

    tau scales with each column's own spread, so the ranks are invariant to rescaling
    x - the property that makes rank correlation worth having. A fixed absolute tau
    would quietly make the result depend on units.
    """
    EPS = 1e-12
    valid_count = jnp.sum(mask, axis=0)
    safe_count = jnp.where(valid_count > 0, valid_count, 1.0)
    mean_x = jnp.sum(x * mask, axis=0) / safe_count
    var_x = jnp.sum(jnp.square((x - mean_x) * mask), axis=0) / safe_count
    has_var = var_x > EPS
    std_x = jnp.sqrt(jnp.where(has_var, var_x, 1.0))   # sanitise before sqrt
    tau = jnp.where(has_var, tau_rel * std_x, 1.0)

    diff = x[:, None, ...] - x[None, :, ...]           # (T, T, N)
    s = jax.nn.sigmoid(diff / tau[None, None, ...])
    s = s * mask[None, :, ...]                         # only valid j contribute
    return jnp.sum(s, axis=1) * mask                   # (T, N)


def correlation_penalty(
    y: Array, y_hat: Array, mask: Array, method: str = "pearson", tau_rel: float = 0.1
) -> Array:
    """Correlation penalty |1-r| over valid sequences, averaged over valid columns.

    Inputs bypass denormalization to constrain execution within normalized space.

    method="pearson"  : linear correlation. Original behaviour, and the default, so
        existing configs are unchanged.
    method="spearman" : correlation of soft ranks, so any MONOTONE relationship can
        reach r=1. Discharge is log-normalized while d_wse_river is an anomaly and a
        rating curve is a power law, so the two are related monotonically but not
        linearly. Measured on the Columbia record, Pearson r(log Q, dh) has median
        +0.469 with 13% of nodes NEGATIVE - meaning |1-r| could never fall below
        ~0.53 and pushed the wrong way at those nodes. Spearman on the same data
        gives median +0.655 with only 4% negative.
    """
    if method == "spearman":
        a = _soft_rank(y_hat, mask, tau_rel)
        b = _soft_rank(y, mask, tau_rel)
    elif method == "pearson":
        a, b = y_hat, y
    else:
        raise ValueError(f"unknown correlation method: {method!r}")

    r, valid = _masked_pearson(a, b, mask)
    node_loss = jnp.where(valid, jnp.abs(1.0 - r), 0.0)
    return jnp.sum(node_loss) / jnp.maximum(jnp.sum(valid), 1.0)
