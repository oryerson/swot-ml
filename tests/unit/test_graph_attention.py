"""1. Graph attention of SparseFusionGRNN (vendor/swot_ml/src/models/sparse_fusion_gann.py).

Edge convention of the graph files and the data loader: edge (src, dest) means src drains
into dest (src = upstream, dest = downstream).

Checked here:
* segment_softmax sums to 1 over each receiving node's incoming edges and equals a
  per-node softmax;
* downstream aggregation: a node attends only over the nodes that drain into it;
* upstream propagation: a node receives only from its single downstream neighbour;
* masked edges and masked (padding) nodes contribute nothing;
* a node with no neighbours receives no information from other nodes;
* one daily step moves information exactly k_hops = 5 hops (undirected graph distance),
  and t+1 days move it 5(t+1) hops.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (chain_edges, make_batch, make_c5_model, random_tree_edges,
                      undirected_distances)
from models.sparse_fusion_gann import (BidirectionalGraphLayer, DownstreamAggregation,
                                       SpatioTemporalLSTMCell, UpstreamPropagation,
                                       segment_softmax)

H, HEADS, S = 128, 4, 6
# Small river tree: 0->2, 1->2, 2->4, 3->4, 4->5 (outlet); node 6 has no edges.
TREE = np.array([[0, 1, 2, 3, 4], [2, 2, 4, 4, 5]], dtype=np.int32)
N_TREE = 7


def _inputs(n, seed=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.normal(k1, (n, H)), jax.random.normal(k2, (n, S))


def _dependence(fn, h):
    """Return D, where D[i, j] is True if output row i depends on input row j (exact Jacobian)."""
    J = jax.jacobian(fn)(h)                      # (N, H, N, H)
    return np.asarray(jnp.any(J != 0, axis=(1, 3)))


# ------------------------------------------------------------------ segment softmax
def test_segment_softmax_sums_to_one_per_receiver():
    """Check that attention weights sum to 1 over each node's incoming edges.

    They must also equal a per-node softmax and stay finite for large logits.
    """
    rng = np.random.default_rng(0)
    E, N = 40, 9
    seg = rng.integers(0, N - 1, E)              # node N-1 receives nothing
    logits = rng.standard_normal((E, HEADS)) * 30.0   # large values: the max-shift must keep it stable
    w = np.asarray(segment_softmax(jnp.asarray(logits), jnp.asarray(seg), N))
    assert np.all(np.isfinite(w)) and np.all(w >= 0)
    for n in range(N - 1):
        k = seg == n
        if k.any():
            np.testing.assert_allclose(w[k].sum(0), 1.0, rtol=1e-5)
            ref = np.exp(logits[k] - logits[k].max(0)); ref /= ref.sum(0)
            np.testing.assert_allclose(w[k], ref, rtol=1e-5, atol=1e-7)


# ------------------------------------------------------------------ downstream aggregation
def test_downstream_attends_only_over_upstream_neighbours():
    """Check that a node's downstream message depends only on the nodes draining into it.

    Exact Jacobian. The node's own state (the query) matters only when it has two or more
    tributaries; nodes without tributaries get exactly 0; the attention weights sum to 1 per
    receiving node and head.
    """
    layer = DownstreamAggregation(HEADS, H, S, key=jax.random.PRNGKey(1))
    h, xs = _inputs(N_TREE)
    nm, em = jnp.ones(N_TREE, bool), jnp.ones(TREE.shape[1], bool)
    out, w = layer(h, xs, tuple(jnp.asarray(TREE)), nm, em)
    D = _dependence(lambda hh: layer(hh, xs, tuple(jnp.asarray(TREE)), nm, em)[0], h)
    for d in range(N_TREE):
        ups = set(TREE[0][TREE[1] == d])
        # values of its tributaries; its own query matters only when there are >= 2 to weigh
        # (a softmax over one edge is 1 whatever the score)
        expected = (ups | {d}) if len(ups) > 1 else ups
        assert set(np.flatnonzero(D[d])) == expected, (d, np.flatnonzero(D[d]), expected)
        if not ups:                                       # headwaters, outlet-less isolated node
            assert np.all(np.asarray(out[d]) == 0.0)
    # attention weights: per receiving node, per head, they sum to 1
    for d in set(TREE[1]):
        np.testing.assert_allclose(np.asarray(w)[TREE[1] == d].sum(0), 1.0, rtol=1e-5)


def test_downstream_masked_edge_contributes_nothing():
    """Check that a masked edge gets attention weight 0 and its sender has no influence."""
    layer = DownstreamAggregation(HEADS, H, S, key=jax.random.PRNGKey(2))
    h, xs = _inputs(N_TREE, 1)
    em = jnp.array([False, True, True, True, True])       # mask edge 0 -> 2 (node 2 keeps edge 1 -> 2)
    f = lambda hh: layer(hh, xs, tuple(jnp.asarray(TREE)), jnp.ones(N_TREE, bool), em)
    D = _dependence(lambda hh: f(hh)[0], h)
    assert not D[2, 0] and D[2, 1]
    assert float(f(h)[1][0].max()) == 0.0                 # weight of the masked edge


def test_downstream_masked_node_output_is_zero():
    """Check that a masked node's downstream message is 0."""
    layer = DownstreamAggregation(HEADS, H, S, key=jax.random.PRNGKey(3))
    h, xs = _inputs(N_TREE, 2)
    nm = jnp.ones(N_TREE, bool).at[4].set(False)
    out, _ = layer(h, xs, tuple(jnp.asarray(TREE)), nm, jnp.ones(5, bool))
    assert np.all(np.asarray(out[4]) == 0.0)


@pytest.mark.xfail(strict=True, reason=(
    "FINDING: if EVERY incoming edge of a real node is masked, all its scores are -1e9, "
    "segment_softmax returns uniform weights and the masked senders' values are summed in. "
    "The data loader never creates this case (padding edges are self-loops on the first "
    "padding node, which is node-masked), so the paper's runs are unaffected; see "
    "test_padding_nodes_never_reach_real_nodes."))
def test_downstream_all_incoming_edges_masked_gives_no_message():
    """Check that a node whose incoming edges are all masked gets no message (expected failure)."""
    layer = DownstreamAggregation(HEADS, H, S, key=jax.random.PRNGKey(4))
    h, xs = _inputs(N_TREE, 3)
    em = jnp.array([False, False, True, True, True])      # both edges into node 2 masked
    out, _ = layer(h, xs, tuple(jnp.asarray(TREE)), jnp.ones(N_TREE, bool), em)
    assert np.all(np.asarray(out[2]) == 0.0)


# ------------------------------------------------------------------ upstream propagation
def test_upstream_receives_only_from_single_downstream_neighbour():
    """Check that a node's upstream message depends only on itself and its one downstream neighbour.

    The message must equal message_net x sigmoid(gate_net) computed by hand; the outlet and the
    isolated node get 0.
    """
    layer = UpstreamPropagation(H, S, key=jax.random.PRNGKey(5))
    h, xs = _inputs(N_TREE, 4)
    nm, em = jnp.ones(N_TREE, bool), jnp.ones(5, bool)
    out, _ = layer(h, xs, tuple(jnp.asarray(TREE)), nm, em)
    D = _dependence(lambda hh: layer(hh, xs, tuple(jnp.asarray(TREE)), nm, em)[0], h)
    for r in range(N_TREE):
        down = TREE[1][TREE[0] == r]
        assert len(down) <= 1
        if len(down) == 0:                                 # outlet and isolated node
            assert not D[r].any() and np.all(np.asarray(out[r]) == 0.0)
            continue
        d = int(down[0])
        assert set(np.flatnonzero(D[r])) == {r, d}         # gate sees receiver and sender
        # by hand: message_net(h_d, x_d) * sigmoid(gate_net(h_d, h_r, x_d, x_r))
        msg = layer.message_net(jnp.concatenate([h[d], xs[d]]))
        gate = jax.nn.sigmoid(layer.gate_net(jnp.concatenate([h[d], h[r], xs[d], xs[r]])))
        np.testing.assert_allclose(np.asarray(out[r]), np.asarray(msg * gate), rtol=1e-5, atol=1e-6)


def test_upstream_masked_edge_contributes_nothing():
    """Check that a masked edge carries no upstream message."""
    layer = UpstreamPropagation(H, S, key=jax.random.PRNGKey(6))
    h, xs = _inputs(N_TREE, 5)
    em = jnp.array([True, True, False, True, True])        # mask 2 -> 4
    out, _ = layer(h, xs, tuple(jnp.asarray(TREE)), jnp.ones(N_TREE, bool), em)
    assert np.all(np.asarray(out[2]) == 0.0)


# ------------------------------------------------------------------ isolated node, padding
def test_isolated_node_receives_no_information():
    """Check that a node without edges depends on no other node.

    Its per-layer update is the constant combine_proj bias, not a message.
    """
    layer = BidirectionalGraphLayer(HEADS, H, S, key=jax.random.PRNGKey(7))
    h, xs = _inputs(N_TREE, 6)
    f = lambda hh: layer(hh, xs, tuple(jnp.asarray(TREE)), jnp.ones(N_TREE, bool), jnp.ones(5, bool))[0]
    D = _dependence(f, h)
    assert set(np.flatnonzero(D[6])) == {6}
    # Its update is the constant combine_proj bias (same for every node), not a message:
    np.testing.assert_allclose(np.asarray(f(h)[6] - h[6]), np.asarray(layer.combine_proj.bias), atol=1e-6)


def test_padding_nodes_never_reach_real_nodes():
    """Check that no padding node influences a real node and that packed graphs stay separate.

    Two chains are packed with the real padding_collate_fn.
    """
    from data import GraphBatch
    from data.cached_basingraphdataloader import padding_collate_fn
    T, nd = 2, 12
    samples = []
    for i, n in enumerate((4, 3)):
        e = chain_edges(n)
        samples.append(("b%d" % i, [str(k) for k in range(n)], np.datetime64("2024-01-01"),
                        GraphBatch(dynamic={"era5": np.zeros((T, n, nd), np.float32)}, graph_edges=e,
                                   static=np.zeros((n, S), np.float32), y={})))
    _, _, _, b = padding_collate_fn(samples, target_nodes_per_batch=10)
    assert b.node_mask.sum() == 7 and b.edge_mask.sum() == 5
    pad_edges = b.graph_edges[:, ~b.edge_mask]
    assert np.all(pad_edges == 7)                          # self-loops on the first padding node
    cell = SpatioTemporalLSTMCell(H, S, H, 5, HEADS, 0.0, key=jax.random.PRNGKey(8))
    h, _ = _inputs(10, 7); xs = jax.random.normal(jax.random.PRNGKey(9), (10, S))
    x = jax.random.normal(jax.random.PRNGKey(10), (10, H))
    f = lambda hh: cell(x, (hh, jnp.zeros((10, H))), xs, tuple(jnp.asarray(b.graph_edges)),
                        jnp.asarray(b.node_mask), jnp.asarray(b.edge_mask), key=jax.random.PRNGKey(0))[0][0]
    D = _dependence(f, h)
    assert not D[:7, 7:].any()                             # no padding row reaches a real node
    assert not D[:4, 4:7].any() and not D[4:7, :4].any()   # the two packed graphs stay separate


# ------------------------------------------------------------------ k-hop reach
def _cell_reach(edges, n, j, k_hops=5):
    cell = SpatioTemporalLSTMCell(H, S, H, k_hops, HEADS, 0.0, key=jax.random.PRNGKey(11))
    x = jax.random.normal(jax.random.PRNGKey(12), (n, H))
    xs = jax.random.normal(jax.random.PRNGKey(13), (n, S))
    st = (jax.random.normal(jax.random.PRNGKey(14), (n, H)), jax.random.normal(jax.random.PRNGKey(15), (n, H)))
    run = lambda xx: cell(xx, st, xs, tuple(jnp.asarray(edges)), jnp.ones(n, bool),
                          jnp.ones(edges.shape[1], bool), key=jax.random.PRNGKey(0))[0]
    (h0, c0), (h1, c1) = run(x), run(x.at[j].add(3.0))
    changed_h = np.flatnonzero(np.any(np.asarray(h0) != np.asarray(h1), axis=1))
    changed_c = np.flatnonzero(np.any(np.asarray(c0) != np.asarray(c1), axis=1))
    return set(changed_h), set(changed_c)


@pytest.mark.parametrize("edges,n,j", [(chain_edges(25), 25, 12), (random_tree_edges(40, 1), 40, 17),
                                       (random_tree_edges(40, 2), 40, 3)])
def test_one_step_reaches_exactly_k_hops(edges, n, j):
    """Check that one daily step moves information exactly 5 hops (undirected distance).

    Perturbing node j's input must change h at exactly the nodes within 5 hops, and c only at node j.
    """
    ch, cc = _cell_reach(edges, n, j)
    dist = undirected_distances(edges, n, j)
    assert ch == set(np.flatnonzero(dist <= 5)), (sorted(ch), dist)
    assert cc == {j}                                       # the cell state is per node (LSTM only)


def test_full_model_reach_grows_5_hops_per_day():
    """Check that the whole model's reach grows by 5 hops per day.

    On a 31-node chain, perturbing one node's forcing on day 0 must change the day-t predictions
    at exactly the nodes within 5(t+1) hops.
    """
    n, j, T, nd = 31, 15, 3, 12
    model = make_c5_model(seq_length=T, static_size=S)
    e = chain_edges(n)
    b0 = make_batch(T=T, edges=e, n_nodes=n, static_size=S)
    b1 = b0._replace(dynamic={"era5": b0.dynamic["era5"].at[0, j].add(3.0)})
    k = jax.random.PRNGKey(0)
    p0, p1 = model(b0, k), model(b1, k)
    for t in range(T):
        diff = np.any(np.asarray(p0["dv_river"]["mu"][t]) != np.asarray(p1["dv_river"]["mu"][t]), axis=1)
        assert set(np.flatnonzero(diff)) == set(range(max(0, j - 5 * (t + 1)), min(n, j + 5 * (t + 1) + 1))), (t, np.flatnonzero(diff))
