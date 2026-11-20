"""Shared helpers for training and prediction on graph v3.5: paths, checksums, configurations, store digests.

This module was written to launch and compare several model configurations on graph v3.5, called C1, C2 and C3
in the code. The paper uses only configuration C5 (365-day sequences, two long-memory inputs, graph v3.5), whose
configuration files are the templates in 05_training/configs/. C5 is not built by this module.

What the paper's pipeline uses from this module
  src()                           folder of the model source, read from the environment variable V35_SRC or from
                                  the one-line file v35_launch/SRC that tools/deploy.py writes
                                  (build_cache_v35.py)
  data_digests(), store_digest()  content digests of the data stores, to make sure a training cache is used only
                                  with the store content it was built from (build_cache_v35.py, train.sh)
  md5(), GRAPH_MD5                checksum of the graph v3.5 file (06_predictions/pred_assert_c5.py and
                                  pred_assert_v35.py)
Everything else is described as it is but is not run for the paper. It built and checked the configurations
C1 to C3, and read the hand-over files of the steps that prepared their inputs.

Configurations named in this module
  A180   an earlier model configuration on graph version 2.2: 180-day sequences, 90-day burn-in, 365 days of
         input history, 50,000 steps. It is the reference the others were compared with.
  C1     A180's training recipe on graph v3.5, with the graph v3.5 store, attributes and lists, and one more
         static attribute
  C2     C1 plus long-memory inputs, trained from scratch
  C3     a C1 checkpoint fine-tuned with long-memory inputs (recipe in FT)

Hand-over files (not used for C5)
  READY_v35.json, READY_v35_pred.json    written when the graph v3.5 training and prediction inputs were complete;
                                         they list files with md5 sums
  longmem_v35/READY.json, LM_SUBSET_v35.json   the same for the long-memory store, and the chosen long-memory inputs

Environment variables
  Every path below can be replaced through a variable named V35_* (for example V35_SRC, V35_CACHE,
  V35_EXPECT_JSON). This was meant for dry runs.
"""
from swotgat.paths import WORK, WORK_FS
import hashlib
import json
import os
import re
import sys
from pathlib import Path

R = Path(f"{WORK}"); C = R/"CONUS"; F = R/"CONUS_storage_fork"; E = F/"eval"
L = E/"v35_launch"; V21 = E/"v21_retrain"


def env(name, default):
    """Return the path in the environment variable `name` if it is set, else `default`."""
    return Path(os.environ[name]) if os.environ.get(name) else Path(default)


IN = env("V35_INPUTS", E/"v35_inputs")                         # graph v3.5 inputs: stores, lists, attributes (only read)
READY = env("V35_READY", IN/"READY_v35.json")
READY_PRED = env("V35_READY_PRED", IN/"READY_v35_pred.json")
LMS = env("V35_LMS", F/"features/longmem_v35")                 # long-memory input store (only read)
LM_READY = env("V35_LM_READY", LMS/"READY.json")
LM_SUBSET = env("V35_LM_SUBSET", E/"featscreen_v2/lmrank/LM_SUBSET_v35.json")
CFG_DIR = env("V35_CFG_DIR", C)                                # run.py names the run dir <cfg dir>/<cfg stem>_<date>
LCHK = env("V35_LCHK_DIR", L/"loadercheck")
STATE = env("V35_STATE_DIR", L)                                # status and approval marker files of the C1-C3 launches
RESULTS = env("V35_RESULTS", L/"results")
CACHE = env("V35_CACHE", L/"_cache")
# configuration files of the earlier configuration A180, one per seed (S1 = seed 0, S2 = seed 1)
A180 = {"S1": C/"era5_conus_v22a180_S1_storage.json", "S2": C/"era5_conus_v22a180_S2_storage_s1.json"}
SFX = {"S1": "S1_storage", "S2": "S2_storage_s1"}
TEMPLATE = {s: env(f"V35_TEMPLATE_{s}", IN/"config"/f"era5_conus_v35c1_{SFX[s]}.TEMPLATE.json") for s in SFX}
APPROVED = {"C1": STATE/"APPROVED_v35_C1.txt", "C2": STATE/"APPROVED_v35_LM.txt", "C3": STATE/"APPROVED_v35_LM.txt"}
GRAPH_MD5 = os.environ.get("V35_GRAPH_MD5", "c580577d7581bbfb975c261bea609051")
EVAL_STEP = {"C1": "step_050000", "C2": "step_050000", "C3": "step_008000"}
FT = dict(steps=8000, peak_lr=5e-5, final_lr=5e-6, warmup=100, save_every=2000, hours=11)   # fine-tuning recipe of C3 (taken from an earlier fine-tuned configuration, lmdft)
LM_MAX = 8                                                     # at most 8 long-memory inputs (they were chosen by forward selection)
TRAIN_FROM = "2022-01-01"; TRAIN_TO = "2026-09-30"             # train window 2023-01..2026-09 + 365 days of input history
H10_FROM = "2009-07-05"; X26_FROM = "2022-09-02"               # prediction inputs start 180 days before 2010-01-01 / 2023-03-01
HEX = re.compile(r"^[0-9a-f]{32}$")


def stem(arm, seed):
    """Return the name of a configuration without ".json".

    `arm` is the configuration label (C1, C2, ...), `seed` is "S1" or "S2". For example,
    stem("C5", "S1") gives "era5_conus_v35c5_S1_storage".
    """
    return f"era5_conus_v35{arm.lower()}_{SFX[seed]}"


def cfg_path(arm, seed):
    """Return the path of a configuration file in CFG_DIR."""
    return CFG_DIR/f"{stem(arm, seed)}.json"


def src():
    """Return the folder of the model source (vendor/swot_ml/src).

    The folder is read from the environment variable V35_SRC, else from the one-line file
    v35_launch/SRC.
    """
    return Path(os.environ.get("V35_SRC") or (L/"SRC").read_text().strip())


def md5(p):
    """Return the md5 of file p as a hex string, or None if the file does not exist."""
    p = Path(p)
    if not p.is_file(): return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def md5s(obj):
    """Return the md5 of a JSON-serializable object (keys sorted)."""
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def load_json(p):
    """Read a JSON file.

    Returns (object, None), or (None, error message) if it cannot be read.
    """
    try: return json.load(open(p)), None
    except Exception as e: return None, f"{p}: {type(e).__name__}: {e}"


# ---------------- tolerant readers for the hand-over files ----------------
# The hand-over (READY) files were written by several scripts in slightly different layouts, so these readers accept
# several layouts. They are not used for the paper's configuration C5.
def ready_entries(obj):
    """Return every absolute path named anywhere in a READY file, with its md5 when one is given next to it.

    Accepts {"path": p, "md5": m} records, {p: m} maps, {"<key>": p, "<key>_md5": m} pairs, "p m"
    strings and bare path strings. Returns (paths, names): paths maps path -> md5 or None; names
    maps file name -> md5 for {"name.ext": m} maps.
    """
    paths, names = {}, {}

    def put(p, m):
        p = str(p).rstrip(".,;:)")
        if p not in paths or (m and not paths[p]): paths[p] = m.lower() if m else None

    def walk(x):
        if isinstance(x, dict):
            p = next((x[k] for k in ("path", "file", "dst", "destination", "dir", "src") if isinstance(x.get(k), str) and x[k].startswith("/")), None)
            m = next((x[k] for k in ("md5", "md5sum", "md5_hex", "checksum") if isinstance(x.get(k), str) and HEX.match(x[k].lower())), None)
            if p: put(p, m)
            for k, v in x.items():
                ks = str(k)
                if ks.startswith("/") and isinstance(v, str) and HEX.match(v.lower()): put(ks, v)
                elif "/" not in ks and "." in ks and isinstance(v, str) and HEX.match(v.lower()): names[ks] = v.lower()
                elif isinstance(v, str) and v.startswith("/") and " " not in v:
                    mk = x.get(f"{ks}_md5") or x.get(f"md5_{ks}")
                    put(v, mk if isinstance(mk, str) and HEX.match(mk.lower()) else None)
                else: walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, str):
            toks = x.split(); ps = [t for t in toks if t.startswith("/")]; hs = [t for t in toks if HEX.match(t.lower())]
            if len(ps) == 1: put(ps[0], hs[0] if len(hs) == 1 else None)
            elif x.startswith("/") and not toks[1:]: put(x, None)
    walk(obj); return paths, names


def ready_flags(obj):
    """Return the top-level status flags of a READY file that say "not ready".

    A flag counts if ready, ok, passed, pass, safe_to_train, all_passed or all_checks_passed is
    false, or if a status, verdict, result or state text names fail, hold, not ready, pending,
    blocked or wip. An empty list means no such flag.
    """
    bad = []
    if not isinstance(obj, dict): return bad
    tops = [obj] + [v for k, v in obj.items() if str(k).lower() in ("status", "summary", "verdict") and isinstance(v, dict)]
    for d in tops:
        for k, v in d.items():
            kl = str(k).lower()
            if kl in ("ready", "ok", "passed", "pass", "safe_to_train", "all_passed", "all_checks_passed") and v is False: bad.append(f"{k}=false")
            if kl in ("status", "verdict", "result", "state") and isinstance(v, str) and re.search(r"fail|hold|not.?ready|pending|blocked|wip", v, re.I):
                bad.append(f"{k}={v!r}")
    return bad


def lookup(paths, names, p):
    """Look up path p in the result of ready_entries().

    Returns (listed, md5). p matches by exact path, by resolved path (symbolic links followed), or
    by file name.
    """
    p = str(p)
    for q in (p, os.path.realpath(p)):
        if q in paths: return True, paths[q]
    rp = {os.path.realpath(k): v for k, v in paths.items()}
    if os.path.realpath(p) in rp: return True, rp[os.path.realpath(p)]
    n = Path(p).name
    if n in names: return True, names[n]
    return False, None


def lm_subset(obj):
    """Return the chosen long-memory inputs named in LM_SUBSET_v35.json.

    Looks for a list of lm_* names (or of {"name": ...} records) anywhere in the object, preferring
    a key named subset, lm_subset, selected, chosen, picked, final, features, vars, inputs or
    extra_dynamic_vars. Returns [] for an explicitly empty choice and None if the file is not clear.
    """
    pref = ("subset", "lm_subset", "selected", "chosen", "picked", "final", "features", "vars", "inputs", "extra_dynamic_vars")
    cands = []

    def names_of(x):
        if all(isinstance(v, str) for v in x): return list(x)
        if all(isinstance(v, dict) for v in x): return [v.get("name") or v.get("feature") or v.get("var") for v in x]
        return None

    def walk(x, key=""):
        if isinstance(x, list):
            n = names_of(x) if x else []
            if n == [] and key.lower() in pref: cands.append((key, []))
            elif n and all(isinstance(v, str) and v.startswith("lm_") for v in n): cands.append((key, n))
            for v in x:
                if isinstance(v, (dict, list)): walk(v, key)
        elif isinstance(x, dict):
            for k, v in x.items(): walk(v, str(k))
    walk(obj)
    for p in pref:
        hit = [v for k, v in cands if k.lower() == p]
        if hit: return hit[0]
    return cands[0][1] if len(cands) == 1 else None


# ---------------- configs ----------------
# Helpers that built and checked the configurations C1, C2 and C3. A configuration is a dict loaded from its JSON file.
# C1_ALLOWED: the keys in which a graph v3.5 configuration may differ from A180's.
C1_ALLOWED = {"zarr_dir", "attributes_file", "graph_network_file", "cache_dir", "train_basin_file", "valid_basin_file", "test_basin_file",
              "target_basin_files", "target_drop_masks", "features", "log_norm_cols", "_v21_note"}
PATH_KEYS = ("zarr_dir", "attributes_file", "graph_network_file", "train_basin_file", "valid_basin_file", "test_basin_file")


def config_paths(d):
    """Return the input paths that configuration d names.

    The result is {key: path}, with keys such as "zarr_dir" or "target_drop_masks.dv_river".
    """
    out = {k: d[k] for k in PATH_KEYS if d.get(k)}
    for g in ("target_basin_files", "target_drop_masks"):
        for k, v in (d.get(g) or {}).items(): out[f"{g}.{k}"] = v
    return out


LM_KEYS = {"extra_dynamic_store", "extra_dynamic_vars", "extra_dynamic_source"}


def diff_vs_a180(d, a, lm=False):
    """Compare a graph v3.5 configuration d with A180's configuration a of the same seed against an allow-list.

    Allowed to differ: zarr_dir, attributes_file, graph_network_file, cache_dir,
    train/valid/test_basin_file, target_basin_files, target_drop_masks, features.static (exactly one
    more column, at the end), log_norm_cols (the same one more column), _v21_note and
    model_args.seed. With lm=True (C2) the three extra_dynamic_* keys may differ too and must be
    set; with lm=False (C1) they must be empty.

    Returns (problems, the new static column). An empty list of problems means the configuration
    passes.
    """
    allowed = C1_ALLOWED | {"model_args"} | (LM_KEYS if lm else set())
    probs = [f"'{k}' differs from A180 (not on the allow-list)" for k in sorted(set(d) | set(a)) if d.get(k) != a.get(k) and k not in allowed]
    ma, mb = d.get("model_args", {}), a.get("model_args", {})
    probs += [f"model_args.{k} differs from A180" for k in sorted(set(ma) | set(mb)) if k != "seed" and ma.get(k) != mb.get(k)]
    ft, fa = d.get("features", {}), a["features"]
    for k in sorted(set(ft) | set(fa)):
        if k != "static" and ft.get(k) != fa.get(k): probs.append(f"features.{k} differs from A180")
    st = ft.get("static", [])
    if st[:-1] != fa["static"] or len(st) != len(fa["static"]) + 1: probs.append("features.static is not A180's list + one new column")
    upa = st[-1] if st else None
    ln, la = d.get("log_norm_cols") or [], a.get("log_norm_cols") or []
    if ln[:-1] != la or not ln or ln[-1] != upa: probs.append("log_norm_cols is not A180's + the new static")
    if lm and not (d.get("extra_dynamic_vars") and d.get("extra_dynamic_store")): probs.append("C2 config has no long-memory inputs")
    if not lm and any(d.get(k) for k in LM_KEYS): probs.append("C1 config has long-memory inputs")
    return probs, upa


def check_template(t, a):
    """Check a C1 template t against A180's configuration a: diff_vs_a180 without long-memory inputs.

    Reports a problem if the template changes anything but the graph-dependent inputs and the one
    new static attribute. Returns what diff_vs_a180 returns.
    """
    return diff_vs_a180(t, a, lm=False)


def c1_config(seed, template):
    """Return the C1 configuration: a copy of the template with cache_dir set to CACHE.

    `seed` is not used.
    """
    d = json.loads(json.dumps(template)); d["cache_dir"] = str(CACHE); return d


def c2_config(c1, subset):
    """Return the C2 configuration.

    It is a copy of the C1 configuration c1 plus the long-memory store and the chosen inputs `subset`.
    """
    d = json.loads(json.dumps(c1))
    d.update(extra_dynamic_store=str(LMS), extra_dynamic_vars=list(subset), extra_dynamic_source="era5")
    d["_v21_note"] = (f"v3.5 C2: C1 + long-memory inputs {list(subset)} from {LMS} (LM_SUBSET_v35.json), from scratch, A180's recipe "
                      f"(seq 180, burn-in 90, history 365, 50k steps). C1 = {c1.get('_v21_note', '')}")
    return d


def c3_sets(subset):
    """Return the list of key=value settings that turn a C1 checkpoint's configuration into C3.

    The settings add the long-memory store and inputs and set the cache folder. They were passed to
    the fine-tuning script (finetune_v35.py --set; that script is not part of this repository).
    """
    return [f"extra_dynamic_store={LMS}", f"extra_dynamic_vars=[{', '.join(subset)}]", "extra_dynamic_source=era5", f"cache_dir={CACHE}"]


def ft_config(base_cfg, sets):
    """Return the fine-tuning configuration for a base configuration and a list of settings.

    Applies the recipe FT (steps, learning rates, warm-up, checkpoint interval, hours) to a copy of
    base_cfg, the configuration of the base checkpoint, then each key=value setting of `sets` in
    order; dotted keys address nested entries and values are parsed as YAML. Mirrors what the
    fine-tuning script did.
    """
    import yaml
    d = json.loads(json.dumps(base_cfg))
    d.update(max_training_steps=FT["steps"], max_training_hours=int(FT["hours"]), warmup_steps=FT["warmup"], initial_lr=FT["peak_lr"],
             final_lr=FT["final_lr"], log_every_n_steps=FT["save_every"], validate_every_n_steps=None)
    for kv in sets:
        k, v = kv.split("=", 1); node = d
        for kk in k.split(".")[:-1]: node = node.setdefault(kk, {})
        node[k.split(".")[-1]] = yaml.safe_load(v)
    return d


def config_hash(d):
    """Return the cache key (config_hash of the model source) for a configuration dict.

    The dict is written to a temporary file and loaded with the Config class of the model source
    named by src(). The function asserts that the cache module really comes from that source folder.
    """
    import tempfile
    s = str(src())
    if s not in sys.path: sys.path.insert(0, s)
    from config import Config
    import data.cached_basingraphdataset as cbd
    assert s in cbd.__file__, cbd.__file__
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(d, f)
    try: return cbd.config_hash(Config.from_file(Path(f.name)))
    finally: os.unlink(f.name)


def write_cfg(d, p):
    """Write configuration d to path p as JSON.

    Returns "unchanged", "rewritten" or "written". The function replaces an existing file with
    different content. The caller must make sure that this happens only before training with that
    configuration has started.
    """
    p = Path(p); new = json.dumps(d, indent=1)
    if p.exists() and p.read_text() == new: return "unchanged"
    state = "rewritten" if p.exists() else "written"
    p.write_text(new); return state


def latest_run(arm, seed, step=None):
    """Return the newest run folder of a configuration and seed that has checkpoints/<step>/model.eqx, or None.

    `arm` is the configuration label (C1, C2, C3), `seed` is "S1" or "S2". `step` defaults to
    EVAL_STEP[arm]. Run folders are searched as <CONUS>/<configuration name>_2* unless
    V35_RUNS_<arm>_<seed> gives another pattern.
    """
    step = step or EVAL_STEP[arm]
    pat = os.environ.get(f"V35_RUNS_{arm}_{seed}") or str(C/f"{stem(arm, seed)}_2*")
    import glob
    for d in sorted(glob.glob(pat), reverse=True):
        if (Path(d)/"checkpoints"/step/"model.eqx").is_file(): return Path(d)
    return None


# ---------------- checker B (2026-10-22): the exact input set, store content ----------------
# Checks on the inputs of a configuration: that it names exactly the expected files (path and md5), that the training
# store holds the HALF-exact lake targets (HALF = the lake outlier filter of Trudel et al.), and content digests of the
# stores. _EX lists the expected inputs of graph v3.5 as key -> (path, md5 or None).
_EX = dict(
    zarr_dir=(str(C/"zarr_store_vd2026_halfexact_v35"), None),
    attributes_file=(str(C/"attributes/static_conus_storage_fixgdw_v35_g4.parquet"), "a146e7dcaf807013ca53d34d5eee7023"),
    train_basin_file=(str(C/"basin_lists/storage/train_nodes_g_v35.csv"), "1f9f7277aff50ecd687736bd567c6e26"),
    **{"target_basin_files.dv_SWOTAE": (str(C/"basin_lists/fallback/dv_SWOTAE_vd2026_fb_g_v35.csv"), "8282d03155b69afaf0646f3911c0781d"),
       "target_drop_masks.dv_river": (str(C/"basin_lists/swotqc/drop_river_paper_min_present_t9b_v35.parquet"), "47cfebc67fc174af686de6e6fa98063a")})
EXPECT = json.load(open(os.environ["V35_EXPECT_JSON"])) if os.environ.get("V35_EXPECT_JSON") else _EX   # a JSON file can replace the expected inputs (for a dry run)
HALF_REF = env("V35_HALF_REF", C/"zarr_store_vd2026_halfexact")      # template store with the HALF-exact lake targets (the v3.5 store copies them bit for bit, by lake id)
NONHALF_REF = env("V35_NONHALF_REF", C/"zarr_store_vd2026")          # store with the lake targets before HALF's coverage rule (source of the SWOT Period prediction stores)


def exact_inputs(d):
    """Check that configuration d names exactly the expected inputs (EXPECT).

    An input is as expected if it has the same path and, where one is given, the same md5. Returns a
    list of problems. A river drop list with any other md5 is a problem.
    """
    probs, P = [], config_paths(d)
    for k, (p, m) in EXPECT.items():
        if P.get(k) != p: probs.append(f"{k} = {P.get(k)} (must be exactly {p})"); continue
        if m and md5(p) != m: probs.append(f"{k} md5 {md5(p)} (must be {m})")
    return probs


def _zopen(g):
    """Open one basin group of a store."""
    import xarray as xr
    try: return xr.open_zarr(g, consolidated=True)
    except Exception: return xr.open_zarr(g, consolidated=False)   # groups without consolidated metadata (the older stores)


def _lake(ds, lakes, dates):
    """Return dv_SWOTAE of a group as a (date, lake) array on the given lakes and dates."""
    import numpy as np
    x = ds.dv_SWOTAE.assign_coords(subbasin=ds.subbasin.values.astype(str)).reindex(subbasin=lakes, date=dates).transpose("date", "subbasin")
    return np.asarray(x.values, float)


def _same(a, b):
    """Return True if a and b have the same shape, NaN in the same places and equal values elsewhere."""
    import numpy as np
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def halfexact_check(z, n=4):
    """Check that store z holds the HALF-exact lake targets.

    The check uses the first n basins in which the template store with HALF-exact targets (HALF_REF)
    and the store with the targets before HALF's coverage rule (NONHALF_REF,
    CONUS/zarr_store_vd2026) differ in dv_SWOTAE at the lake ids all three stores share. There z
    must equal HALF_REF exactly (float equality, NaN equal to NaN). A store that holds the other
    targets fails. Fewer than n such basins is reported as a problem.

    Returns (problems, {"basins": basins checked}).
    """
    probs, done = [], []
    for g in sorted(p for p in Path(z).iterdir() if (p/"zarr.json").exists() or (p/".zmetadata").exists()):
        if len(done) >= n: break
        if not ((HALF_REF/g.name).exists() and (NONHALF_REF/g.name).exists()): continue
        a, r, q = _zopen(g), _zopen(HALF_REF/g.name), _zopen(NONHALF_REF/g.name)
        if not all("dv_SWOTAE" in x for x in (a, r, q)): continue
        lakes = sorted(set(a.subbasin.values.astype(str)) & set(r.subbasin.values.astype(str)) & set(q.subbasin.values.astype(str)))
        dates = r.date.values
        rv, qv = _lake(r, lakes, dates), _lake(q, lakes, dates)
        if _same(rv, qv): continue                         # the two reference stores agree here: this basin cannot tell them apart
        if not _same(_lake(a, lakes, dates), rv): probs.append(f"{g.name}: dv_SWOTAE is not {HALF_REF.name}'s (HALF-exact) at the shared lake ids")
        done.append(g.name)
    if len(done) < n: probs.append(f"only {len(done)} basins where {HALF_REF.name} and {NONHALF_REF.name} differ (need {n})")
    return probs, dict(basins=done)


def store_digest(z, variables=None):
    """Return one md5 over every file of a Zarr store z: relative file names, metadata and chunk bytes.

    A rewrite of the data at the same path therefore changes the digest. `variables` limits the
    digest to those arrays plus the date and subbasin coordinates and the group metadata. About 1
    minute for the 2.7 GB training store. Used by the paper's pipeline (through data_digests).
    """
    h = hashlib.md5(); z = Path(z)
    for g in sorted(p for p in z.iterdir() if p.is_dir()):
        for f in sorted(g.rglob("*")):
            if not f.is_file(): continue
            rel = f.relative_to(z)
            if variables is not None and len(rel.parts) > 2 and rel.parts[1] not in variables and rel.parts[1] not in ("date", "subbasin"): continue
            h.update(str(rel).encode()); h.update(hashlib.md5(f.read_bytes()).digest())
    return h.hexdigest()


def data_digests(d):
    """Return the content digests that a training cache built from configuration d depends on.

    {"store": digest of the training store} and, if the configuration has long-memory inputs,
    {"lm": digest of those inputs in the long-memory store}. build_cache_v35.py records them and
    train.sh compares them before every job.
    """
    out = dict(store=store_digest(d["zarr_dir"]))
    if d.get("extra_dynamic_vars"): out["lm"] = store_digest(d["extra_dynamic_store"], set(d["extra_dynamic_vars"]))
    return out


def disk_free_gb(path=f"{WORK_FS}"):
    """Return the free disk space in GB at `path` (default: the file system of the work tree)."""
    import shutil
    return shutil.disk_usage(path).free / 1e9
