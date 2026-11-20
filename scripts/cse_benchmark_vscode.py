r"""
CSE-CIC-IDS2018 accuracy check for Moro & Prati, Commun. Phys. 6, 269 (2023).
Run it in VS Code with the "Run Python File" button. No arguments needed.

WHERE TO PUT THINGS
    C:\RBM\cse_benchmark_vscode.py      <- this file
    C:\RBM\CSE\*.csv                    <- the ten files of "Processed Traffic Data for ML Algorithms"
Download (no AWS account needed), in PowerShell after installing the AWS CLI:
    aws s3 sync --no-sign-request --region ca-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" C:\RBM\CSE
Results go to C:\RBM\results\ ; the cleaned and encoded data are cached in C:\RBM\results\cse_cache\ so that
the slow CSV reading (several minutes) only happens once.

WHAT IT DOES (preprocessing as described by Moro & Prati)
  - all CSV files concatenated; identifier columns and the timestamp removed
  - rows with missing or infinite values removed; duplicate records removed
  - the 16 features Moro & Prati list as discarded are dropped
  - binary label: Benign (normal) vs any attack
  - random 80/20 train/test split; training set balanced by under-sampling the larger class
  - binary encoding with 153 feature bits + 3 label bits = 156 visible units (their CSE model size)
  Part 1  ceiling: best accuracy ANY classifier could reach with each representation
  Part 2  standard classifiers and classical RBMs (CD-k, 90 hidden units, ADAM 0.01, batch 128),
          all trained in parallel, one per logical CPU
Moro & Prati report on this dataset: classical RBM accuracy 0.800 (k=1) to 0.924 (k=1000), D-Wave RBM 0.854.
Needs: numpy, pandas, scikit-learn
"""

# =============================================================================================
# SETTINGS -- change these, then press Run
# =============================================================================================
DATA_FOLDER = "CSE"            # folder with the CSV files, relative to this script
SEED = 0                       # seed for the split, balancing and classifiers

TIME_BUDGET_HOURS = 3.75       # the whole script stops within this time: RBM runs keep training epoch after epoch
                               # and stop when the next epoch would not finish in time (None = no limit)
RBM_K_VALUES = [1, 10, 100]    # CD-k values for the classical RBM (Moro & Prati Table 1: 1, 10, 100, 1000)
RBM_SEEDS = [0, 1, 2, 3, 4]    # one RBM run per k and seed (15 runs + 16 classifiers = 31 tasks for 32 CPUs)
RBM_PASSES = 2000              # maximum epochs (as published); the time budget normally stops the runs earlier
EVAL_SUBSAMPLE = 200_000       # test records used for the progress lines (the final result uses the full test set)

CLASSIFIER_TRAIN_SIZE = 500_000   # balanced training records for the standard classifiers (subsample, for speed)
KNN_TEST_SIZE = 100_000           # k-NN is scored on a random subsample of the test set (it is very slow)
N_WORKERS = "auto"             # "auto" = one task per logical CPU, or a number
REBUILD_CACHE = False          # True = re-read the CSV files even if a cache exists

# =============================================================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time, json, glob, datetime, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CACHE = OUT / "cse_cache"
F32 = np.float32

DISCARDED = ["Fwd Seg Size Avg", "TotLen Fwd Pkts", "Bwd IAT Tot", "Bwd Pkt Len Max", "Subflow Fwd Byts",
             "Pkt Size Avg", "Bwd Seg Size Avg", "Subflow Fwd Pkts", "Bwd IAT Max", "Subflow Bwd Pkts",
             "Flow Pkts/s", "Flow IAT Mean", "Idle Max", "Active Min", "Active Max", "Pkt Len Min"]
ID_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
PROTO, PORT = "Protocol", "Dst Port"
N_FEATURE_BITS = 153


def log(msg="", file=None):
    print(msg, flush=True)
    if file is not None:
        with open(file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def save_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def hms(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


# =============================================================================================
# 1. Read, clean, split and encode (done once, then cached)
# =============================================================================================
def read_one(f):
    """Read and clean one CSV file (runs in its own process). Uses the multi-threaded pyarrow reader if
    it is installed, otherwise pandas' standard reader in chunks."""
    def clean(chunk):
        chunk = chunk.drop(columns=[c for c in ID_COLS if c in chunk.columns])
        feats = [c for c in chunk.columns if c != "Label" and c not in DISCARDED]
        lab = chunk["Label"].astype(str).str.strip()
        X = chunk[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(X).all(1) & (lab != "Label").to_numpy()      # drops repeated headers, NaN, inf
        missing = [c for c in DISCARDED if c not in chunk.columns]
        # X stays float64 until duplicates are removed (float32 would merge records that differ only in fine decimals)
        return X[ok], (lab[ok].str.lower() == "benign").to_numpy().astype(np.int8), len(chunk), feats, missing
    t0 = time.perf_counter()
    try:
        df = pd.read_csv(f, engine="pyarrow")
        X, y, n, feats, missing = clean(df); del df
        engine = "pyarrow"
    except Exception:
        parts = [clean(c) for c in pd.read_csv(f, chunksize=1_000_000, low_memory=False)]
        X = np.concatenate([q[0] for q in parts]); y = np.concatenate([q[1] for q in parts])
        n = sum(q[2] for q in parts); feats, missing = parts[0][3], parts[0][4]
        engine = "pandas"
    return {"name": Path(f).name, "X": X, "y": y, "n": n, "feats": feats, "missing": missing,
            "seconds": time.perf_counter() - t0, "engine": engine}


def build_cache(logfile):
    files = sorted(glob.glob(str(HERE / DATA_FOLDER / "*.csv")))
    if not files:
        raise SystemExit(f"ERROR: no CSV files in {HERE / DATA_FOLDER}. Download them first (see the top of this file).")
    t0 = time.perf_counter()
    log(f"   reading {len(files)} files ({sum(os.path.getsize(f) for f in files) / 1e9:.2f} GB) in parallel, "
        f"one process per file; a line is printed as each file finishes", logfile)
    from multiprocessing import Pool
    with Pool(min(len(files), os.cpu_count() or 1)) as pool:
        got = {}
        for r in pool.imap_unordered(read_one, files):
            got[r["name"]] = r
            log(f"      {r['name']}: {r['n']:,} rows, {len(r['y']):,} kept  ({r['engine']} reader, {r['seconds']:.0f} s)   "
                f"{len(got)}/{len(files)} files done, elapsed {hms(time.perf_counter() - t0)}", logfile)
    order = [got[Path(f).name] for f in files]
    feats = order[0]["feats"]
    for r in order:
        if r["feats"] != feats:
            raise SystemExit(f"ERROR: {r['name']} has different feature columns from {order[0]['name']}")
    if order[0]["missing"]:
        log(f"   note: discarded features not found under these names: {order[0]['missing']}", logfile)
    n_raw = sum(r["n"] for r in order)
    X_parts = [r["X"] for r in order]; y_parts = [r["y"] for r in order]; del got, order
    X = np.concatenate(X_parts); y = np.concatenate(y_parts); del X_parts, y_parts
    log(f"   after removing missing/infinite values: {len(y):,} of {n_raw:,} rows", logfile)
    # duplicates: hash each row (features + label) and keep the first occurrence
    h = pd.util.hash_pandas_object(pd.DataFrame(X).assign(_y=y), index=False).to_numpy()
    keep = ~pd.Series(h).duplicated().to_numpy()
    X, y = X[keep].astype(F32), y[keep]   # duplicates found at full (float64) precision, then stored as float32
    log(f"   after removing duplicates: {len(y):,} records, {len(feats)} features, normal fraction {y.mean():.3f}"
        f"   (Moro & Prati report M = 3,040,074 processed records)", logfile)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); n_te = int(round(0.2 * len(y)))
    te_idx, tr_idx = np.sort(perm[:n_te]), np.sort(perm[n_te:])
    Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]; del X, y

    # 153-bit encoding: destination port 16 bits, protocol one-hot (3), the rest binary-coded quantile-bin indices
    num = [c for c in feats if c not in (PROTO, PORT)]
    nuniq = np.array([len(np.unique(Xtr[:200_000, feats.index(c)])) for c in num])
    budget = N_FEATURE_BITS - 3 - 16
    nbits = {c: 1 for c in num}; order = [num[i] for i in np.argsort(-nuniq)]
    while sum(nbits.values()) < budget:
        for c in order:
            if sum(nbits.values()) >= budget:
                break
            nbits[c] += 1
    sample = Xtr[rng.choice(len(Xtr), min(len(Xtr), 1_000_000), replace=False)]
    edges = {c: np.unique(np.quantile(sample[:, feats.index(c)], np.linspace(0, 1, 2 ** nbits[c] + 1)[1:-1])) for c in num}
    protos = [0.0, 6.0, 17.0]

    def encode(X):
        out = np.empty((len(X), N_FEATURE_BITS), np.uint8); j = 0
        for c in num:
            idx = np.digitize(X[:, feats.index(c)], edges[c])
            for b in range(nbits[c] - 1, -1, -1):
                out[:, j] = (idx >> b) & 1; j += 1
        for p in protos:
            out[:, j] = X[:, feats.index(PROTO)] == p; j += 1
        port = np.clip(X[:, feats.index(PORT)], 0, 65535).astype(np.int64)
        for b in range(15, -1, -1):
            out[:, j] = (port >> b) & 1; j += 1
        return out

    log("   encoding ...", logfile)
    Btr, Bte = encode(Xtr), encode(Xte)
    # raw representation for the standard classifiers: log-scaled and standardised
    def rawify(X):
        Z = np.sign(X) * np.log1p(np.abs(X)); return Z
    Ztr = rawify(Xtr); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "ytr.npy", ytr); np.save(CACHE / "yte.npy", yte)
    np.save(CACHE / "Btr.npy", Btr); np.save(CACHE / "Bte.npy", Bte)
    np.save(CACHE / "Rtr.npy", ((Ztr - mu) / sd).astype(F32)); np.save(CACHE / "Rte.npy", ((rawify(Xte) - mu) / sd).astype(F32))
    info = {"n_raw_rows": int(n_raw), "n_records": int(len(ytr) + len(yte)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "n_features": len(feats), "features": feats, "normal_fraction": float((ytr.sum() + yte.sum()) / (len(ytr) + len(yte))),
            "bits_per_feature": nbits, "seed": SEED, "files": [Path(f).name for f in files]}
    save_json(info, CACHE / "info.json")
    log(f"   cache written to {CACHE}  ({hms(time.perf_counter() - t0)})", logfile)


# =============================================================================================
# 2. Parallel tasks
# =============================================================================================
_D = {}

def init_worker():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for k in ("ytr", "yte", "Btr", "Bte", "Rtr", "Rte"):
        _D[k] = np.load(CACHE / f"{k}.npy", mmap_mode="r")          # shared through the OS cache, not copied


def balanced_idx(y, rng, cap=None):
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(i0), len(i1), (cap // 2) if cap else 10 ** 12)
    idx = np.concatenate([rng.choice(i0, n, replace=False), rng.choice(i1, n, replace=False)])
    return np.sort(idx)


def metrics(y, p):
    c = confusion_matrix(y, p, labels=[0, 1]).astype(float); col = c.sum(0)
    cm = np.divide(c, col, out=np.zeros_like(c), where=col > 0)
    return {"acc": float(accuracy_score(y, p)), "f1": float(f1_score(y, p, pos_label=0, zero_division=0)),
            "TP": float(cm[0, 0]), "FP": float(cm[1, 0]), "FN": float(cm[0, 1]), "TN": float(cm[1, 1])}


def make_model(name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    return {"Logistic regression": lambda: LogisticRegression(max_iter=2000),
            "Naive Bayes (binary)": lambda: BernoulliNB(), "Naive Bayes (Gaussian)": lambda: GaussianNB(),
            "k-NN (k=5)": lambda: KNeighborsClassifier(5),
            "Decision tree": lambda: DecisionTreeClassifier(random_state=SEED),
            "Random forest": lambda: RandomForestClassifier(100, random_state=SEED, n_jobs=1),
            "Extra trees": lambda: ExtraTreesClassifier(100, random_state=SEED, n_jobs=1),
            "Gradient boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
            "MLP (100-50)": lambda: MLPClassifier((100, 50), max_iter=50, early_stopping=True, random_state=SEED)}[name]()


def task_classifier(t):
    rep, name = t["rep"], t["model"]
    Xtr_all, Xte = (_D["Rtr"], _D["Rte"]) if rep == "raw" else (_D["Btr"], _D["Bte"])
    ytr_all, yte = _D["ytr"], _D["yte"]
    idx = balanced_idx(np.asarray(ytr_all), np.random.default_rng(SEED + 1), CLASSIFIER_TRAIN_SIZE)
    Xt, yt = np.asarray(Xtr_all[idx]), np.asarray(ytr_all[idx])
    m = make_model(name); t0 = time.perf_counter(); m.fit(Xt, yt)
    if name.startswith("k-NN"):
        sel = np.sort(np.random.default_rng(SEED + 2).choice(len(yte), min(KNN_TEST_SIZE, len(yte)), replace=False))
        Xe, ye = np.asarray(Xte[sel]), np.asarray(yte[sel])
    else:
        Xe, ye = Xte, np.asarray(yte)
    p = np.concatenate([m.predict(np.asarray(Xe[i:i + 500_000])) for i in range(0, len(ye), 500_000)])
    r = metrics(ye, p)
    r.update({"kind": "classifier", "rep": rep, "model": name, "n_train": int(len(yt)), "n_test": int(len(ye)),
              "seconds": time.perf_counter() - t0})
    return r


class RBM:
    """CD-k with ADAM; identical algorithm to the NSL-KDD baseline script (preallocated, in-place)."""
    def __init__(self, n_vis, n_hid, lr, rng, batch=128):
        self.nv, self.nh, self.lr, self.rng = n_vis, n_hid, lr, rng
        n = n_hid * n_vis + n_vis + n_hid
        self.theta = np.zeros(n, F32)
        self.W = self.theta[:n_hid * n_vis].reshape(n_hid, n_vis)
        self.a = self.theta[n_hid * n_vis:n_hid * n_vis + n_vis]; self.b = self.theta[n_hid * n_vis + n_vis:]
        self.W[...] = rng.normal(0, 0.01, (n_hid, n_vis))
        self.grad = np.zeros(n, F32)
        self.gW = self.grad[:n_hid * n_vis].reshape(n_hid, n_vis)
        self.ga = self.grad[n_hid * n_vis:n_hid * n_vis + n_vis]; self.gb = self.grad[n_hid * n_vis + n_vis:]
        self.m = np.zeros(n, F32); self.v = np.zeros(n, F32); self.tmp = np.zeros(n, F32); self.t = 0
        self._alloc(batch)

    def _alloc(self, B):
        self.B = B
        self.ph, self.phk, self.h, self.rh = (np.empty((B, self.nh), F32) for _ in range(4))
        self.vk, self.pv, self.rv = (np.empty((B, self.nv), F32) for _ in range(3))

    @staticmethod
    def _sig(x):
        np.negative(x, out=x); np.exp(x, out=x); x += 1.0; np.reciprocal(x, out=x)

    def update(self, v, k):
        if v.shape[0] != self.B:
            self._alloc(v.shape[0])
        W, a, b, rng = self.W, self.a, self.b, self.rng
        with np.errstate(over="ignore"):
            np.matmul(v, W.T, out=self.ph); self.ph += b; self._sig(self.ph)
            p = self.ph
            for _ in range(k):
                rng.random(out=self.rh, dtype=F32); np.less(self.rh, p, out=self.h)
                np.matmul(self.h, W, out=self.pv); self.pv += a; self._sig(self.pv)
                rng.random(out=self.rv, dtype=F32); np.less(self.rv, self.pv, out=self.vk)
                np.matmul(self.vk, W.T, out=self.phk); self.phk += b; self._sig(self.phk); p = self.phk
        inv = 1.0 / v.shape[0]
        np.matmul(self.ph.T, v, out=self.gW); self.gW -= self.phk.T @ self.vk; self.gW *= inv
        np.subtract(v.sum(0), self.vk.sum(0), out=self.ga); self.ga *= inv
        np.subtract(self.ph.sum(0), self.phk.sum(0), out=self.gb); self.gb *= inv
        self.t += 1
        self.m *= 0.9; self.m += 0.1 * self.grad
        np.multiply(self.grad, self.grad, out=self.tmp); self.tmp *= 0.001; self.v *= 0.999; self.v += self.tmp
        c1, c2 = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        np.divide(self.v, c2, out=self.tmp); np.sqrt(self.tmp, out=self.tmp); self.tmp += 1e-8
        np.divide(self.m, self.tmp, out=self.tmp); self.tmp *= self.lr / c1
        self.theta += self.tmp

    def predict(self, B):
        """Free-energy comparison of label bits 111 (normal) and 000 (attack), in chunks."""
        out = []
        Wf, Wl = self.W[:, :-3], self.W[:, -3:]
        for i in range(0, len(B), 200_000):
            X = np.asarray(B[i:i + 200_000], dtype=F32)
            pre = X @ Wf.T + self.b
            f1 = -(X @ self.a[:-3]) - self.a[-3:].sum() - np.logaddexp(0, pre + Wl.sum(1)).sum(1)
            f0 = -(X @ self.a[:-3]) - np.logaddexp(0, pre).sum(1)
            out.append((f1 < f0).astype(np.int8))
        return np.concatenate(out)


def task_rbm(t):
    k, seed = t["k"], t["seed"]
    tag = f"[RBM CD-{k:<3} seed {seed}]"
    rng = np.random.default_rng(1000 + seed)
    ytr = np.asarray(_D["ytr"]); yte = np.asarray(_D["yte"])
    idx = balanced_idx(ytr, rng)
    Xb = np.asarray(_D["Btr"][idx]); yb = ytr[idx]                       # balanced set in RAM (uint8)
    ev = np.sort(rng.choice(len(yte), min(EVAL_SUBSAMPLE, len(yte)), replace=False))
    Bev, yev = np.asarray(_D["Bte"][ev]), yte[ev]
    model = RBM(Xb.shape[1] + 3, 90, 0.01, rng)
    n_up = int(np.ceil(len(yb) / 128)); curve = {}; t0 = time.perf_counter(); u = 0
    deadline = t.get("deadline")
    print(f"{tag} started: {len(yb):,} balanced training records, {n_up:,} updates per pass, up to {RBM_PASSES} passes"
          + (f", stopping before {datetime.datetime.fromtimestamp(deadline):%H:%M}" if deadline else ""), flush=True)
    for ep in range(1, RBM_PASSES + 1):
        if deadline and ep > 1:
            per_epoch = (time.perf_counter() - t0) / (ep - 1)
            reserve = 0.15 * per_epoch + 120                     # time for the final full-test evaluation
            if time.time() + per_epoch + reserve > deadline:
                print(f"{tag} time budget reached: stopping after {ep - 1} epochs", flush=True)
                break
        perm = rng.permutation(len(yb))
        for i in range(0, len(yb), 128):
            sel = perm[i:i + 128]
            v = np.empty((len(sel), Xb.shape[1] + 3), F32)
            v[:, :-3] = Xb[sel]; v[:, -3:] = yb[sel, None]
            model.update(v, k); u += 1
        r = metrics(yev, model.predict(Bev)); curve[ep] = r
        el = time.perf_counter() - t0
        print(f"{tag} pass {ep:>3}/{RBM_PASSES}  acc {r['acc']:.4f}  F1 {r['f1']:.4f}  (on {len(yev):,} test records)  "
              f"elapsed {hms(el)}" + (f"  (budget ends {datetime.datetime.fromtimestamp(deadline):%H:%M})" if deadline
              else f"  left {hms(el / ep * (RBM_PASSES - ep))}"), flush=True)
        save_json({"k": k, "seed": seed, "curve": curve, "passes_done": ep}, OUT / f"cse_rbm_cd{k}_seed{seed}.json")
    final = metrics(yte, model.predict(_D["Bte"]))                     # full test set
    final.update({"kind": "rbm", "k": k, "seed": seed, "curve": curve, "epochs_done": len(curve),
                  "n_train": int(len(yb)), "n_test": int(len(yte)),
                  "ms_per_update": 1e3 * (time.perf_counter() - t0) / u, "seconds": time.perf_counter() - t0})
    print(f"{tag} FINISHED after {len(curve)} epochs: accuracy on the full test set {final['acc']:.4f}, F1 {final['f1']:.4f}", flush=True)
    return final


def run_task(t):
    return task_rbm(t) if t["kind"] == "rbm" else task_classifier(t)


def ceiling(B, y):
    h = pd.util.hash_pandas_object(pd.DataFrame(np.asarray(B)), index=False).to_numpy()
    g = pd.DataFrame({"h": h, "y": np.asarray(y)}).groupby("h")["y"].agg(["sum", "count"])
    return float(np.maximum(g["sum"], g["count"] - g["sum"]).sum() / len(y)), int(len(g))


# =============================================================================================
def main():
    OUT.mkdir(exist_ok=True)
    logfile = OUT / f"cse_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    deadline = time.time() + 3600 * TIME_BUDGET_HOURS if TIME_BUDGET_HOURS else None
    log(f"CSE-CIC-IDS2018 benchmark. Started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
        + (f"; time budget {TIME_BUDGET_HOURS} h, RBM runs stop before {datetime.datetime.fromtimestamp(deadline):%H:%M}" if deadline else ""), logfile)
    if REBUILD_CACHE or not (CACHE / "info.json").exists():
        log("=" * 100, logfile); log("PREPROCESSING (only the first time; the result is cached)", logfile)
        build_cache(logfile)
    else:
        log(f"using cached data in {CACHE} (set REBUILD_CACHE = True to re-read the CSV files)", logfile)
    with open(CACHE / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    yte = np.load(CACHE / "yte.npy"); ytr = np.load(CACHE / "ytr.npy")
    log(f"{info['n_records']:,} records after cleaning ({info['n_raw_rows']:,} rows in the files); "
        f"train {info['n_train']:,} / test {info['n_test']:,}; {info['n_features']} features; "
        f"normal fraction {info['normal_fraction']:.3f}", logfile)
    log(f"balanced training set: {2 * min(int((ytr == 0).sum()), int((ytr == 1).sum())):,} records", logfile)
    log("Moro & Prati report: classical RBM 0.800 (k=1), 0.897 (k=10), 0.915 (k=100), 0.924 (k=1000); D-Wave RBM 0.854", logfile)

    log("=" * 100, logfile)
    log("PART 1. CEILING: best accuracy any classifier using this representation could reach on this test set", logfile)
    res = {"info": info, "ceiling": {}, "results": []}
    for rep, key in (("raw", "Rte"), (f"{N_FEATURE_BITS}-bit", "Bte")):
        c, n = ceiling(np.load(CACHE / f"{key}.npy", mmap_mode="r"), yte)
        res["ceiling"][rep] = c
        log(f"   {rep:<10} ceiling {c:.4f}   ({n:,} distinct test records)", logfile)
    save_json(res, OUT / "cse_benchmark.json")

    tasks = [{"kind": "rbm", "k": k, "seed": s, "deadline": deadline} for k in sorted(RBM_K_VALUES, reverse=True) for s in RBM_SEEDS]
    for rep in ("raw", "bits"):
        for m in ["k-NN (k=5)", "MLP (100-50)", "Random forest", "Extra trees", "Gradient boosting",
                  "Logistic regression", "Decision tree", "Naive Bayes (Gaussian)" if rep == "raw" else "Naive Bayes (binary)"]:
            tasks.append({"kind": "classifier", "rep": rep, "model": m})
    n_cpu = os.cpu_count() or 1
    workers = n_cpu if N_WORKERS == "auto" else int(N_WORKERS)
    workers = max(1, min(workers, len(tasks)))
    log("=" * 100, logfile)
    log(f"PART 2. {len(tasks)} tasks ({len(RBM_K_VALUES) * len(RBM_SEEDS)} RBM runs, the rest standard classifiers) "
        f"on {workers} parallel workers ({n_cpu} logical CPUs)", logfile)
    log("RBM runs print a line after every pass; classifiers print one line when they finish.", logfile)
    log("-" * 100, logfile)
    t0 = time.perf_counter()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=init_worker) as pool:
            for i, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), 1):
                res["results"].append(r); save_json(res, OUT / "cse_benchmark.json")
                if r["kind"] == "classifier":
                    log(f"   [classifier] {r['rep']:<5} {r['model']:<24} acc {r['acc']:.4f}  F1 {r['f1']:.4f}  "
                        f"(train {r['n_train']:,}, test {r['n_test']:,}, {r['seconds']:.0f} s)   {i}/{len(tasks)} done, "
                        f"elapsed {hms(time.perf_counter() - t0)}", logfile)
                else:
                    log(f"   [RBM] CD-{r['k']} seed {r['seed']} finished: full-test accuracy {r['acc']:.4f}, F1 {r['f1']:.4f}   "
                        f"{i}/{len(tasks)} done, elapsed {hms(time.perf_counter() - t0)}", logfile)
    else:
        init_worker()
        for i, t in enumerate(tasks, 1):
            res["results"].append(run_task(t)); save_json(res, OUT / "cse_benchmark.json")

    log("=" * 100, logfile)
    log("SUMMARY (accuracy on the CSE-CIC-IDS2018 test set)", logfile)
    log("Moro & Prati: classical RBM 0.800-0.924, D-Wave RBM 0.854", logfile)
    for k in RBM_K_VALUES:
        rr = [r for r in res["results"] if r["kind"] == "rbm" and r["k"] == k]
        if rr:
            a = [r["acc"] for r in rr]; f = [r["f1"] for r in rr]
            best = max(max(c["acc"] for c in r["curve"].values()) for r in rr)
            log(f"   classical RBM CD-{k}, {min(r['epochs_done'] for r in rr)}-{max(r['epochs_done'] for r in rr)} epochs, {len(rr)} seeds: accuracy {np.mean(a):.4f} ± "
                f"{np.std(a, ddof=1) if len(a) > 1 else 0:.4f}, F1 {np.mean(f):.4f}; best at any pass {best:.4f}", logfile)
    cl = [r for r in res["results"] if r["kind"] == "classifier"]
    if cl:
        A = pd.DataFrame(cl)
        log(f"   standard classifiers ({len(A)}): min {A['acc'].min():.4f}, median {A['acc'].median():.4f}, max {A['acc'].max():.4f}", logfile)
        for _, r in A.sort_values("acc", ascending=False).iterrows():
            log(f"      {r['rep']:<5} {r['model']:<24} {r['acc']:.4f}", logfile)
    log(f"   finished in {hms(time.perf_counter() - t0)}. Saved: results/cse_benchmark.json and {logfile.name}", logfile)


if __name__ == "__main__":
    main()
