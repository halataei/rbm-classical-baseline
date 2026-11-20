r"""
Classical RBM baseline for Moro & Prati, Commun. Phys. 6, 269 (2023), NSL-KDD.
Run it in VS Code with the "Run Python File" button (the triangle at the top right). No arguments needed.

WHERE TO PUT THINGS
    C:\RBM\mp_baseline_vscode.py        <- this file
    C:\RBM\Dataset\KDDTrain+.txt        <- official training file
    C:\RBM\Dataset\KDDTest+.txt         <- official test file
Results are written to C:\RBM\results\ (a log file, one JSON file per run, and a summary).

WHAT IT DOES
  Part 1  Times the negative phase (k Gibbs steps) for the two model sizes of Moro & Prati,
          for comparison with their Table 3.
  Part 2  Trains the RBM exactly as far as the paper specifies it:
          official KDDTrain+ / KDDTest+, training set balanced by under-sampling,
          85 visible units (82 feature bits + 3 label bits), 30 hidden units,
          batch size 128, ADAM with learning rate 0.01, CD-k, 2000 passes over the data.
          Several runs (seeds, k values) are trained in parallel, one per CPU core.
          Progress is printed every few seconds.
Needs: numpy, pandas, scikit-learn
"""

# =============================================================================================
# SETTINGS -- change these, then press Run
# =============================================================================================
TRAIN_FILE = "Dataset/KDDTrain+.txt"   # relative to the folder of this script
TEST_FILE = "Dataset/KDDTest+.txt"

RUN_TIMING = False         # Part 1 (about 1-2 minutes). Already measured twice; set True to repeat.
RUN_TRAINING = True        # Part 2

# Every combination of BINS_LIST x K_VALUES x SEEDS is one training run. Runs execute in parallel, one per
# logical CPU, so to use the whole processor give at least as many runs as you have logical CPUs (32 on an
# i9-13900K). One run cannot use more than one core: each update depends on the previous one.
BINS_LIST = ["quantile", "width"]          # how real-valued features are binned (not given in the paper)
K_VALUES = [100]                           # CD-k values (Moro & Prati Table 1: 1, 10, 100, 1000)
SEEDS = list(range(15))                    # 2 x 1 x 15 = 30 runs
PASSES = 2000              # full passes (epochs) over the balanced training set (Moro & Prati: 2000)
PRINT_EVERY = 10           # evaluate on the test set and print a progress line every this many passes
VOTE_STEPS = 10            # Gibbs steps for the sampling read-out of the label at evaluation time
HIDDEN = 30                # hidden units (Moro & Prati: 30 for NSL-KDD)
LEARNING_RATE = 0.01       # ADAM learning rate (Moro & Prati: 0.01)
BATCH_SIZE = 128           # Moro & Prati: 128 for classical training
N_WORKERS = "auto"         # "auto" = one run per logical CPU; or a number, e.g. 8. 1 = no parallelism.

# =============================================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one thread per run: parallel runs do not compete
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, json, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
F32 = np.float32
UPDATE_MARKS = [47, 161, 689, 1000, 2000]    # "epoch" counts quoted by Moro & Prati, read as single updates
REPORT_PASSES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]


def log(msg="", file=None):
    """Print immediately and append to the log file."""
    print(msg, flush=True)
    if file is not None:
        with open(file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def save_json(obj, path):
    """Write JSON safely: to a temporary file first, then replace, and always close the file."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def hms(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


# =============================================================================================
# Data: official files, 82 feature bits, balanced training set
# =============================================================================================
COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "class", "difficulty"]
ONEHOT = ["protocol_type", "flag"]                                  # one-hot: 3 + 11 bits
FLAGS = ["land", "logged_in", "is_host_login", "is_guest_login"]    # already 0/1: 4 bits


def load_data(bins):
    tr = pd.read_csv(HERE / TRAIN_FILE, header=None, names=COLS).drop(columns=["num_outbound_cmds"])
    te = pd.read_csv(HERE / TEST_FILE, header=None, names=COLS).drop(columns=["num_outbound_cmds"])
    num = [c for c in tr.columns if c not in ONEHOT + FLAGS + ["service", "class", "difficulty"]]
    # Real-valued features -> binary-coded histogram-bin index (as described by Moro & Prati).
    # 24 features with most distinct values: 4 bins (2 bits); remaining 9: 2 bins (1 bit). 48 + 9 = 57 bits.
    order = tr[num].nunique().sort_values(ascending=False).index.tolist()
    nbits = {c: (2 if i < 24 else 1) for i, c in enumerate(order)}
    edges = {}
    for c in num:
        x, nb = tr[c].to_numpy(float), 2 ** nbits[c]
        edges[c] = (np.histogram_bin_edges(x, bins=nb)[1:-1] if bins == "width"
                    else np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])))
    levels = {c: sorted(tr[c].unique()) for c in ONEHOT}
    service_code = {s: i + 1 for i, s in enumerate(sorted(tr["service"].unique()))}   # 70 services -> 7 bits

    def encode(d):
        X = []
        for c in num:
            idx = np.digitize(d[c].to_numpy(float), edges[c])
            X.append((idx[:, None] >> np.arange(nbits[c])[::-1]) & 1)
        X.append(d[FLAGS].to_numpy())
        for c in ONEHOT:
            X.append(np.stack([(d[c] == l).to_numpy() for l in levels[c]], 1))
        code = d["service"].map(service_code).fillna(0).astype(int).to_numpy()
        X.append((code[:, None] >> np.arange(7)[::-1]) & 1)
        return np.hstack(X).astype(F32), (d["class"] == "normal").to_numpy().astype(np.int8)

    Xtr, ytr = encode(tr)
    Xte, yte = encode(te)
    info = {"n_train_file": len(tr), "n_test": len(te), "feature_bits": Xtr.shape[1],
            "attack_types_train": int(tr["class"].nunique() - 1), "attack_types_test": int(te["class"].nunique() - 1),
            "test_unseen_attack_fraction": float((~te["class"].isin(tr["class"].unique())).mean())}
    return Xtr, ytr, Xte, yte, info


def balance(y, rng):
    """Indices of a balanced subset: under-sample the larger class."""
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(i0), len(i1))
    idx = np.concatenate([rng.choice(i0, n, replace=False), rng.choice(i1, n, replace=False)])
    rng.shuffle(idx)
    return idx


# =============================================================================================
# RBM with 3 label bits, CD-k, ADAM. All parameters live in one flat array, so the ADAM step is
# a handful of vectorised operations instead of one set per parameter (faster for small batches).
# =============================================================================================
class RBM:
    def __init__(self, n_vis, n_hid, lr, rng, batch=BATCH_SIZE):
        self.nv, self.nh, self.lr, self.rng = n_vis, n_hid, lr, rng
        n = n_hid * n_vis + n_vis + n_hid
        self.theta = np.zeros(n, F32)
        self.W = self.theta[:n_hid * n_vis].reshape(n_hid, n_vis)        # views into theta
        self.a = self.theta[n_hid * n_vis:n_hid * n_vis + n_vis]
        self.b = self.theta[n_hid * n_vis + n_vis:]
        self.W[...] = rng.normal(0, 0.01, (n_hid, n_vis))
        self.grad = np.zeros(n, F32)
        self.gW = self.grad[:n_hid * n_vis].reshape(n_hid, n_vis)
        self.ga = self.grad[n_hid * n_vis:n_hid * n_vis + n_vis]
        self.gb = self.grad[n_hid * n_vis + n_vis:]
        self.m = np.zeros(n, F32); self.v = np.zeros(n, F32); self.tmp = np.zeros(n, F32)
        self.t = 0
        self._alloc(batch)

    def _alloc(self, B):
        """Preallocated work arrays for a batch of B records (avoids allocating memory at every step)."""
        self.B = B
        self.ph, self.phk, self.h, self.rh = (np.empty((B, self.nh), F32) for _ in range(4))
        self.vk, self.pv, self.rv = (np.empty((B, self.nv), F32) for _ in range(3))

    @staticmethod
    def _sig(x):
        """In-place logistic function."""
        np.negative(x, out=x); np.exp(x, out=x); x += 1.0; np.reciprocal(x, out=x)

    def gibbs(self, v, k):
        """k alternating Gibbs steps from visible states v (used for timing). Returns final v and P(h=1|v)."""
        if v.shape[0] != self.B:
            self._alloc(v.shape[0])
        W, a, b, rng = self.W, self.a, self.b, self.rng
        with np.errstate(over="ignore"):
            np.matmul(v, W.T, out=self.phk); self.phk += b; self._sig(self.phk)
            for _ in range(k):
                rng.random(out=self.rh, dtype=F32); np.less(self.rh, self.phk, out=self.h)
                np.matmul(self.h, W, out=self.pv); self.pv += a; self._sig(self.pv)
                rng.random(out=self.rv, dtype=F32); np.less(self.rv, self.pv, out=self.vk)
                np.matmul(self.vk, W.T, out=self.phk); self.phk += b; self._sig(self.phk)
        return self.vk, self.phk

    def update(self, v, k):
        if v.shape[0] != self.B:
            self._alloc(v.shape[0])
        W, a, b, rng = self.W, self.a, self.b, self.rng
        with np.errstate(over="ignore"):
            np.matmul(v, W.T, out=self.ph); self.ph += b; self._sig(self.ph)          # positive phase
            p = self.ph                                                             # negative phase: CD-k from the data
            for _ in range(k):
                rng.random(out=self.rh, dtype=F32); np.less(self.rh, p, out=self.h)
                np.matmul(self.h, W, out=self.pv); self.pv += a; self._sig(self.pv)
                rng.random(out=self.rv, dtype=F32); np.less(self.rv, self.pv, out=self.vk)
                np.matmul(self.vk, W.T, out=self.phk); self.phk += b; self._sig(self.phk); p = self.phk
        inv = 1.0 / v.shape[0]
        np.matmul(self.ph.T, v, out=self.gW); self.gW -= self.phk.T @ self.vk; self.gW *= inv
        np.subtract(v.sum(0), self.vk.sum(0), out=self.ga); self.ga *= inv
        np.subtract(self.ph.sum(0), self.phk.sum(0), out=self.gb); self.gb *= inv
        self.t += 1                                                                 # ADAM (0.9, 0.999, 1e-8)
        self.m *= 0.9; self.m += 0.1 * self.grad
        np.multiply(self.grad, self.grad, out=self.tmp); self.tmp *= 0.001; self.v *= 0.999; self.v += self.tmp
        c1, c2 = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        np.divide(self.v, c2, out=self.tmp); np.sqrt(self.tmp, out=self.tmp); self.tmp += 1e-8
        np.divide(self.m, self.tmp, out=self.tmp); self.tmp *= self.lr / c1
        self.theta += self.tmp

    def free_energy(self, v):
        return -(v @ self.a) - np.logaddexp(0, v @ self.W.T + self.b).sum(1)

    def predict_free_energy(self, X):
        """Exact read-out: compare free energies with label bits 111 (normal) and 000 (attack)."""
        z = np.zeros((len(X), 3), F32)
        return (self.free_energy(np.hstack([X, z + 1])) < self.free_energy(np.hstack([X, z]))).astype(np.int8)

    def predict_gibbs_vote(self, X, k):
        """Sampling read-out as in the paper: Gibbs-sample the 3 label bits with the features clamped,
        then majority vote."""
        W, a, b, rng = self.W, self.a, self.b, self.rng
        lab = (rng.random((len(X), 3), dtype=F32) < 0.5).astype(F32)
        pre = X @ W[:, :-3].T + b                           # feature contribution, fixed
        for _ in range(k):
            ph = sigmoid(pre + lab @ W[:, -3:].T)
            h = (rng.random(ph.shape, dtype=F32) < ph).astype(F32)
            pl = sigmoid(h @ W[:, -3:] + a[-3:])
            lab = (rng.random(pl.shape, dtype=F32) < pl).astype(F32)
        return (lab.sum(1) >= 2).astype(np.int8)


def metrics(y, p):
    """accuracy, F1 for the attack class, and the column-normalised confusion matrix used by Moro & Prati."""
    c = confusion_matrix(y, p, labels=[0, 1]).astype(float)          # rows: true class, columns: predicted
    col = c.sum(0)
    cm = np.divide(c, col, out=np.zeros_like(c), where=col > 0)      # column-normalised, no divide-by-zero
    return {"acc": float(accuracy_score(y, p)), "f1": float(f1_score(y, p, pos_label=0, zero_division=0)),
            "TP": float(cm[0, 0]), "FP": float(cm[1, 0]), "FN": float(cm[0, 1]), "TN": float(cm[1, 1])}


# =============================================================================================
# Part 1: negative-phase timing
# =============================================================================================
def run_timing(logfile):
    log("=" * 100, logfile)
    log("PART 1. TIME TO EVALUATE THE NEGATIVE PHASE (k Gibbs steps, one process, one thread, float32)", logfile)
    log("Moro & Prati, Table 3:  NSL-KDD model (85+30), k=10:  70 ms single core | 35 ms 128 threads | D-Wave 16 ms", logfile)
    log("                        CSE model (156+90),   k=100: 1100 ms single core | 710 ms 128 threads | D-Wave 17 ms", logfile)
    log("-" * 100, logfile)
    log(f"{'model (vis+hid)':<17}{'k':>6}{'1 chain (ms)':>15}{'10 chains (ms)':>16}{'128 chains (ms)':>17}", logfile)
    rng = np.random.default_rng(0)
    results = []
    for nv, nh in ((85, 30), (156, 90)):
        m = RBM(nv, nh, 0.01, rng)
        m.W[...] = rng.normal(0, 0.1, (nh, nv)); m.a[...] = rng.normal(0, 0.1, nv); m.b[...] = rng.normal(0, 0.1, nh)
        for k in (1, 10, 100, 1000):
            row = []
            for chains in (1, 10, 128):
                v0 = (rng.random((chains, nv)) < 0.5).astype(F32)
                m.gibbs(v0, min(k, 10))                                  # warm-up
                reps, ts = max(5, min(300, 3000 // k)), []
                for _ in range(reps):
                    t = time.perf_counter(); m.gibbs(v0, k); ts.append(time.perf_counter() - t)
                row.append(1e3 * float(np.median(ts)))
            results.append({"n_vis": nv, "n_hid": nh, "k": k, "ms_1": row[0], "ms_10": row[1], "ms_128": row[2]})
            log(f"{str(nv) + '+' + str(nh):<17}{k:>6}{row[0]:>15.3f}{row[1]:>16.3f}{row[2]:>17.3f}", logfile)
    # one full training update (positive phase + negative phase + ADAM) for the NSL-KDD model, batch 128
    m = RBM(85, 30, 0.01, rng); v = (rng.random((128, 85)) < 0.5).astype(F32)
    for k in (1, 10):
        ts = []
        for _ in range(200):
            t = time.perf_counter(); m.update(v, k); ts.append(time.perf_counter() - t)
        ms = 1e3 * float(np.median(ts))
        results.append({"full_update_k": k, "ms": ms})
        log(f"one complete training update, NSL-KDD model, batch 128, k={k}: {ms:.3f} ms", logfile)
    save_json(results, OUT / "timing.json")
    log("saved: results/timing.json", logfile)


# =============================================================================================
# Part 2: one training run (executed in its own process when N_WORKERS > 1)
# =============================================================================================
def train_one(job):
    k, seed, bins = job["k"], job["seed"], job["bins"]
    tag = f"[{bins:<8} CD-{k:<4} seed {seed:>2}]"
    Xtr, ytr, Xte, yte, info = load_data(bins)
    rng = np.random.default_rng(1000 + seed)
    idx = balance(ytr, rng)
    V = np.hstack([Xtr[idx], np.repeat(ytr[idx, None].astype(F32), 3, 1)]).astype(F32)
    n_up = int(np.ceil(len(V) / BATCH_SIZE))
    model = RBM(V.shape[1], HIDDEN, LEARNING_RATE, rng)
    curve, marks, best = {}, {}, (0.0, 0)
    t0 = time.perf_counter(); u = 0
    print(f"{tag} started: {len(V)} balanced training records, {n_up} updates per pass, {PASSES} passes", flush=True)
    for ep in range(1, PASSES + 1):
        Vs = V[rng.permutation(len(V))]                    # shuffled copy, then contiguous slices
        for i in range(0, len(V), BATCH_SIZE):
            model.update(Vs[i:i + BATCH_SIZE], k)
            u += 1
            if u in UPDATE_MARKS:
                marks[u] = metrics(yte, model.predict_free_energy(Xte))
        if ep % PRINT_EVERY == 0 or ep in REPORT_PASSES or ep == PASSES:
            r = metrics(yte, model.predict_free_energy(Xte))
            r["acc_vote"] = float(accuracy_score(yte, model.predict_gibbs_vote(Xte, VOTE_STEPS)))
            curve[ep] = r
            if r["acc"] > best[0]:
                best = (r["acc"], ep)
            el = time.perf_counter() - t0
            print(f"{tag} pass {ep:>5}/{PASSES}  acc {r['acc']:.4f}  F1 {r['f1']:.4f}  vote-acc {r['acc_vote']:.4f}  "
                  f"best {best[0]:.4f} (pass {best[1]})  elapsed {hms(el)}  left {hms(el / ep * (PASSES - ep))}",
                  flush=True)
            save_json({"k": k, "seed": seed, "bins": bins, "settings": SETTINGS, "info": info, "passes_done": ep,
                       "curve": curve, "update_marks": marks, "ms_per_update": 1e3 * el / u},
                      OUT / f"run_{bins}_cd{k}_seed{seed}.json")      # partial results are kept
    el = time.perf_counter() - t0
    print(f"{tag} FINISHED in {hms(el)}: final acc {curve[PASSES]['acc']:.4f}, best {best[0]:.4f} at pass {best[1]}",
          flush=True)
    return {"k": k, "seed": seed, "bins": bins, "curve": curve, "marks": marks, "n_up": n_up, "ms_per_update": 1e3 * el / u}


SETTINGS = {"BINS_LIST": BINS_LIST, "K_VALUES": K_VALUES, "SEEDS": SEEDS, "PASSES": PASSES, "HIDDEN": HIDDEN,
            "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, "VOTE_STEPS": VOTE_STEPS}


def summarise(results, logfile):
    log("=" * 100, logfile)
    log("SUMMARY (mean ± sample s.d. over seeds; accuracy and F1 on the official KDDTest+)", logfile)
    log("Moro & Prati, Table 1 (2000 epochs): k=1 0.906, k=10 0.935, k=100 0.937, k=1000 0.938; D-Wave QRBM 0.937", logfile)
    for bins in BINS_LIST:
      for k in K_VALUES:
        runs = [r for r in results if r["k"] == k and r["bins"] == bins]
        if not runs:
            continue
        passes = sorted(set.intersection(*[set(r["curve"]) for r in runs]))
        log("-" * 100, logfile)
        log(f"{bins} bins, CD-{k}: {len(runs)} runs, {runs[0]['n_up']} updates per pass, "
            f"{np.mean([r['ms_per_update'] for r in runs]):.3f} ms per update (including evaluation)", logfile)
        log(f"{'pass':>6}{'accuracy':>20}{'F1 (attack)':>20}{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>8}{'acc, Gibbs vote':>20}", logfile)
        sd = (lambda x: np.std(x, ddof=1)) if len(runs) > 1 else (lambda x: 0.0)
        for p in passes:
            if p not in REPORT_PASSES and p != passes[-1]:
                continue
            g = lambda key: [r["curve"][p][key] for r in runs]
            log(f"{p:>6}{np.mean(g('acc')):>12.4f} ± {sd(g('acc')):.4f}{np.mean(g('f1')):>12.4f} ± {sd(g('f1')):.4f}"
                f"{np.mean(g('TP')):>8.3f}{np.mean(g('FP')):>8.3f}{np.mean(g('FN')):>8.3f}{np.mean(g('TN')):>8.3f}"
                f"{np.mean(g('acc_vote')):>12.4f} ± {sd(g('acc_vote')):.4f}", logfile)
        mean_curve = [np.mean([r["curve"][p]["acc"] for r in runs]) for p in passes]
        j = int(np.argmax(mean_curve))
        log(f"best mean accuracy at any evaluated pass (selected on the test set, i.e. optimistic): "
            f"{mean_curve[j]:.4f} at pass {passes[j]}; best single run "
            f"{max(max(c['acc'] for c in r['curve'].values()) for r in runs):.4f}", logfile)
        log("if one 'epoch' meant one parameter update:", logfile)
        for q in UPDATE_MARKS:
            vals = [r["marks"][q]["acc"] for r in runs if q in r["marks"]]
            if vals:
                log(f"   after {q:>5} updates: accuracy {np.mean(vals):.4f} ± {sd(vals):.4f}", logfile)


def main():
    OUT.mkdir(exist_ok=True)
    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    logfile = OUT / f"log_{stamp}.txt"
    n_cpu = os.cpu_count() or 1
    jobs = [{"bins": bn, "k": k, "seed": sd} for bn in BINS_LIST for k in K_VALUES for sd in SEEDS]
    jobs.sort(key=lambda j: -j["k"])                    # longest runs first, so the CPU stays full to the end
    workers = n_cpu if N_WORKERS == "auto" else int(N_WORKERS)
    workers = max(1, min(workers, len(jobs)))
    log(f"Classical RBM baseline for Moro & Prati (2023). Started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", logfile)
    log(f"settings: {SETTINGS}", logfile)
    log(f"{n_cpu} logical CPUs; {len(jobs)} runs; {workers} run at the same time "
        f"(up to {100 * workers / n_cpu:.0f}% of the CPU)", logfile)
    if len(jobs) < n_cpu and N_WORKERS == "auto":
        log(f"note: only {len(jobs)} runs for {n_cpu} CPUs. Add seeds, k values or bins to use the whole processor.", logfile)
    for f in (TRAIN_FILE, TEST_FILE):
        if not (HERE / f).exists():
            log(f"ERROR: cannot find {HERE / f}. Put the NSL-KDD files in a 'Dataset' folder next to this script.", logfile)
            return
    if RUN_TIMING:
        run_timing(logfile)
    if not RUN_TRAINING:
        return
    Xtr, ytr, Xte, yte, info = load_data(BINS_LIST[0])
    n_bal = 2 * min(int((ytr == 0).sum()), int((ytr == 1).sum()))
    n_up = int(np.ceil(n_bal / BATCH_SIZE))
    log("=" * 100, logfile)
    log("PART 2. TRAINING ON KDDTrain+, TESTING ON THE OFFICIAL KDDTest+", logfile)
    log(f"train file {info['n_train_file']} records -> balanced {n_bal}; test {info['n_test']} records; "
        f"{info['feature_bits']} feature bits + 3 label bits = {info['feature_bits'] + 3} visible units", logfile)
    log(f"attack types: {info['attack_types_train']} in training, {info['attack_types_test']} in test; "
        f"{100 * info['test_unseen_attack_fraction']:.1f}% of test records are attack types never seen in training", logfile)
    # time estimate: single-core speed, slowed down when every logical CPU is busy (hyper-threading, E-cores)
    m = RBM(85, HIDDEN, LEARNING_RATE, np.random.default_rng(0))
    v = (np.random.default_rng(0).random((BATCH_SIZE, 85)) < .5).astype(F32)
    slow = 1.0 + 1.5 * (workers - 1) / max(n_cpu - 1, 1)
    for k in K_VALUES:
        reps = max(5, 2000 // k)
        t = time.perf_counter()
        for _ in range(reps):
            m.update(v, k)
        per = (time.perf_counter() - t) / reps
        log(f"CD-{k}: {1e3 * per:.3f} ms per update on one idle core; one run of {PASSES} passes "
            f"({n_up * PASSES:,} updates) should take about {hms(per * n_up * PASSES * slow)} with {workers} runs in parallel", logfile)
    log(f"A progress line is printed for each run every {PRINT_EVERY} passes. Partial results: results/run_*.json", logfile)
    log("-" * 100, logfile)
    t0 = time.perf_counter()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            results = list(pool.imap_unordered(train_one, jobs, chunksize=1))
    else:
        results = [train_one(j) for j in jobs]
    log(f"all runs finished in {hms(time.perf_counter() - t0)}", logfile)
    summarise(results, logfile)
    save_json(results, OUT / f"summary_{stamp}.json")
    log(f"saved: results/summary_{stamp}.json and the log {logfile.name}", logfile)


if __name__ == "__main__":       # required on Windows for parallel runs
    main()
