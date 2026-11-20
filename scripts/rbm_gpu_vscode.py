r"""
Classical RBM baseline on the GPU: many independent training runs (seeds) trained together as one batched
computation. Same algorithm and settings as mp_baseline_vscode.py (CD-k, ADAM, batch 128, label bits 111/000).
Two read-outs are measured at every evaluation:
  - free energy: compares the model's probabilities of label 111 and 000 for each record (exact, no sampling);
  - Gibbs + majority vote, as described by Moro & Prati: the feature bits are held fixed, the 3 label bits start
    random, VOTE_STEPS Gibbs steps are run on the label bits, and the majority of the 3 sampled bits decides.
    The paper does not say whether features are held fixed, how the label bits start, or how many chains are
    used; this script uses one chain per record from a random start (the natural reading).
For CSE the test set is balanced (all attacks + an equal random number of normal records), because the class
mix implied by Moro & Prati's Tables 1-2 is 50:50. Run it in VS Code with the "Run Python File" button, using the Python that has PyTorch
with CUDA.

WHERE TO PUT THINGS
    C:\RBM\rbm_gpu_vscode.py              <- this file
    C:\RBM\Dataset\KDDTrain+.txt, KDDTest+.txt          (for DATASET = "nsl")
    C:\RBM\results\cse_cache\*.npy        (for DATASET = "cse": created by cse_benchmark_vscode.py)
Results: C:\RBM\results\gpu_<dataset>_<date>.json and a log file, updated at every evaluation.

Why batched: one RBM run (batch 128, 85-156 visible, 30-90 hidden units) is far too small to keep a GPU busy,
and each update depends on the previous one. Stacking all runs lets one GPU kernel do the work of every run.
Each run keeps its own initial weights, its own balanced training subset and its own data order.
SPEED (version 3): the run is limited by how fast the GPU can be given small jobs, not by the GPU's arithmetic,
so more CPU cores or RAM cannot help directly. Three things do:
  1. CUDA graphs: each training update is recorded once and then replayed as a single GPU command;
  2. fewer GPU jobs per update (one-step Bernoulli sampling; each epoch's data gathered once on the GPU);
  3. PARALLEL_K: the CD-k values run at the same time, in separate processes sharing the GPU.
Before training, a self-check compares graph updates with ordinary updates; if they disagree, or a graph cannot be
recorded, the script falls back to ordinary updates and says so. Sampling uses the GPU's default random generator,
so the random numbers (not the algorithm) differ from earlier versions.
Needs: torch (with CUDA), numpy, pandas (pandas only for DATASET = "nsl")
"""

# =============================================================================================
# SETTINGS -- change these, then press Run
# =============================================================================================
DATASET = "cse"            # "cse" (needs the cache from cse_benchmark_vscode.py) or "nsl"
K_VALUES = [1, 10]         # CD-k values; runs with the same k are trained together (in parallel if PARALLEL_K)
SEEDS = [0, 1, 2, 3, 4]    # one run per seed, per learning rate and per k
LEARNING_RATES = [0.01]    # ADAM learning rates compared side by side (Moro & Prati: 0.01). For a given seed,
                                 # every learning rate starts from the same weights and uses the same training subset.
PASSES = 2000              # epochs = full passes over the balanced training set (Moro & Prati: 2000)
EVAL_EVERY = 10            # evaluate and print every this many epochs (also at epochs 1, 2, 5, 10, 20, ...)
VOTE_STEPS = [1, 10, 100]  # Gibbs steps for the majority-vote read-out (one chain; read at each of these steps)
CSE_BALANCED_TEST = True   # CSE only: evaluate on a balanced test set (as Moro & Prati's numbers imply)
EVAL_SUBSAMPLE = 200_000   # CSE only: test records used at intermediate evaluations (final = whole test set)
NSL_BINS = "quantile"      # NSL only: "quantile" or "width"
BATCH_SIZE = 128
SMOKE_TEST = False         # True = run only 2 epochs to check that everything works
USE_CUDA_GRAPHS = True     # record each update once and replay it (much faster); falls back automatically
PARALLEL_K = True          # run the CD-k values at the same time in separate processes (one log/JSON per k)

# =============================================================================================
import os, time, json, datetime
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CACHE = OUT / "cse_cache"
REPORT = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000}
MP = {"nsl": {1: 0.906, 10: 0.935, 100: 0.937, 1000: 0.938}, "cse": {1: 0.800, 10: 0.897, 100: 0.915, 1000: 0.924}}


def log(msg, file):
    print(msg, flush=True)
    with open(file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def hms(s):
    s = int(max(s, 0)); return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


# =============================================================================================
# Data
# =============================================================================================
def load_nsl():
    """Same 82-bit encoding as mp_baseline_vscode.py."""
    import pandas as pd
    cols = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
            "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"]
    onehot, flags = ["protocol_type", "flag"], ["land", "logged_in", "is_host_login", "is_guest_login"]
    tr = pd.read_csv(HERE / "Dataset" / "KDDTrain+.txt", header=None, names=cols).drop(columns=["num_outbound_cmds"])
    te = pd.read_csv(HERE / "Dataset" / "KDDTest+.txt", header=None, names=cols).drop(columns=["num_outbound_cmds"])
    num = [c for c in tr.columns if c not in onehot + flags + ["service", "class", "difficulty"]]
    order = tr[num].nunique().sort_values(ascending=False).index.tolist()
    nbits = {c: (2 if i < 24 else 1) for i, c in enumerate(order)}
    edges = {}
    for c in num:
        x, nb = tr[c].to_numpy(float), 2 ** nbits[c]
        edges[c] = (np.histogram_bin_edges(x, bins=nb)[1:-1] if NSL_BINS == "width"
                    else np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])))
    levels = {c: sorted(tr[c].unique()) for c in onehot}
    svc = {s: i + 1 for i, s in enumerate(sorted(tr["service"].unique()))}

    def enc(d):
        X = []
        for c in num:
            idx = np.digitize(d[c].to_numpy(float), edges[c]); X.append((idx[:, None] >> np.arange(nbits[c])[::-1]) & 1)
        X.append(d[flags].to_numpy())
        for c in onehot:
            X.append(np.stack([(d[c] == l).to_numpy() for l in levels[c]], 1))
        code = d["service"].map(svc).fillna(0).astype(int).to_numpy(); X.append((code[:, None] >> np.arange(7)[::-1]) & 1)
        return np.hstack(X).astype(np.uint8), (d["class"] == "normal").to_numpy().astype(np.int8)
    Btr, ytr = enc(tr); Bte, yte = enc(te)
    return Btr, ytr, Bte, yte, 30


def load_cse():
    if not (CACHE / "Btr.npy").exists():
        raise SystemExit(f"ERROR: no CSE cache in {CACHE}. Run cse_benchmark_vscode.py first (it creates the cache).")
    return (np.load(CACHE / "Btr.npy"), np.load(CACHE / "ytr.npy"), np.load(CACHE / "Bte.npy"),
            np.load(CACHE / "yte.npy"), 90)


def balanced_idx(y, rng):
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; n = min(len(i0), len(i1))
    return np.sort(np.concatenate([rng.choice(i0, n, replace=False), rng.choice(i1, n, replace=False)]))


def metrics(y, p):
    """accuracy, F1 for the attack class (label 0), column-normalised confusion matrix (as in Moro & Prati)."""
    y = y.astype(np.int64); p = p.astype(np.int64)
    tp = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 0) & (y == 1)).sum())
    fn = int(((p == 1) & (y == 0)).sum()); tn = int(((p == 1) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    cp0, cp1 = tp + fp, fn + tn
    return {"acc": (tp + tn) / len(y), "f1": f1, "TP": tp / cp0 if cp0 else 0.0, "FP": fp / cp0 if cp0 else 0.0,
            "FN": fn / cp1 if cp1 else 0.0, "TN": tn / cp1 if cp1 else 0.0}


# =============================================================================================
# Batched RBM: R independent runs; parameters of all runs in one tensor
# =============================================================================================
class BatchedRBM:
    def __init__(self, R, nv, nh, seeds, lrs, dev):
        self.R, self.nv, self.nh, self.dev = R, nv, nh, dev
        self.lr = torch.tensor(np.array(lrs, np.float32).reshape(R, 1), device=dev)   # one learning rate per run
        P = nh * nv + nv + nh
        theta = np.zeros((R, P), np.float32)
        for r, s in enumerate(seeds):                                   # each run: its own initial weights
            theta[r, :nh * nv] = np.random.default_rng(1000 + s).normal(0, 0.01, nh * nv)
        self.theta = torch.tensor(theta, device=dev)
        self.grad = torch.zeros_like(self.theta)
        self.m = torch.zeros_like(self.theta); self.v = torch.zeros_like(self.theta)
        self.t = torch.zeros((), device=dev)                            # ADAM step counter, kept on the GPU
        self.W, self.a, self.b = self._views(self.theta)
        self.gW, self.ga, self.gb = self._views(self.grad)
        self.graph = None

    def _views(self, T):
        nh, nv = self.nh, self.nv
        return T[:, :nh * nv].view(self.R, nh, nv), T[:, nh * nv:nh * nv + nv], T[:, nh * nv + nv:]

    def state(self):
        return [x.clone() for x in (self.theta, self.m, self.v, self.t)]

    def set_state(self, st):
        for x, y in zip((self.theta, self.m, self.v, self.t), st):
            x.copy_(y)                                                   # in place: graph keeps its addresses

    def update(self, v, k):
        """v: (R, B, nv) float32 visible data. One CD-k step + ADAM for every run (default CUDA generator)."""
        W, a, b = self.W, self.a, self.b
        Wt = W.transpose(1, 2)
        ph = torch.sigmoid(torch.baddbmm(b.unsqueeze(1), v, Wt))          # positive phase (R, B, nh)
        p = ph
        for _ in range(k):                                               # negative phase, chains start at data
            h = torch.bernoulli(p)
            pv = torch.sigmoid(torch.baddbmm(a.unsqueeze(1), h, W))
            vk = torch.bernoulli(pv)
            p = torch.sigmoid(torch.baddbmm(b.unsqueeze(1), vk, Wt))
        B = v.shape[1]
        pos = torch.bmm(ph.transpose(1, 2), v)
        self.gW.copy_(torch.baddbmm(pos, p.transpose(1, 2), vk, beta=1.0 / B, alpha=-1.0 / B))
        self.ga.copy_(v.mean(1) - vk.mean(1))
        self.gb.copy_(ph.mean(1) - p.mean(1))
        self.t.add_(1)                                                   # ADAM (0.9, 0.999, 1e-8), ascent
        self.m.mul_(0.9).add_(self.grad, alpha=0.1)
        self.v.mul_(0.999).addcmul_(self.grad, self.grad, value=0.001)
        c1 = 1 - torch.pow(0.9, self.t); c2 = 1 - torch.pow(0.999, self.t)
        self.theta.add_(self.lr * (self.m / c1) / ((self.v / c2).sqrt_().add_(1e-8)))

    def make_graph(self, B, k):
        """Record one full update (batch size B) as a CUDA graph. Model state is restored afterwards."""
        st = self.state()
        self.v_static = torch.zeros((self.R, B, self.nv), device=self.dev)
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):                                          # warm-up (required before recording)
                self.update(self.v_static, k)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.update(self.v_static, k)
        torch.cuda.synchronize()
        self.set_state(st)
        self.graph = g

    def step(self, feats_u8, labels_u8, k, use_graph):
        """One update from uint8 features (R, B, nf) and labels (R, B)."""
        nf = feats_u8.shape[2]
        if use_graph and self.graph is not None and feats_u8.shape[1] == self.v_static.shape[1]:
            self.v_static[:, :, :nf].copy_(feats_u8)
            self.v_static[:, :, nf:].copy_(labels_u8.unsqueeze(2))
            self.graph.replay()
        else:
            v = torch.empty((self.R, feats_u8.shape[1], self.nv), device=self.dev)
            v[:, :, :nf].copy_(feats_u8); v[:, :, nf:].copy_(labels_u8.unsqueeze(2))
            self.update(v, k)

    @torch.no_grad()
    def predict(self, X_dev, chunk=100_000):
        """Free-energy comparison of label bits 111 (normal) vs 000 (attack). X_dev: (N, nf) uint8 on device.
        Returns (R, N) int8 predictions (1 = normal)."""
        Wf, Wl = self.W[:, :, :-3], self.W[:, :, -3:]
        lab_h = Wl.sum(2).unsqueeze(1)                                   # (R, 1, nh)
        lab_a = self.a[:, -3:].sum(1).unsqueeze(1)                        # (R, 1)
        out = []
        for i in range(0, X_dev.shape[0], chunk):
            X = X_dev[i:i + chunk].float()
            pre = torch.matmul(X, Wf.transpose(1, 2)) + self.b.unsqueeze(1)          # (R, n, nh)
            s1 = torch.nn.functional.softplus(pre + lab_h).sum(2)
            s0 = torch.nn.functional.softplus(pre).sum(2)
            out.append(((-lab_a - s1) < (-s0)).to(torch.int8).cpu())                  # F(111) < F(000)
        return torch.cat(out, 1).numpy()

    @torch.no_grad()
    def predict_vote(self, X_dev, steps, gen, chunk=100_000):
        """Moro & Prati's read-out: features fixed, label bits start random, Gibbs steps on the label bits,
        majority of the 3 sampled bits. One chain per record, read after each number of steps in `steps`.
        Returns {steps: (R, N) int8 predictions (1 = normal)}."""
        Wf, Wl = self.W[:, :, :-3], self.W[:, :, -3:]                    # (R, nh, nf), (R, nh, 3)
        WlT = Wl.transpose(1, 2)                                         # (R, 3, nh)
        al = self.a[:, -3:].unsqueeze(1)                                 # (R, 1, 3)
        kmax = max(steps); out = {s: [] for s in steps}
        for i in range(0, X_dev.shape[0], chunk):
            X = X_dev[i:i + chunk].float()
            pre = torch.matmul(X, Wf.transpose(1, 2)) + self.b.unsqueeze(1)          # (R, n, nh), fixed features
            lab = (torch.rand((self.R, X.shape[0], 3), device=self.dev, generator=gen) < 0.5).float()
            for s in range(1, kmax + 1):
                ph = torch.sigmoid(pre + torch.bmm(lab, WlT))
                h = (torch.rand(ph.shape, device=self.dev, generator=gen) < ph).float()
                pl = torch.sigmoid(torch.bmm(h, Wl) + al)
                lab = (torch.rand(pl.shape, device=self.dev, generator=gen) < pl).float()
                if s in out:
                    out[s].append((lab.sum(2) >= 2).to(torch.int8).cpu())          # majority of 3 bits
        return {s: torch.cat(v, 1).numpy() for s, v in out.items()}


# =============================================================================================
def prepare(dev, logfile):
    Btr, ytr, Bte, yte, nh = load_nsl() if DATASET == "nsl" else load_cse()
    nf = Btr.shape[1]; nv = nf + 3
    if DATASET == "cse" and CSE_BALANCED_TEST:
        bt = balanced_idx(yte, np.random.default_rng(99))
        Bte, yte = Bte[bt], yte[bt]
    rng_ev = np.random.default_rng(7)
    if DATASET == "nsl" or len(yte) <= EVAL_SUBSAMPLE:
        ev = np.arange(len(yte))
    else:                                                                 # subsample with the same class mix
        i0, i1 = np.where(yte == 0)[0], np.where(yte == 1)[0]; n0 = int(round(EVAL_SUBSAMPLE * len(i0) / len(yte)))
        ev = np.sort(np.concatenate([rng_ev.choice(i0, n0, replace=False),
                                     rng_ev.choice(i1, EVAL_SUBSAMPLE - n0, replace=False)]))
    return Btr, ytr, Bte, yte, nh, nf, nv, ev


def self_check(model, Ftr, Ltr, bal_d, k, logfile):
    """Compare graph updates with ordinary updates on the same 60 batches from the same start.
    Two ordinary runs with different random numbers set the expected agreement."""
    R = model.R; st = model.state(); nb = 60
    idx = bal_d[:, :nb * BATCH_SIZE]
    F = Ftr[idx]; L = Ltr[idx]
    steps_done = {}
    def run(seed, use_graph):
        model.set_state(st); torch.cuda.manual_seed(seed)
        for i in range(nb):
            sl = slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
            model.step(F[:, sl], L[:, sl], k, use_graph)
        torch.cuda.synchronize()
        steps_done[use_graph] = float(model.t - st[3])
        return (model.theta - st[0]).flatten()
    d1, d2, dg = run(1, False), run(2, False), run(3, True)
    model.set_state(st)
    cc = lambda x, y: float(torch.corrcoef(torch.stack([x, y]))[0, 1])
    c_ee, c_eg = cc(d1, d2), 0.5 * (cc(d1, dg) + cc(d2, dg))
    ratio = float(dg.norm() / (0.5 * (d1.norm() + d2.norm())))
    ok = c_eg >= c_ee - 0.05 and 0.8 <= ratio <= 1.25 and steps_done[True] == nb == steps_done[False]
    log(f"[CD-{k}] self-check over {nb} updates: agreement ordinary-vs-ordinary {c_ee:.3f}, graph-vs-ordinary "
        f"{c_eg:.3f}, size ratio {ratio:.3f} -> {'PASSED, using CUDA graphs' if ok else 'FAILED, using ordinary updates'}",
        logfile)
    return ok


def run_k(k, stamp):
    """Train all runs for one CD-k value. Writes its own log and JSON."""
    OUT.mkdir(exist_ok=True)
    tag = f"gpu_{DATASET}_k{k}"
    logfile = OUT / f"{tag}_log_{stamp}.txt"
    resfile = OUT / f"{tag}_{stamp}.json"
    passes = 2 if SMOKE_TEST else PASSES
    if not torch.cuda.is_available():
        log("WARNING: CUDA is not available to this Python; running on the CPU (much slower).", logfile)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False                      # full float32 precision, as on the CPU
    torch.backends.cudnn.allow_tf32 = False
    log(f"GPU RBM baseline (version 3), dataset {DATASET}, CD-{k}. Started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        logfile)
    log(f"device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}; torch {torch.__version__}", logfile)
    log(f"settings: SEEDS={SEEDS}, LEARNING_RATES={LEARNING_RATES}, PASSES={passes}, EVAL_EVERY={EVAL_EVERY}, "
        f"BATCH={BATCH_SIZE}, USE_CUDA_GRAPHS={USE_CUDA_GRAPHS}" + (f", NSL_BINS={NSL_BINS}" if DATASET == "nsl" else ""),
        logfile)
    Btr, ytr, Bte, yte, nh, nf, nv, ev = prepare(dev, logfile)
    RUNS = [(lr, sd) for lr in LEARNING_RATES for sd in SEEDS]                   # one batched run per (lr, seed)
    R = len(RUNS); run_seeds = [sd for _, sd in RUNS]; run_lrs = [lr for lr, _ in RUNS]
    bal = [balanced_idx(ytr, np.random.default_rng(1000 + sd)) for sd in run_seeds]   # same subset for the same seed
    nbal = len(bal[0]); n_up = (nbal + BATCH_SIZE - 1) // BATCH_SIZE
    log(f"train {len(ytr):,} records -> balanced {nbal:,} per run; test {len(yte):,}"
        + (" (balanced)" if DATASET == "cse" and CSE_BALANCED_TEST else "") +
        f"; {nf} feature bits + 3 label bits = {nv} visible, {nh} hidden units; {n_up:,} updates per epoch", logfile)
    mp = MP[DATASET].get(k)
    log(f"Moro & Prati ({DATASET}, k = {k}): {f'{mp:.3f}' if mp else 'not reported'}", logfile)

    Ftr = torch.tensor(Btr, device=dev)                                   # uint8 features on the GPU
    Ltr = torch.tensor(ytr.astype(np.uint8), device=dev)                  # uint8 labels (1 = normal)
    bal_d = torch.tensor(np.stack(bal), device=dev, dtype=torch.long)     # (R, nbal)
    Xte = torch.tensor(Bte, device=dev)
    Xev = Xte[torch.tensor(ev, device=dev)] if len(ev) < len(yte) else Xte
    yev = yte[ev]
    del Btr, Bte
    results = {"dataset": DATASET, "settings": {"K": k, "SEEDS": SEEDS, "PASSES": passes,
               "LEARNING_RATES": LEARNING_RATES, "BATCH_SIZE": BATCH_SIZE, "hidden": nh, "visible": nv,
               "NSL_BINS": NSL_BINS if DATASET == "nsl" else None, "n_balanced": nbal, "eval_records": int(len(ev)),
               "VOTE_STEPS": VOTE_STEPS, "CSE_BALANCED_TEST": CSE_BALANCED_TEST, "test_records": int(len(yte)),
               "torch": torch.__version__, "version": 3}, "runs": {}}

    model = BatchedRBM(R, nv, nh, run_seeds, run_lrs, dev)
    use_graph = False
    if USE_CUDA_GRAPHS and dev.type == "cuda":
        try:
            model.make_graph(BATCH_SIZE, k)
            use_graph = self_check(model, Ftr, Ltr, bal_d, k, logfile)
        except Exception as e:                                          # noqa: BLE001
            log(f"[CD-{k}] CUDA graph could not be recorded ({type(e).__name__}: {e}); using ordinary updates", logfile)
            model.graph = None
    results["settings"]["cuda_graphs_used"] = use_graph
    # speed estimate
    st = model.state()
    F = Ftr[bal_d[:, :BATCH_SIZE]]; L = Ltr[bal_d[:, :BATCH_SIZE]]
    for _ in range(5):
        model.step(F, L, k, use_graph)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter(); nrep = 50
    for _ in range(nrep):
        model.step(F, L, k, use_graph)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    per = (time.perf_counter() - t) / nrep
    model.set_state(st)
    log(f"CD-{k}: {R} runs together, {1e3 * per:.3f} ms per batched update -> about {hms(per * n_up)} per epoch, "
        f"{hms(per * n_up * passes)} for {passes} epochs (plus evaluation)", logfile)

    torch.manual_seed(12345 + k)                                         # sampling in the updates
    if dev.type == "cuda":
        torch.cuda.manual_seed(12345 + k)
    gen = torch.Generator(device=dev); gen.manual_seed(12345 + k)        # data order
    gen_ev = torch.Generator(device=dev); gen_ev.manual_seed(777 + k)    # read-out sampling (separate stream)
    Fep = torch.empty((R, nbal, nf), dtype=torch.uint8, device=dev)        # this epoch's data, gathered once
    Lep = torch.empty((R, nbal), dtype=torch.uint8, device=dev)
    curves = [dict() for _ in RUNS]
    t0 = time.perf_counter()
    for ep in range(1, passes + 1):
        perm = torch.argsort(torch.rand((R, nbal), device=dev, generator=gen), dim=1)
        order = torch.gather(bal_d, 1, perm)                              # (R, nbal): each run its own data order
        torch.index_select(Ftr, 0, order.reshape(-1), out=Fep.view(-1, nf))
        torch.index_select(Ltr, 0, order.reshape(-1), out=Lep.view(-1))
        del perm, order
        with torch.no_grad():
            for i in range(0, nbal, BATCH_SIZE):
                model.step(Fep[:, i:i + BATCH_SIZE], Lep[:, i:i + BATCH_SIZE], k, use_graph)
        if ep % EVAL_EVERY == 0 or ep in REPORT or ep == passes:
            P = model.predict(Xev)
            V = model.predict_vote(Xev, VOTE_STEPS, gen_ev)
            ms = [metrics(yev, P[r]) for r in range(R)]
            for r in range(R):
                for s in VOTE_STEPS:
                    mv = metrics(yev, V[s][r])
                    ms[r][f"acc_vote{s}"] = mv["acc"]; ms[r][f"f1_vote{s}"] = mv["f1"]
                curves[r][ep] = ms[r]
            el = time.perf_counter() - t0
            parts = []
            for lr in LEARNING_RATES:
                ids = [i for i, (l, _) in enumerate(RUNS) if l == lr]
                fe = np.array([ms[i]["acc"] for i in ids])
                f1 = np.array([ms[i]["f1"] for i in ids])
                txt = (f"lr {lr:g}: ACCURACY, free-energy read-out " + " ".join(f"{x:.3f}" for x in fe) +
                       f" (F1 mean {f1.mean():.3f})")
                for s in VOTE_STEPS:
                    vv = np.array([ms[i][f"acc_vote{s}"] for i in ids]); vf = np.array([ms[i][f"f1_vote{s}"] for i in ids])
                    txt += f" | vote-{s} read-out {vv.mean():.3f} (F1 {vf.mean():.3f})"
                parts.append(txt)
            log(f"[CD-{k}] epoch {ep:>4}/{passes}  " + "  ||  ".join(parts) +
                f"   elapsed {hms(el)}  left {hms(el / ep * (passes - ep))}", logfile)
            results["runs"][f"k={k}"] = {"runs": RUNS, "curves": curves}
            save_json(results, resfile)
    P = model.predict(Xte)                                                # final: whole test set
    V = model.predict_vote(Xte, VOTE_STEPS, gen_ev)
    final = [metrics(yte, P[r]) for r in range(R)]
    for r in range(R):
        for s in VOTE_STEPS:
            mv = metrics(yte, V[s][r])
            final[r][f"acc_vote{s}"] = mv["acc"]; final[r][f"f1_vote{s}"] = mv["f1"]
    results["runs"][f"k={k}"] = {"runs": RUNS, "curves": curves, "final": final, "seconds": time.perf_counter() - t0}
    save_json(results, resfile)
    log(f"[CD-{k}] FINISHED in {hms(time.perf_counter() - t0)}. Summary (best = evaluation records, "
        f"final = whole test set):", logfile)
    for lr in LEARNING_RATES:
        ids = [i for i, (l, _) in enumerate(RUNS) if l == lr]
        log(f"   lr {lr:g}  (Moro & Prati, k = {k}: {f'{mp:.3f}' if mp else 'not reported'})", logfile)
        for key, name in [("acc", "free energy")] + [(f"acc_vote{s}", f"vote, {s} step" + "s" * (s > 1)) for s in VOTE_STEPS]:
            fin = np.array([final[i][key] for i in ids])
            ff1 = np.array([final[i][key.replace("acc", "f1")] for i in ids])
            curv = [np.array([curves[i][e][key] for e in sorted(curves[i])]) for i in ids]
            best = [c.max() for c in curv]; best_ep = [sorted(curves[i])[int(c.argmax())] for i, c in zip(ids, curv)]
            sd = float(np.std(fin, ddof=1)) if len(fin) > 1 else 0.0
            reach = f"; runs ever reaching {mp}: {sum(b >= mp for b in best)}/{len(best)}" if mp else ""
            log(f"      {name:<18} final accuracy {fin.mean():.4f} ± {sd:.4f} (range {fin.min():.3f}-{fin.max():.3f}), "
                f"F1 {ff1.mean():.4f}; "
                f"best epoch per run {' '.join(f'{b:.3f}@{e}' for b, e in zip(best, best_ep))}{reach}", logfile)
    log(f"saved: {resfile.name} and {logfile.name}", logfile)


def main():
    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    if PARALLEL_K and len(K_VALUES) > 1:
        import multiprocessing as mp_
        ctx = mp_.get_context("spawn")
        print(f"Starting {len(K_VALUES)} processes, one per CD-k value {K_VALUES}; each writes its own log and JSON "
              f"in {OUT} (stamp {stamp}).", flush=True)
        procs = [ctx.Process(target=run_k, args=(k, stamp), name=f"CD-{k}") for k in K_VALUES]
        for p in procs:
            p.start(); time.sleep(20)                                     # stagger the data loading
        for p in procs:
            p.join()
        print("All processes finished: " + ", ".join(f"{p.name} exit code {p.exitcode}" for p in procs), flush=True)
    else:
        for k in K_VALUES:
            run_k(k, stamp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
