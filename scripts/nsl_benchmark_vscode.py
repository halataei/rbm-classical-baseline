r"""
How high can binary accuracy go on the official NSL-KDD test set (KDDTest+)?
Run it in VS Code with the "Run Python File" button. No arguments needed.

WHERE TO PUT THINGS
    C:\RBM\nsl_benchmark_vscode.py      <- this file
    C:\RBM\Dataset\KDDTrain+.txt        <- official training file
    C:\RBM\Dataset\KDDTest+.txt         <- official test file
Results are written to C:\RBM\results\ (a log file and nsl_benchmark.json, updated after every model).
All classifier fits are independent, so they run in parallel, one per logical CPU (single-threaded each).

WHAT IT DOES (everything is trained on KDDTrain+ only and scored on KDDTest+ and its hard subset KDDTest-21)
  Part 1  CEILING. For a fixed encoding, records with identical codes must receive the same prediction, so no
          classifier using that encoding -- even one that has seen the test labels -- can exceed
              sum over distinct test codes of max(#normal, #attack) / #test records.
  Part 2  STANDARD CLASSIFIERS. Eleven common classifiers (plus an RBM-feature pipeline on binary encodings) on
          four representations: the raw 41 features, two 82-bit encodings like Moro & Prati's (equal-width or
          quantile bins), and a 59-bit mean-threshold encoding; each trained on the balanced training set
          (as Moro & Prati) and on the full training set.
  Part 3  SUMMARY, including the best result selected with knowledge of the test labels (deliberately optimistic).
Needs: numpy, pandas, scikit-learn
"""

# =============================================================================================
# SETTINGS -- change these, then press Run
# =============================================================================================
TRAIN_FILE = "Dataset/KDDTrain+.txt"     # relative to the folder of this script
TEST_FILE = "Dataset/KDDTest+.txt"

REPRESENTATIONS = ["raw", "82bit_width", "82bit_quantile", "59bit_mean"]   # remove any you do not want
TRAINING_SETS = ["balanced", "full"]
N_TREES = 300              # trees for random forest / extra trees, rounds for AdaBoost
MLP_MAX_ITER = 200         # maximum epochs for the multilayer perceptron
N_WORKERS = "auto"         # classifiers trained at the same time: "auto" = one per logical CPU, or a number
SEED = 0

# =============================================================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one thread per process; parallelism comes from running many fits at once
import time, json, datetime, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # reported in the log instead

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "class", "difficulty"]
CAT = ["protocol_type", "service", "flag"]
FLAGS = ["land", "logged_in", "is_host_login", "is_guest_login"]


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


# ---------------------------------------------------------------------------------------------- representations
def build_representations(tr, te, num):
    reps = {}

    def raw():
        sc = StandardScaler().fit(np.log1p(tr[num].clip(lower=0)))
        def f(d):
            X = [sc.transform(np.log1p(d[num].clip(lower=0))), d[FLAGS].to_numpy(float)]
            for c in CAT:
                X.append(np.stack([(d[c] == l).to_numpy() for l in sorted(tr[c].unique())], 1).astype(float))
            return np.hstack(X)
        return f(tr), f(te), False

    def bits82(bins):
        order = tr[num].nunique().sort_values(ascending=False).index.tolist()
        nbits = {c: (2 if i < 24 else 1) for i, c in enumerate(order)}
        edges = {}
        for c in num:
            x, nb = tr[c].to_numpy(float), 2 ** nbits[c]
            edges[c] = (np.histogram_bin_edges(x, bins=nb)[1:-1] if bins == "width"
                        else np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])))
        lv = {s: i + 1 for i, s in enumerate(sorted(tr["service"].unique()))}
        def f(d):
            X = []
            for c in num:
                idx = np.digitize(d[c].to_numpy(float), edges[c])
                X.append((idx[:, None] >> np.arange(nbits[c])[::-1]) & 1)
            X.append(d[FLAGS].to_numpy())
            for c in ["protocol_type", "flag"]:
                X.append(np.stack([(d[c] == l).to_numpy() for l in sorted(tr[c].unique())], 1))
            code = d["service"].map(lv).fillna(0).astype(int).to_numpy()
            X.append((code[:, None] >> np.arange(7)[::-1]) & 1)
            return np.hstack(X).astype(np.uint8)
        return f(tr), f(te), True

    def bits59():
        th = tr[num].mean()
        lv = {s: i + 1 for i, s in enumerate(sorted(tr["service"].unique()))}
        def f(d):
            X = [(d[num] >= th).to_numpy(), d[FLAGS].to_numpy()]
            for c in ["protocol_type", "flag"]:
                X.append(np.stack([(d[c] == l).to_numpy() for l in sorted(tr[c].unique())], 1))
            code = d["service"].map(lv).fillna(0).astype(int).to_numpy()
            X.append((code[:, None] >> np.arange(7)[::-1]) & 1)
            return np.hstack(X).astype(np.uint8)
        return f(tr), f(te), True

    makers = {"raw": raw, "82bit_width": lambda: bits82("width"), "82bit_quantile": lambda: bits82("quantile"),
              "59bit_mean": bits59}
    for name in REPRESENTATIONS:
        reps[name] = makers[name]()
    return reps


def ceiling(X, y, mask=None):
    if mask is not None:
        X, y = X[mask], y[mask]
    keys = pd.Series([r.tobytes() for r in np.ascontiguousarray(X)])
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return float(np.maximum(g["sum"], g["count"] - g["sum"]).sum() / len(y)), int(len(g))


def balance(X, y, seed):
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(i0), len(i1))
    idx = np.concatenate([rng.choice(i0, n, replace=False), rng.choice(i1, n, replace=False)])
    return X[idx], y[idx]


def models(binary):
    m = {
        "Logistic regression": LogisticRegression(max_iter=3000),
        "Linear SVM": LinearSVC(dual=False, max_iter=5000),
        "Naive Bayes": BernoulliNB() if binary else GaussianNB(),
        "k-NN (k=1)": KNeighborsClassifier(1, n_jobs=1),
        "k-NN (k=5)": KNeighborsClassifier(5, n_jobs=1),
        "Decision tree": DecisionTreeClassifier(random_state=SEED),
        "Random forest": RandomForestClassifier(N_TREES, random_state=SEED, n_jobs=1),
        "Extra trees": ExtraTreesClassifier(N_TREES, random_state=SEED, n_jobs=1),
        "Gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=N_TREES, random_state=SEED),
        "MLP (100-50)": MLPClassifier((100, 50), max_iter=MLP_MAX_ITER, early_stopping=True, random_state=SEED),
    }
    if binary:
        m["RBM features + logistic"] = make_pipeline(
            BernoulliRBM(n_components=30, learning_rate=0.01, n_iter=20, random_state=SEED),
            LogisticRegression(max_iter=3000))
    return m


# ---------------------------------------------------------------------------------------------- parallel workers
_DATA = {}

def load_all():
    tr = pd.read_csv(HERE / TRAIN_FILE, header=None, names=COLS).drop(columns=["num_outbound_cmds"])
    te = pd.read_csv(HERE / TEST_FILE, header=None, names=COLS).drop(columns=["num_outbound_cmds"])
    ytr = (tr["class"] == "normal").astype(int).to_numpy()
    yte = (te["class"] == "normal").astype(int).to_numpy()
    hard = te["difficulty"].to_numpy() < 21
    num = [c for c in tr.columns if c not in CAT + FLAGS + ["class", "difficulty"]]
    return tr, te, ytr, yte, hard, num


def init_worker():
    """Runs once in each worker process: load the data and build the representations."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    tr, te, ytr, yte, hard, num = load_all()
    _DATA.update(reps=build_representations(tr, te, num), ytr=ytr, yte=yte, hard=hard)


def fit_one(task):
    rep, tset, mn = task
    Xa, Xb, binary = _DATA["reps"][rep]
    ytr, yte, hard = _DATA["ytr"], _DATA["yte"], _DATA["hard"]
    Xt, yt = balance(Xa, ytr, SEED) if tset == "balanced" else (Xa, ytr)
    m = models(binary)[mn]
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(Xt, yt)
        conv = any(issubclass(x.category, ConvergenceWarning) for x in w)
    p = m.predict(Xb)
    return {"representation": rep, "training_set": tset, "model": mn, "n_train": int(len(yt)),
            "acc": float(accuracy_score(yte, p)), "f1_attack": float(f1_score(yte, p, pos_label=0)),
            "acc_test21": float(accuracy_score(yte[hard], p[hard])), "seconds": time.perf_counter() - t0,
            "not_converged": bool(conv)}


# rough relative cost, used only to start the slowest fits first
COST = {"k-NN (k=1)": 10, "k-NN (k=5)": 10, "AdaBoost": 6, "MLP (100-50)": 6, "Random forest": 5, "Extra trees": 5,
        "RBM features + logistic": 4, "Gradient boosting": 3, "Logistic regression": 1, "Linear SVM": 1,
        "Decision tree": 1, "Naive Bayes": 0}


def main():
    OUT.mkdir(exist_ok=True)
    logfile = OUT / f"benchmark_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    log(f"NSL-KDD benchmark on the official test set. Started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", logfile)
    for f in (TRAIN_FILE, TEST_FILE):
        if not (HERE / f).exists():
            log(f"ERROR: cannot find {HERE / f}. Put the NSL-KDD files in a 'Dataset' folder next to this script.", logfile)
            return
    tr, te, ytr, yte, hard, num = load_all()
    log(f"KDDTrain+ {len(tr)} records, KDDTest+ {len(te)} records ({hard.sum()} in KDDTest-21); "
        f"{100 * (~te['class'].isin(tr['class'].unique())).mean():.1f}% of test records are attack types absent from training", logfile)
    log("Moro & Prati report on KDDTest+: classical RBM 0.906-0.938, D-Wave RBM 0.937 (accuracy)", logfile)
    log("building representations ...", logfile)
    reps = build_representations(tr, te, num)
    res = {"ceiling": {}, "models": [], "settings": {"REPRESENTATIONS": REPRESENTATIONS, "TRAINING_SETS": TRAINING_SETS,
                                                       "N_TREES": N_TREES, "MLP_MAX_ITER": MLP_MAX_ITER, "SEED": SEED}}

    log("=" * 100, logfile)
    log("PART 1. CEILING: the best accuracy ANY classifier using this representation could reach, even if tuned on the test labels", logfile)
    log("-" * 100, logfile)
    for name, (Xa, Xb, binary) in reps.items():
        Xc = Xb if binary else np.round(Xb, 6)
        c_all, n_codes = ceiling(Xc, yte)
        c_hard, _ = ceiling(Xc, yte, hard)
        res["ceiling"][name] = {"KDDTest+": c_all, "KDDTest-21": c_hard, "distinct_test_codes": n_codes,
                                "n_features": int(Xb.shape[1])}
        log(f"   {name:<16} ({Xb.shape[1]:>3} features)   KDDTest+ ceiling {c_all:.4f}   KDDTest-21 ceiling {c_hard:.4f}   "
            f"({n_codes} distinct test codes)", logfile)
    save_json(res, OUT / "nsl_benchmark.json")

    tasks = [(rep, tset, mn) for rep, (_, _, binary) in reps.items() for tset in TRAINING_SETS for mn in models(binary)]
    tasks.sort(key=lambda t: -COST.get(t[2], 1))
    n_cpu = os.cpu_count() or 1
    workers = n_cpu if N_WORKERS == "auto" else int(N_WORKERS)
    workers = max(1, min(workers, len(tasks)))
    log("=" * 100, logfile)
    log("PART 2. STANDARD CLASSIFIERS (trained on KDDTrain+, scored on KDDTest+ and KDDTest-21)", logfile)
    log(f"{len(tasks)} fits on {workers} parallel workers ({n_cpu} logical CPUs). Each worker first loads the data "
        f"(a few seconds); then one line is printed as each fit finishes.", logfile)
    log(f"   {'representation':<16}{'training':<10}{'model':<26}{'acc Test+':>10}{'F1 att.':>9}{'Test-21':>9}{'time':>8}   progress", logfile)
    t_start = time.perf_counter()

    def report(r, done):
        res["models"].append(r)
        save_json(res, OUT / "nsl_benchmark.json")
        el = time.perf_counter() - t_start
        log(f"   {r['representation']:<16}{r['training_set']:<10}{r['model']:<26}{r['acc']:>10.4f}{r['f1_attack']:>9.4f}"
            f"{r['acc_test21']:>9.4f}{r['seconds']:>7.0f}s   {done}/{len(tasks)} done, elapsed {hms(el)}"
            f"{'   (did not fully converge)' if r['not_converged'] else ''}", logfile)

    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=init_worker) as pool:
            for i, r in enumerate(pool.imap_unordered(fit_one, tasks, chunksize=1), 1):
                report(r, i)
    else:
        init_worker()
        for i, t in enumerate(tasks, 1):
            report(fit_one(t), i)

    log("=" * 100, logfile)
    log("PART 3. SUMMARY", logfile)
    A = pd.DataFrame(res["models"])
    for rep in reps:
        for tset in TRAINING_SETS:
            s_ = A[(A["representation"] == rep) & (A["training_set"] == tset)].sort_values("acc", ascending=False)
            log(f"   {rep}, {tset}: " + ", ".join(f"{m} {a:.3f}" for m, a in zip(s_["model"], s_["acc"])), logfile)
    b = A.loc[A["acc"].idxmax()]
    log("-" * 100, logfile)
    log(f"   {len(A)} classifier / representation / training-set combinations", logfile)
    log(f"   accuracy on KDDTest+: min {A['acc'].min():.4f}, median {A['acc'].median():.4f}, "
        f"90th percentile {A['acc'].quantile(.9):.4f}, max {A['acc'].max():.4f}", logfile)
    log(f"   best (selected on the test set, i.e. optimistic): {b['model']} on {b['representation']}, "
        f"{b['training_set']} training set: accuracy {b['acc']:.4f}, F1 {b['f1_attack']:.4f}", logfile)
    for thr in (0.93, 0.90, 0.85):
        log(f"   combinations reaching {thr:.2f}: {(A['acc'] >= thr).sum()}", logfile)
    for rep in reps:
        s_ = A[A["representation"] == rep]
        log(f"   best on {rep:<16}: {s_['acc'].max():.4f}   (ceiling {res['ceiling'][rep]['KDDTest+']:.4f})", logfile)
    log(f"   finished in {hms(time.perf_counter() - t_start)}. Saved: results/nsl_benchmark.json and {logfile.name}", logfile)


if __name__ == "__main__":     # required on Windows for parallel workers
    main()
