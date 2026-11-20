r"""
Which preprocessing reproduces Moro & Prati's CSE-CIC-IDS2018 record count, M = 3,040,074?
Run it in VS Code with the "Run Python File" button. Takes a few minutes; uses one process per CSV file.

WHERE TO PUT IT
    C:\RBM\check_cse_count.py      <- this file
    C:\RBM\CSE\*.csv               <- the ten downloaded files

The paper says: duplicates and missing data were removed, 16 features were discarded, and "the datasets"
(Methods) or "the two training datasets" (Results) were balanced by under-sampling the most common class.
3,040,074 = 2 x 1,520,037, i.e. exactly balanced. This script tries 12 plausible ways of removing missing values
and duplicates and reports, for each, the number of attack records and the size M would have if
  (a) the whole dataset were balanced:             M = 2 x attacks
  (b) only the 80% training part were balanced:    M = 2 x 0.8 x attacks (approximately)
  (c) nothing were balanced:                       M = all records
Needs: numpy, pandas
"""
TARGET = 3_040_074
DATA_FOLDER = "CSE"

import os, glob, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DISCARDED = ["Fwd Seg Size Avg", "TotLen Fwd Pkts", "Bwd IAT Tot", "Bwd Pkt Len Max", "Subflow Fwd Byts",
             "Pkt Size Avg", "Bwd Seg Size Avg", "Subflow Fwd Pkts", "Bwd IAT Max", "Subflow Bwd Pkts",
             "Flow Pkts/s", "Flow IAT Mean", "Idle Max", "Active Min", "Active Max", "Pkt Len Min"]
ID_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]          # only in one file


def h(df):
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def one_file(f):
    t0 = time.perf_counter()
    try:
        df = pd.read_csv(f, engine="pyarrow")
    except Exception:
        df = pd.read_csv(f, low_memory=False)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    lab = df["Label"].astype(str).str.strip()
    real = (lab != "Label").to_numpy()                             # repeated header lines inside some files
    feats = [c for c in df.columns if c not in ("Label", "Timestamp")]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    Xv = X.to_numpy(np.float64)
    nan = np.isnan(Xv).any(1); inf = np.isinf(Xv).any(1)
    y = (lab.str.lower() == "benign").to_numpy().astype(np.int8)
    kept = [c for c in feats if c not in DISCARDED]
    ts = df["Timestamp"].astype(str).str.strip()
    Xl = X.assign(_label=lab.to_numpy())
    hashes = {
        "all columns incl. timestamp": h(Xl.assign(_ts=ts.to_numpy())),
        "all features, float64": h(Xl),
        "all features, float32": h(X.astype(np.float32).assign(_label=lab.to_numpy())),
        "62 kept features, float64": h(X[kept].assign(_label=lab.to_numpy())),
        "62 kept features, label ignored": h(X[kept]),
    }
    masks = {"drop NaN and inf": real & ~nan & ~inf, "drop NaN only (pandas dropna)": real & ~nan}
    out = {}
    for mn, m in masks.items():
        out[(mn, "no duplicate removal")] = (None, y[m])
        for hn, hv in hashes.items():
            out[(mn, hn)] = (hv[m], y[m])
    return Path(f).name, out, time.perf_counter() - t0


def main():
    files = sorted(glob.glob(str(HERE / DATA_FOLDER / "*.csv")))
    if not files:
        print(f"no CSV files in {HERE / DATA_FOLDER}"); return
    print(f"reading {len(files)} files in parallel ...", flush=True)
    from multiprocessing import Pool
    parts = {}
    t0 = time.perf_counter()
    with Pool(min(len(files), os.cpu_count() or 1)) as pool:
        for name, out, sec in pool.imap_unordered(one_file, files):
            parts[name] = out
            print(f"   {name} done ({sec:.0f} s)   {len(parts)}/{len(files)}, elapsed {time.perf_counter() - t0:.0f} s", flush=True)
    names = [Path(f).name for f in files]
    combos = list(parts[names[0]].keys())
    print("\n" + "=" * 118)
    print(f"{'missing values':<32}{'duplicates removed on':<34}{'records':>12}{'attacks':>11}"
          f"{'(a) 2 x att.':>14}{'(b) 1.6 x att.':>15}")
    print("-" * 118)
    rows = []
    for c in combos:
        ys = np.concatenate([parts[n][c][1] for n in names])
        if parts[names[0]][c][0] is None:
            keep = np.ones(len(ys), bool)
        else:
            hs = np.concatenate([parts[n][c][0] for n in names])
            keep = ~pd.Series(hs).duplicated().to_numpy()
        n = int(keep.sum()); att = int((ys[keep] == 0).sum())
        a, b = 2 * att, int(round(1.6 * att))
        rows.append((c, n, att, a, b))
        flag = "   <-- matches (a)" if a == TARGET else ("   <-- matches (c)" if n == TARGET else
               ("   <-- within 0.5% of (b)" if abs(b - TARGET) / TARGET < 0.005 else
               ("   <-- within 1% of (a)" if abs(a - TARGET) / TARGET < 0.01 else "")))
        print(f"{c[0]:<32}{c[1]:<34}{n:>12,}{att:>11,}{a:>14,}{b:>15,}{flag}", flush=True)
    print("-" * 118)
    best = min(rows, key=lambda r: min(abs(r[3] - TARGET), abs(r[4] - TARGET), abs(r[1] - TARGET)))
    print(f"Moro & Prati: M = {TARGET:,} (= 2 x {TARGET // 2:,}). Closest combination: {best[0][0]}, {best[0][1]} "
          f"(records {best[1]:,}, attacks {best[2]:,}).")


if __name__ == "__main__":
    main()
