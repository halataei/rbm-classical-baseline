# Classical baseline for the quantum-annealing RBM of Moro & Prati (2023)

Code and results for the Matters Arising

> S. M. H. Halataei, N. Samadi, M. Ramezani, A. Bahrampour and A. Bahrampour,
> *A standard classical baseline and the cost of training challenge a reported quantum annealing speed-up*,
> arising from L. Moro & E. Prati, *Commun. Phys.* **6**, 269 (2023), https://doi.org/10.1038/s42005-023-01390-y

Every number in the article and its Supplementary Information is produced by the scripts below.
Each script has its settings at the top and can be run with no arguments (written for VS Code on Windows; they also run on Linux).

## Requirements
Python 3.12 or later, `numpy`, `pandas`, `scikit-learn`, `matplotlib`; `torch` with CUDA for `rbm_gpu_vscode.py`; `pyarrow` (optional, faster CSV reading).

## Data
* **NSL-KDD** (included): `Dataset/KDDTrain+.txt` (125,973 records) and `Dataset/KDDTest+.txt` (22,544 records), the official training and test files.
  The Canadian Institute for Cybersecurity no longer offers NSL-KDD for download (https://www.unb.ca/cic/datasets/nsl.html).
  We obtained the files from a public copy on Kaggle (https://www.kaggle.com/datasets/hassan06/nslkdd); their record and class counts match those reported for the original release (KDDTrain+: 67,343 normal, 58,630 attack; KDDTest+: 9,711 normal, 12,833 attack).
  Their MD5 checksums are in `Dataset/MD5SUMS.txt`.
  The NSL-KDD licence permits redistribution, provided the data set and the following paper are cited:
  M. Tavallaee, E. Bagheri, W. Lu and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set", *2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications*, 1–6 (IEEE, 2009).
* **CSE-CIC-IDS2018** (not included): the ten CSV files of "Processed Traffic Data for ML Algorithms", downloaded into `CSE/` with `scripts/download_cse.py` (public AWS bucket `cse-cic-ids2018`).
  Please cite: I. Sharafaldin, A. Habibi Lashkari and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization", *Proc. ICISSP*, 108–116 (2018).

## Scripts and what they produce
| Script | Produces |
|---|---|
| `mp_baseline_vscode.py` | Timing of the negative phase (Fig. 1, Supplementary Table 4); NSL-KDD RBM runs, 2,000 epochs, both read-outs (Supplementary Table 5, Supplementary Figs 1–2) |
| `nsl_benchmark_vscode.py` | 94 standard-classifier configurations on NSL-KDD and the encoding ceilings (Supplementary Table 3) |
| `cse_benchmark_vscode.py` | CSE-CIC-IDS2018 cleaning, 153-bit encoding and cache; 16 standard-classifier configurations (Supplementary Table 3) |
| `check_cse_count.py` | Record counts under twelve cleaning variants (Supplementary Note 4) |
| `rbm_gpu_vscode.py` | CSE-CIC-IDS2018 RBM runs on a GPU, 2,000 epochs, both read-outs, balanced test set (Supplementary Table 5, Supplementary Figs 1–2) |
| `make_figures.py` | All figures, from the files in `results/` |

Run `cse_benchmark_vscode.py` once before `rbm_gpu_vscode.py`: it builds the encoded data cache that the GPU script reads.

## Results
`results/` holds the logs and JSON files of the runs reported in the article:
`timing.json`, `summary_quantile.json`, `summary_width.json` and their logs (NSL-KDD RBM);
`nsl_benchmark.json`; `cse_benchmark.json`; `gpu_cse_k1_20261120_073100.json`, `gpu_cse_k10_20261120_073100.json` and their logs (CSE-CIC-IDS2018 RBM);
`gpu_cse_20261119_134857.json` (earlier 1,000-epoch CSE run, mentioned in Supplementary Note 5).

## Licence
The code is released under the MIT licence (see `LICENSE`). The NSL-KDD files in `Dataset/` are redistributed under the terms of the NSL-KDD data set, which require the citation given above.
