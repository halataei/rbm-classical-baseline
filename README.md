# Classical baseline for the quantum-annealing RBM of Moro & Prati (2023)

Code and results for the Matters Arising

> S. M. H. Halataei, M. Ramezani, A. Bahrampour and A. Bahrampour,
> *A standard classical baseline and the cost of training challenge a reported quantum annealing speed-up*,
> arising from L. Moro & E. Prati, *Commun. Phys.* **6**, 269 (2023), https://doi.org/10.1038/s42005-023-01390-y

Every number in the article and its Supplementary Information is produced by the scripts below.
Each script has its settings at the top and can be run with no arguments (written for VS Code on Windows; they also run on Linux).

## Requirements
Python 3.12 or later, `numpy`, `pandas`, `scikit-learn`, `matplotlib`; `torch` with CUDA for `rbm_gpu_vscode.py`; `pyarrow` (optional, faster CSV reading).

## Data (not included)
* **NSL-KDD**: `KDDTrain+.txt` and `KDDTest+.txt` from the Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/nsl.html), placed in `Dataset/`.
* **CSE-CIC-IDS2018**: the ten CSV files of "Processed Traffic Data for ML Algorithms", downloaded into `CSE/` with `scripts/download_cse.py` (public AWS bucket `cse-cic-ids2018`).

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
MIT.
