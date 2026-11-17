# CP-PINN-MIM

Code, data and results for the manuscript *A parameterized coupled physics-informed neural network (CP-PINN) for mobile-immobile contaminant transport in porous media* (Akhilesh Paswan, School of Civil and Environmental Engineering, IIT Mandi).

The repository contains everything needed to repeat the calculations of the paper: the finite-difference solver and the script that generated the 50-case data set, the single data file, the CP-PINN training code, the baseline models, the five-seed study, the raw results of every run, and the tables with the parameters, normalization statistics, hyperparameters and random seeds.

## 1. Set-up
Python 3.10 or newer is required. The results were produced with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, PyTorch 2.15 (CPU), scikit-learn 1.9 and Matplotlib 3.11. No GPU is needed.
```
git clone https://github.com/AkhileshP-coder/CP-PINN-MIM.git
cd CP-PINN-MIM
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Run every command below from the folder that is named in front of it. The scripts find the data file in `data/` by themselves.

## 2. Folder content
| Folder | Content |
|---|---|
| `data/` | `BTC_spatial_Cm_Cim.csv`: the single data file used by all scripts. It holds, for the 50 cases, the eight transport parameters and the finite-difference mobile (`Cm`) and immobile (`Cim`) concentrations at 11 positions (x/L = 0, 0.1, ..., 1) and at t = 0-25.3 d (0.1 d step, 254 time points per curve). The inlet concentration is C0 = 10 mg/L. The breakthrough curve (BTC) of a case is its outlet slice `x_over_L == 1` (`Cm`). `BTC_original_10cases.csv` holds the ten curves of the original data file and is used only for the solver comparison. |
| `fd_solver/` | Crank-Nicolson finite-difference solver (`fd_mim.py`), the data generator (`generate_dataset.py`: Latin-hypercube sampling of the eight parameters, seed 20261115), the grid and time-step convergence script and the comparison with the ten original curves. |
| `pinn/` | `cp_pinn_train.py` (CP-PINN training with seeds 1234, 1334 and 2024, loss-weight sweep, test export, inference timing; its output of the reported run is stored in `CP_PINN_COMPREHENSIVE_20261115_122349/`) and `cp_pinn_seed_study.py` (five-seed study: Fig. 7, loss-spike statistics, early-stopping rules; it reads `benchmark/results`). |
| `benchmark/` | Baseline models (CP-PINN, PINN without PDE loss, data-only network, single-branch network, physics-only PINN) in `bench.py`, the Gaussian-process baseline (`gp.py`, `gp_learning_curve.py`), job lists, summaries, inference and solver timing, final PDE residual, the comparison figures (`make_comparison_figures.py`). `results/` holds one result file per run. |
| `tables/` | Parameters of the 50 cases and their split (`parameters_50_cases.csv`, Table S1), hyperparameters, seeds and split (`hyperparameters.csv`, `seeds_and_split.csv`, `data_split.json`), normalization statistics (`param_stats.json`), test metrics and predictions of the main run, per-run benchmark results (`benchmark_results_per_run.csv`), benchmark summary, seed-study and early-stopping tables, GP learning curve, PDE residual, solver convergence and timing. |

## 3. Reproducing the results
### 3.1 Quick check (a few minutes)
```
cd fd_solver && python verify_convergence.py          # convergence of the solver (Table A4 of the paper)
cd ../benchmark && python bench.py cp 1234 20          # a 20-epoch test run of the CP-PINN
cd ../pinn && python cp_pinn_seed_study.py             # Fig. 7 and the early-stopping tables, from the stored histories
```
The last command uses the stored result files and does not retrain anything.

### 3.2 Data set (optional; the file is already included)
```
cd fd_solver && python generate_dataset.py             # writes ../data/BTC_spatial_Cm_Cim.csv (50 cases)
```

### 3.3 Main run (Figs. 5, 6, 8 and 9, Tables 4-6 and 8, Table S2)
```
cd pinn && python cp_pinn_train.py
```
It trains the CP-PINN on the 30 training cases (case1-case30) for 2000 epochs with the seeds 1234, 1334 and 2024, runs the loss-weight sweep, evaluates the 20 test cases (case31-case50), and times the inference. All output goes into a new folder `CP_PINN_COMPREHENSIVE_<date>_<time>`. `QUICK_TEST=1` gives a short test run. Reported for the main run (seed 1234, all 20 test cases pooled, 5,080 observations): R2 = 0.9941 and RMSE = 0.0218 in units of Cm/C0. The five test cases shown in the figures (TC-1 to TC-5 = case33, case35, case36, case42, case47) are the five with the highest R2; the metrics in the text use all 20 test cases (`tables/Table_test_metrics_per_case.csv`). The script also writes a plot of the observed and predicted curves of the five displayed cases; it is not used in the manuscript (Fig. 10 shows the same cases together with the other models).

### 3.4 Baselines and five-seed study (Figs. 7, 10 and 11, Tables 7-10)
```
cd benchmark
python bench.py <variant> <seed> <epochs> [weight=multiplier ...]   # one run; variants: cp, nopde, dataonly, single, physonly
./runjobs.sh jobs_all.txt                                           # every run of the paper, two at a time (see also jobs_*.txt)
python gp.py                                                        # Gaussian-process baseline
python gp_learning_curve.py                                         # GP accuracy against the number of training cases
python summarize.py                                                 # results/*.json -> summary_five_seed.json (numbers of Tables 6-10)
python export_tables.py                                             # per-run table in ../tables
python make_comparison_figures.py <shown_test_cases.csv>            # Figs. 10 and 11
cd ../pinn && python cp_pinn_seed_study.py                          # Fig. 7, early-stopping rules
```
The seeds are 1234, 1334, 2024, 1434 and 1534 for every model. One CP-PINN run of 2000 epochs takes about 8 minutes on one CPU core (more when several runs share the CPU).

### 3.5 Immobile concentration and extrapolation (Figs. 12 and 13, Section 3.6, Tables 11, S3-S5)
```
cd benchmark
python immobile_validation.py     # mobile and immobile concentrations inside the column against the finite-difference solution (tables/interior_field_per_case.csv)
python immobile_weight_check.py  # R2 of the immobile and mobile fields for the loss-weight runs already available (tables/immobile_weight_check.csv, Section 3.5.3)
python extrapolation_plots.py     # Fig. 13: outlet curves and mobile profiles of extrapolation cases (training box widened by 25 %) with R2 above 0.95 (tables/extrapolation_plot_cases.csv)
python extrapolation_test.py      # the five trained CP-PINNs on new finite-difference cases outside the training ranges, no retraining (tables/extrapolation_*.csv)
```
`extrapolation_test.py` needs the trained networks `benchmark/results/cp_s<seed>_e2000.pt` (included). In `immobile_validation.py` the two cases of Fig. 12 are the two with the highest R2 of the immobile concentration (best cases, not typical ones); the values of all twenty test cases are in the csv file.

### 3.6 Solver, timing and PDE residual
```
cd fd_solver && python verify_convergence.py && python compare_with_original_curves.py
cd ../benchmark && python timing.py && python fd_timing.py && python resid.py
```

## 4. Data and settings
* All 50 breakthrough curves are synthetic outputs of the finite-difference solver (801 nodes, dt = 0.005 d, pulse duration 7.708 d). They are not experimental data. The eight parameters (alpha, theta_m, theta_im, f, rho, kd, Dxx, qx) were sampled by Latin-hypercube sampling inside the ranges of the paper.
* Training cases: case1-case30. Test cases: case31-case50. Case n in the paper is `case<n>` in the files.
* Column `qx` is the Darcy flux in m/d (the pore-water velocity is qx/theta_m); `Dxx` is the dispersion coefficient in m2/d.
* Network and training: two encoder-decoder branches (3 x 64 Tanh encoders, 32-unit Tanh decoders, sigmoid output), orthogonal initialization (gain 0.8), Adam with learning rate 1e-3, gradient-norm clipping 0.1, 2000 epochs (500 for the weight sweeps), one optimizer step per training case (30 per epoch), loss weights lambda_data 10, lambda_PDE 1, lambda_IC = lambda_BCin = lambda_BCout = 0.5, collocation points 100 interior, 50 inlet, 50 outlet and 50 initial-condition per case and epoch. Parameters are standardized with the mean and standard deviation of the training cases (`tables/param_stats.json`) and the concentration is divided by C0 = 10 mg/L. Only the outlet breakthrough curve is used as data.
* Two implementations of the same network, loss and optimizer are included: `pinn/cp_pinn_train.py` uses x and t directly and a fixed case order; `benchmark/bench.py` scales x and t by L and T and visits the training cases in random order in every epoch. The five-seed study, Table 7 and Figs. 7, 11 and 12 use the benchmark implementation; the main run uses `pinn/cp_pinn_train.py`.

## 5. Notes
* Repeated runs differ slightly, because the network weights and the collocation points are random (the seed fixes both) and because floating-point results depend on the hardware and on the PyTorch version. The paper reports the spread over seeds, so small differences from the stored values are expected.
* `benchmark/results/cp_s<seed>_e2000.json` hold the final metrics and the loss and test R2 of every epoch (`loss_full`, `r2_full`) used for Fig. 7 and the early-stopping analysis.
* The physics-only PINN (no concentration data) did not reproduce the breakthrough curves accurately; its results are in `benchmark/results/physonly_*`.
* The loss-weight sweeps use three seeds (1234, 1334, 2024), the other comparisons five seeds. The Gaussian-process baseline uses every eighth time point of the training curves.
* Timings were measured on a shared two-core CPU and depend on the machine.
* Figures are not stored in the repository; the scripts write them to a `figures/` folder or to the output folder of the run.
* Running `bench.py` adds a result file to `benchmark/results`; `run_seeds.sh` overwrites the stored files of the five seeds. Copy the folder first if you want to keep the original results.

## 6. License
MIT (see `LICENSE`).

### Comparison with the experimental breakthrough curve (Fig. 14)
`benchmark/experimental_comparison.py` computes the breakthrough curves for the parameters of Table 3 (k = 0.426 and 0.616) with the finite-difference solver and the five trained CP-PINNs (no retraining) and compares them with the experimental points of Fig. 4 (`data/experimental_fig4_digitized.csv`, approximate values read from Fig. 4, `data/fig4_source_image.jpeg`). Output: `figures/fig14_experimental_comparison.png`, `tables/experimental_comparison.json`.
