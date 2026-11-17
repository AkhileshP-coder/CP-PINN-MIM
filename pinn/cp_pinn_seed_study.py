"""Seed study (five seeds) of the CP-PINN (Section 3.2 and Fig. 7): loss histories, test-R2 histories and the
analysis of late-stage loss spikes and of early-stopping rules.

Step 1 (training, about 3 min per seed on one CPU thread; produces benchmark/results/cp_s<seed>_e2000.json):
    cd benchmark && ./run_seeds.sh
Step 2 (this script, reads the json files, writes figures/fig07_seed_robustness.png and
tables/seed_study_summary.csv, tables/early_stopping_rules.csv):
    python pinn/cp_pinn_seed_study.py
"""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(ROOT, 'benchmark', 'results')
SEEDS = [1434, 1534, 1234, 1334, 2024]
H = {s: json.load(open(os.path.join(RES, f'cp_s{s}_e2000.json'))) for s in SEEDS}
loss = {s: np.array(H[s]['loss_full']) for s in SEEDS}
r2 = {s: np.array(H[s]['r2_full']) for s in SEEDS}
E = len(loss[SEEDS[0]])
mv = lambda x, w: np.convolve(x, np.ones(w) / w, mode='valid')

# ---------- late-stage spikes: epochs 1501-2000 against 501-1000 -------------------------------------------
rows = []
for s in SEEDS:
    l = loss[s]; early, late = l[500:1000], l[1500:2000]
    rows.append(dict(seed=s, median_500_1000=np.median(early), median_1501_2000=np.median(late),
                     frac_gt3med_500_1000=(early > 3 * np.median(early)).mean(),
                     frac_gt3med_1501_2000=(late > 3 * np.median(late)).mean(),
                     max_over_median_1501_2000=late.max() / np.median(late),
                     test_R2_final=r2[s][-1], test_R2_sd_last200=r2[s][-200:].std(), test_R2_min_last200=r2[s][-200:].min(),
                     test_R2_mean_last200=r2[s][-200:].mean()))
T = pd.DataFrame(rows)

# ---------- early-stopping rules (no validation curve exists; rules use the training loss only) -------------
def stop_epoch(l, w=50, patience=200, rel=0.01):
    """stop when the w-epoch moving average of the training loss has not improved by more than rel for `patience` epochs"""
    m = mv(l, w); best = m[0]; last = 0
    for i in range(1, len(m)):
        if m[i] < best * (1 - rel): best = m[i]; last = i
        if i - last >= patience: return i + w - 1
    return len(l) - 1
rule_rows = []
for s in SEEDS:
    for name, ep in [('train-loss patience (w=50, 200 epochs, 1%)', stop_epoch(loss[s])),
                     ('train-loss patience (w=50, 100 epochs, 1%)', stop_epoch(loss[s], patience=100)),
                     ('fixed 1000 epochs', 999), ('fixed 1500 epochs', 1499), ('fixed 2000 epochs (reported)', 1999),
                     ('best moving-average train loss', int(np.argmin(mv(loss[s], 50))) + 49),
                     ('oracle: best test R2 (uses test data, not usable)', int(np.argmax(r2[s])))]:
        rule_rows.append(dict(rule=name, seed=s, stop_epoch=ep + 1, test_R2=r2[s][ep]))
R = pd.DataFrame(rule_rows)
S = R.groupby('rule', sort=False).agg(stop_epoch_mean=('stop_epoch', 'mean'), R2_mean=('test_R2', 'mean'),
                                      R2_sd=('test_R2', 'std'), R2_min=('test_R2', 'min'), R2_max=('test_R2', 'max')).reset_index()
# paired comparison with the final epoch (same seed): mean change of the test R2 and number of seeds that improved
fin_r2 = R[R.rule.str.startswith('fixed 2000')].set_index('seed').test_R2
pair = {}
for name in R.rule.unique():
    d_ = R[R.rule == name].set_index('seed').test_R2 - fin_r2
    pair[name] = (d_.mean(), int((d_ > 0).sum()))
S['mean_change_vs_final_epoch'] = S.rule.map(lambda n: pair[n][0]); S['seeds_improved_of_5'] = S.rule.map(lambda n: pair[n][1])
os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True); os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
T.to_csv(os.path.join(ROOT, 'tables', 'seed_study_summary.csv'), index=False)
S.to_csv(os.path.join(ROOT, 'tables', 'early_stopping_rules.csv'), index=False)
R.to_csv(os.path.join(ROOT, 'tables', 'early_stopping_rules_by_seed.csv'), index=False)
print(T.round(4).to_string()); print(S.round(4).to_string())

# ---------- Fig. 7 ----------------------------------------------------------------------------------------
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig = plt.figure(figsize=(11, 8.2)); gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.32, wspace=0.22)
ax = fig.add_subplot(gs[0, :]); ep = np.arange(1, E + 1)
cols = {1234: '#1f77b4', 1334: '#ff7f0e', 2024: '#2ca02c'}
for s in SEEDS:
    if s not in cols: ax.semilogy(ep, loss[s], color='0.55', lw=0.7, alpha=0.6)
for s in cols: ax.semilogy(ep, loss[s], color=cols[s], lw=1.0, alpha=0.9, label=f'seed = {s}')
ax.semilogy(ep, np.median(np.array([loss[s] for s in SEEDS]), axis=0), color='k', lw=2, label='median of five seeds')
ax.plot([], [], color='0.55', lw=1.2, label='two further seeds')
ax.set_xlabel('Epoch', fontweight='bold'); ax.set_ylabel('Total training loss', fontweight='bold'); ax.grid(alpha=0.3, which='both')
ax.legend(ncol=1, loc='upper right', fontsize=9, frameon=True); ax.text(0.01, 0.04, '(a)', transform=ax.transAxes, fontweight='bold')
ax = fig.add_subplot(gs[1, 0])
for s in SEEDS: ax.plot(ep, r2[s], color=cols.get(s, '0.55'), lw=0.9 if s in cols else 0.7, alpha=0.9 if s in cols else 0.6)
ax.plot(ep, np.median(np.array([r2[s] for s in SEEDS]), axis=0), color='k', lw=2)
ax.set_ylim(0.88, 1.0); ax.set_xlabel('Epoch', fontweight='bold'); ax.set_ylabel('Test R$^2$ (outlet curves)', fontweight='bold'); ax.grid(alpha=0.3)
ax.text(0.01, 0.04, '(b)', transform=ax.transAxes, fontweight='bold')
ax = fig.add_subplot(gs[1, 1]); fin = np.array([r2[s][-1] for s in SEEDS]); x = np.arange(len(SEEDS))
ax.axhspan(fin.mean() - fin.std(ddof=1), fin.mean() + fin.std(ddof=1), color='k', alpha=0.12)
ax.axhline(fin.mean(), color='k', lw=1.5)
ax.scatter(x, fin, c=[cols.get(s, '0.45') for s in SEEDS], s=45, zorder=3, edgecolor='k', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS], rotation=60, fontsize=9); ax.set_xlabel('Seed', fontweight='bold')
ax.set_ylabel('Final-epoch test R$^2$', fontweight='bold'); ax.grid(alpha=0.3); ax.text(0.04, 0.04, '(c)', transform=ax.transAxes, fontweight='bold')
ax.set_title(f'{fin.mean():.3f} ± {fin.std(ddof=1):.3f}', fontsize=10)
fig.savefig(os.path.join(ROOT, 'figures', 'fig07_seed_robustness.png'), dpi=300, bbox_inches='tight')
print('figure saved')
