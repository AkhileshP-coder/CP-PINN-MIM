"""Figs. 11 and 12 (comparison with the baseline models) from benchmark/results.
usage: python make_comparison_figures.py <shown_test_cases.csv>   (the five test cases written by pinn/cp_pinn_train.py)"""
import os; os.makedirs("../figures", exist_ok=True)
import json, glob, sys, textwrap, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bench import make_cases, TT
matplotlib.rcParams['font.family'] = 'DejaVu Serif'; matplotlib.rcParams['font.size'] = 10
NAMES = {'cp': 'CP-PINN', 'nopde': 'PINN without PDE loss', 'single': 'Single-branch network', 'dataonly': 'Data-only network', 'physonly': 'Physics-only PINN', 'gp': 'Gaussian process'}
COL = {'cp': '#d62728', 'nopde': '#ff7f0e', 'single': '#2ca02c', 'dataonly': '#9467bd', 'physonly': '#8c564b', 'gp': '#17becf'}
show = pd.read_csv(sys.argv[1])['case'].tolist()
R = [json.load(open(f)) for f in glob.glob('results/*_s*_e2000.json') if not f.endswith('_norm.json')]
R = [r for r in R if not r['mult']]
gp = json.load(open('results/gp.json')); cs = {c['name']: c for c in make_cases()}
def pred(v, c):
    if v == 'gp': return np.array(next(x['pred'] for x in gp['per_case'] if x['case'] == c))
    r = next(r for r in R if r['variant'] == v and r['seed'] == 1234); return np.array(next(x['pred'] for x in r['per_case'] if x['case'] == c))
fig, ax = plt.subplots(3, 2, figsize=(7.6, 9.2)); ax = ax.ravel()
for k, c in enumerate(show):
    a = ax[k]; a.plot(TT, cs[c]['C'], 'k.', ms=3, label='Reference (finite difference)')
    for v in ['cp', 'nopde', 'single', 'dataonly', 'gp', 'physonly']:
        a.plot(TT, pred(v, c), '-', color=COL[v], lw=1.5 if v == 'cp' else 1.0, label=NAMES[v])
    a.set_ylim(-0.05, 1.15); a.set_xlabel('Time (d)'); a.set_ylabel('Cm/C0'); a.grid(alpha=.3); a.set_title(f'({"abcde"[k]}) TC-{k+1}', loc='left', fontweight='bold')
h, l = ax[0].get_legend_handles_labels(); ax[5].axis('off'); ax[5].legend(h, l, loc='center', frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig('../figures/fig11.png', dpi=220); plt.close()
S = json.load(open('summary_five_seed.json'))
keys = ['gp', 'single', 'nopde', 'cp', 'dataonly', 'physonly']
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.9))
rng = np.random.RandomState(0)
for a, (m, lab) in zip(ax, [('R2', 'Test R²'), ('RMSE', 'Test RMSE (Cm/C0)')]):
    for i, k in enumerate(keys):
        if k == 'gp': val, sd = S['gp'][m], 0.0
        else: val, sd = S[k][m]
        a.bar(i, val, yerr=sd, color=COL[k], capsize=4, alpha=.85)
        for r in [r for r in R if r['variant'] == k]: a.plot(i + rng.uniform(-.15, .15), r[m], 'k.', ms=3)
    a.set_xticks(range(len(keys))); a.set_xticklabels([textwrap.fill(NAMES[k], 11) for k in keys], fontsize=6.5); a.set_ylabel(lab); a.grid(axis='y', alpha=.3)
    a.set_title('(a)' if m == 'R2' else '(b)', loc='left', fontweight='bold')
lo = min(S[k]['R2'][0] - S[k]['R2'][1] for k in keys if k not in ('physonly', 'gp'))
ax[0].set_ylim(max(-0.3, lo - 0.02), 1.002)
ax[0].text(5, ax[0].get_ylim()[0]+0.0015, f"R² = {S['physonly']['R2'][0]:.2f}\n(below axis)", ha='center', fontsize=7)
plt.tight_layout(); plt.savefig('../figures/fig12.png', dpi=220); plt.close()
print('figures written for', show)
