"""Comparison of the CP-PINN mobile and immobile concentrations inside the column with the finite-difference solution (20 test cases, five seeds).
Writes ../tables/interior_field_per_case.csv and ../figures/fig12_interior_fields.png. Uses benchmark/results/cp_s<seed>_e2000.json."""
import json, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SEEDS = [1234, 1334, 2024, 1434, 1534]; XF = [0.25, 0.5, 0.75]; TF = np.round(np.arange(0.5, 25.3001, 0.5), 1)
R = {s: json.load(open(f'results/cp_s{s}_e2000.json')) for s in SEEDS}
r2 = lambda o, p: 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()
rows = []
for i, c in enumerate(R[1234]['interior']):
    d = dict(case=c['case'])
    for s in SEEDS:
        e = R[s]['interior'][i]; d[f'R2_mobile_s{s}'] = e['R2_m']; d[f'R2_immobile_s{s}'] = e['R2_im']
    d['R2_mobile_mean'] = np.mean([d[f'R2_mobile_s{s}'] for s in SEEDS]); d['R2_immobile_mean'] = np.mean([d[f'R2_immobile_s{s}'] for s in SEEDS])
    d['NRMSE_immobile_s1234'] = c['RMSE_im']; d['peak_immobile_fd'] = max(c['oi']); d['peak_immobile_pinn_s1234'] = max(c['pi'])
    rows.append(d)
df = pd.DataFrame(rows); os.makedirs('../tables', exist_ok=True); df.to_csv('../tables/interior_field_per_case.csv', index=False)
print(df[['case', 'R2_mobile_mean', 'R2_immobile_mean']].round(3).to_string())
# figure: seed 1234; the two test cases with the highest R2 of the immobile field (best cases, not typical ones; all twenty cases are in interior_field_per_case.csv)
order = df.sort_values('R2_immobile_s1234')['case'].tolist(); pick = order[::-1][:2]
lab = ['highest', '2nd highest']
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
cols = {0.25: '#1b9e77', 0.5: '#d95f02', 0.75: '#7570b3'}
for k, (cn, lb) in enumerate(zip(pick, lab)):
    e = next(x for x in R[1234]['interior'] if x['case'] == cn)
    for j, x in enumerate(XF):
        sl = slice(j * len(TF), (j + 1) * len(TF))
        ax[k].plot(TF, np.array(e['oi'])[sl], '-', c=cols[x], lw=2, label=f'FD, x/L = {x}'); ax[k].plot(TF, np.array(e['pi'])[sl], '--', c=cols[x], lw=2, label=f'CP-PINN, x/L = {x}')
    ax[k].set_title(f"{cn.replace('case', 'Case ')} ({lb} immobile R$^2$ = {e['R2_im']:.2f})", fontsize=11)
    ax[k].set_ylabel('Immobile concentration $C_{im}/C_0$'); ax[k].set_xlabel('Time (d)'); ax[k].grid(alpha=.3)
    ax[k].text(0.02, 0.93, ['(a)', '(b)'][k], transform=ax[k].transAxes, fontweight='bold')
ax[0].legend(fontsize=8, ncol=1, frameon=False, loc='upper right')
os.makedirs('../figures', exist_ok=True); fig.savefig('../figures/fig12_interior_fields.png', dpi=200)
fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
for a_, key, nm in ((ax[0], ('om', 'pm'), 'Mobile'), (ax[1], ('oi', 'pi'), 'Immobile')):
    o = np.concatenate([e[key[0]] for e in R[1234]['interior']]); p = np.concatenate([e[key[1]] for e in R[1234]['interior']])
    a_.plot(o, p, '.', ms=3, alpha=.4, c='#1f77b4'); m = max(o.max(), p.max()); a_.plot([0, m], [0, m], 'k--'); a_.set_xlabel(f'{nm} concentration, finite-difference ($C/C_0$)'); a_.set_ylabel(f'{nm} concentration, CP-PINN ($C/C_0$)')
    a_.set_title(f'{nm}: R$^2$ = {r2(o, p):.3f} (n = {len(o)})'); a_.grid(alpha=.3); a_.set_aspect('equal')
fig.savefig('../figures/fig12_scatter.png', dpi=200); print(pick)
