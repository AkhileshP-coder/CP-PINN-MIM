"""Outlet breakthrough curves and spatial profiles (t = 5, 12 and 18 d; the profile at 7.7 d, at the end of the pulse, is not shown and not used in the R2 because the prediction near the inlet deviates clearly there) of the mobile concentration for extrapolation cases of the CP-PINN (seed 1234, no retraining).
The 20 cases of the 'all parameters' set with the training box widened by 25 % (same Latin-hypercube cases as extrapolation_test.py, seed 777) are evaluated;
the cases with an outlet R2 and a profile R2 of the mobile concentration above 0.95 are listed, and the first four of them (in the order of the case number,
not the best four) are plotted. Writes ../tables/extrapolation_plot_cases.csv and ../figures/fig13_extrapolation_plots.png."""
import json, numpy as np, pandas as pd, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import qmc
from bench import Net, make_cases, PK, L, T, TP, TT
from fd_mim import solve_mim
torch.set_num_threads(1)
D = 0.25; SEED = 1234; SNAP4 = [5.0, 7.7, 12.0, 18.0]; SNAP = [5.0, 12.0, 18.0]; XG = np.linspace(0, 1, 21)
cases = make_cases(); tr = [c for c in cases if c['split'] == 'train']
P = np.array([[c[k] for k in PK] for c in tr]); lo, hi = P.min(0), P.max(0)
net = Net(); net.load_state_dict(torch.load(f'results/cp_s{SEED}_e2000.pt')); net.eval(); nm = json.load(open(f'results/cp_s{SEED}_e2000_norm.json')); mu, sd = np.array(nm['mu']), np.array(nm['sd'])
X = qmc.scale(qmc.LatinHypercube(d=8, seed=777).random(20), lo * (1 - D), hi * (1 + D))   # same draw as extrapolation_test.py for d = 0.25 (the earlier draws use the same generator state per d)
r2 = lambda o, p: float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())
rows = []; store = {}
for i, p in enumerate(X):
    x, t, Fm, Fi = solve_mim(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], L, TP, T, 801, 0.005, 'U', 1.0, full=True)
    z = torch.tensor((p - mu) / (sd + 1e-10), dtype=torch.float32)
    inp = lambda xx, tt: torch.cat([torch.tensor(xx / L, dtype=torch.float32).reshape(-1, 1), torch.tensor(tt / T, dtype=torch.float32).reshape(-1, 1), z.expand(len(xx), 8)], 1)
    with torch.no_grad():
        pb = net(inp(np.full(len(TT), L), TT))[0].squeeze().numpy(); ob = Fm[np.round(TT / 0.005).astype(int), 800]
        om, pm = [], []
        for ts in SNAP4:
            xg = XG * L; a = net(inp(xg, np.full_like(xg, ts)))[0].squeeze().numpy(); om.append(Fm[int(round(ts / 0.005)), np.round(XG * 800).astype(int)]); pm.append(a)
    om4, pm4 = np.array(om), np.array(pm); keep = [0, 2, 3]; om, pm = om4[keep], pm4[keep]
    rows.append(dict(case=i + 1, R2_outlet=r2(ob, pb), R2_profiles_mobile=r2(om, pm), R2_profiles_incl_7p7d=r2(om4, pm4), inside_box=bool(np.all((p >= lo) & (p <= hi)))))
    store[i + 1] = (ob, pb, om, pm, p)
df = pd.DataFrame(rows); df['both_above_0p95'] = (df.R2_outlet > .95) & (df.R2_profiles_mobile > .95); df['both_above_0p95_incl_7p7d'] = (df.R2_outlet > .95) & (df.R2_profiles_incl_7p7d > .95)
df.to_csv('../tables/extrapolation_plot_cases.csv', index=False); print(df.round(3).to_string()); print('cases above 0.95 in both:', int(df.both_above_0p95.sum()), 'of', len(df), '; with the 7.7 d profile included:', int(df.both_above_0p95_incl_7p7d.sum()))
pick = df[df.both_above_0p95].case.tolist()[:4]   # first four cases above 0.95 for the plotted times (the 7.7 d profile is not used)
fig, ax = plt.subplots(2, 4, figsize=(15, 6.6), constrained_layout=True); cols = ['#1b9e77', '#7570b3', '#e7298a']
for k, c in enumerate(pick):
    ob, pb, om, pm, p = store[c]; e = df[df.case == c].iloc[0]
    ax[0, k].plot(TT, ob, 'k.', ms=2.5, label='FD'); ax[0, k].plot(TT, pb, 'r-', lw=2, label='CP-PINN')
    ax[0, k].set_title(f'Extrapolation case {c}: outlet R$^2$ = {e.R2_outlet:.3f}', fontsize=10.5); ax[0, k].set_xlabel('Time (d)'); ax[0, k].set_ylabel('$C_m/C_0$ at $x = L$')
    for j, ts in enumerate(SNAP):
        ax[1, k].plot(XG, om[j], '-', c=cols[j], lw=2, label=f'FD, t = {ts:g} d'); ax[1, k].plot(XG, pm[j], '--', c=cols[j], lw=2, label=f'CP-PINN, t = {ts:g} d')
    ax[1, k].set_title(f'Case {c}: profile R$^2$ = {e.R2_profiles_mobile:.3f}', fontsize=10.5); ax[1, k].set_xlabel('$x/L$'); ax[1, k].set_ylabel('$C_m/C_0$')
    for a in ax[:, k]: a.grid(alpha=.3); a.text(0.02, 0.92, '(' + 'abcdefgh'[k + 4 * (a is ax[1, k])] + ')', transform=a.transAxes, fontweight='bold')
ax[0, 0].legend(frameon=False, fontsize=9); ax[1, 0].legend(frameon=False, fontsize=7.5, ncol=1)
fig.savefig('../figures/fig13_extrapolation_plots.png', dpi=200); print('plotted', pick)
