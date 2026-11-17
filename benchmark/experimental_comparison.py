"""Fig. 14: breakthrough curves for the parameters of Table 3 (experiment of van Genuchten et al., 1977, as in Fig. 4) computed with the finite-difference solver and with the trained CP-PINN (no retraining), compared with the experimental points.
The experimental points were digitized from the marker positions of Fig. 4 (../data/fig4_source_image.jpeg; ../data/experimental_fig4_digitized.csv), so they carry a digitization error of about 0.01 in C/C0 and 0.03 in pore volume.
Pore volume PV = qx*t/((theta_m + theta_im)*L). Writes ../tables/experimental_comparison.csv/json and ../figures/fig14_experimental_comparison.png."""
import json, numpy as np, pandas as pd, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bench import Net, make_cases, PK, L, T, TP, TT
from fd_mim import solve_mim
torch.set_num_threads(1)
cases = make_cases(); tr = [c for c in cases if c['split'] == 'train']; P = np.array([[c[k] for k in PK] for c in tr]); lo, hi = P.min(0), P.max(0)
base = dict(alpha=0.13, theta_m=0.413, theta_im=0.066, f=0.376, rho=1.361, D_xx=9.1e-4, qx=0.0459)   # Table 3 (D in cm2/d -> m2/d, q in cm/d -> m/d)
ex = pd.read_csv('../data/experimental_fig4_digitized.csv'); ex['C_rel'] = ex.C_rel.clip(lower=0)
r2 = lambda o, p: float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()); rmse = lambda o, p: float(np.sqrt(((o - p) ** 2).mean()))
SEEDS = [1234, 1334, 2024, 1434, 1534]; nets = {}
for s in SEEDS:
    n = Net(); n.load_state_dict(torch.load(f'results/cp_s{s}_e2000.pt')); n.eval(); nm = json.load(open(f'results/cp_s{s}_e2000_norm.json')); nets[s] = (n, np.array(nm['mu']), np.array(nm['sd']))
res = {}; fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
for a, kd, lab in zip(ax, (0.426, 0.616), 'ab'):
    p = np.array([base[k] if k != 'kd' else kd for k in PK]); pv = lambda t: base['qx'] * t / ((base['theta_m'] + base['theta_im']) * L)
    x, t, Fm, Fi = solve_mim(*p, L, TP, T, 801, 0.005, 'U', 1.0, full=True); fd = Fm[np.round(TT / 0.005).astype(int), 800]
    preds = []
    for s in SEEDS:
        n, mu, sd = nets[s]; z = torch.tensor((p - mu) / (sd + 1e-10), dtype=torch.float32)
        inp = torch.cat([torch.full((len(TT), 1), 1.0), torch.tensor(TT / T, dtype=torch.float32).reshape(-1, 1), z.expand(len(TT), 8)], 1)
        with torch.no_grad(): preds.append(n(inp)[0].squeeze().numpy())
    preds = np.array(preds); pv_t = pv(TT); pe = np.interp(ex.PV, pv_t, preds[0]); fe = np.interp(ex.PV, pv_t, fd)
    m = ex.PV.between(0.0, pv_t.max())
    rr = dict(kd=kd, outside_training_box=[PK[i] for i in range(8) if not lo[i] <= p[i] <= hi[i]], kd_excess_pct=float(max(0, (kd - hi[5]) / hi[5] * 100)),
              R2_FD_vs_exp=r2(ex.C_rel[m], fe[m]), RMSE_FD_vs_exp=rmse(ex.C_rel[m], fe[m]),
              R2_CPPINN_vs_exp_seed1234=r2(ex.C_rel[m], pe[m]), RMSE_CPPINN_vs_exp_seed1234=rmse(ex.C_rel[m], pe[m]),
              R2_CPPINN_vs_exp_5seeds=[float(np.mean([r2(ex.C_rel[m], np.interp(ex.PV, pv_t, q)[m]) for q in preds])), float(np.std([r2(ex.C_rel[m], np.interp(ex.PV, pv_t, q)[m]) for q in preds]))],
              R2_CPPINN_vs_FD_seed1234=r2(fd, preds[0]), R2_CPPINN_vs_FD_5seeds=[float(np.mean([r2(fd, q) for q in preds])), float(np.std([r2(fd, q) for q in preds]))],
              peak_FD=float(fd.max()), peak_CPPINN=float(preds[0].max()), peak_PV_FD=float(pv_t[fd.argmax()]), peak_PV_CPPINN=float(pv_t[preds[0].argmax()]))
    res[str(kd)] = rr
    a.plot(ex.PV, ex.C_rel, 'v', mfc='none', mec='#1f4eff', ms=6, label='Experimental')
    a.plot(pv_t, fd, 'k-', lw=2.2, label='Finite-difference solver'); a.fill_between(pv_t, preds.min(0), preds.max(0), color='#e41a1c', alpha=.18, lw=0); a.plot(pv_t, preds[0], 'r--', lw=2, label='CP-PINN (seed 1234; band: five seeds)')
    a.set_xlim(0, 8); a.set_ylim(0, 1.0); a.set_xlabel('Pore volume'); a.set_ylabel('Relative concentration, $C_m/C_0$'); a.grid(alpha=.3)
    a.set_title(f'({lab}) $k$ = {kd}: R$^2$ vs. exp. FD {rr["R2_FD_vs_exp"]:.3f}, CP-PINN {rr["R2_CPPINN_vs_exp_seed1234"]:.3f}', fontsize=10)
ax[0].legend(frameon=False, fontsize=8.5, loc='upper right'); fig.savefig('../figures/fig14_experimental_comparison.png', dpi=200)
json.dump(res, open('../tables/experimental_comparison.json', 'w'), indent=1); print(json.dumps(res, indent=1))
