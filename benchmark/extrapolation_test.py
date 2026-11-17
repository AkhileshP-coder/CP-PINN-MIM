"""Extrapolation test of the trained CP-PINNs (five seeds, benchmark implementation) on new finite-difference cases whose parameters
lie outside the training ranges. No retraining. Writes ../tables/extrapolation_*.csv and results/extrapolation.json.
 (a) one parameter at a time: the parameter is moved below the training minimum and above the training maximum by 10, 25 and 50 % (value x 0.9, 0.75, 0.5 or x 1.1, 1.25, 1.5), all other parameters stay at the mid-point of the training range;
 (b) all parameters at once: 20 Latin-hypercube cases in the training box widened on both sides by 10, 25 and 50 % (minimum x (1 - d), maximum x (1 + d))."""
import json, numpy as np, pandas as pd, torch
from scipy.stats import qmc
from bench import Net, make_cases, PK, L, T, TP, TT
from fd_mim import solve_mim
torch.set_num_threads(1)
cases = make_cases(); tr = [c for c in cases if c['split'] == 'train']
P = np.array([[c[k] for k in PK] for c in tr]); lo, hi = P.min(0), P.max(0); span = hi - lo; mid = (lo + hi) / 2
SEEDS = [1234, 1334, 2024, 1434, 1534]
nets = {}
for s in SEEDS:
    n = Net(); n.load_state_dict(torch.load(f'results/cp_s{s}_e2000.pt')); n.eval()
    nm = json.load(open(f'results/cp_s{s}_e2000_norm.json')); nets[s] = (n, np.array(nm['mu']), np.array(nm['sd']))
XF = np.array([0.25, 0.5, 0.75]); TF = np.round(np.arange(0.5, 25.3001, 0.5), 1)
def fd(p):
    return solve_mim(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], L, TP, T, 801, 0.005, 'U', 1.0, full=True)
def r2(o, p): return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())
def evaluate(p):
    x, t, Fm, Fi = fd(p); ob = Fm[np.round(TT / 0.005).astype(int), 800] / 1.0          # outlet curve (C0 = 1)
    xs = np.repeat(XF, len(TF)); ts = np.tile(TF, len(XF)); xi = np.round(xs * 800).astype(int); ti = np.round(ts / 0.005).astype(int)
    om, oi = Fm[ti, xi], Fi[ti, xi]; res = []
    for s, (net, mu, sd) in nets.items():
        z = torch.tensor((p - mu) / (sd + 1e-10), dtype=torch.float32)
        def inp(xx, tt): return torch.cat([torch.tensor(xx / L, dtype=torch.float32).reshape(-1, 1), torch.tensor(tt / T, dtype=torch.float32).reshape(-1, 1), z.expand(len(xx), 8)], 1)
        with torch.no_grad():
            pb = net(inp(np.full(len(TT), L), TT))[0].squeeze().numpy(); pm, pi = [a.squeeze().numpy() for a in net(inp(xs * L, ts))]
        res.append((s, r2(ob, pb), float(np.sqrt(((ob - pb) ** 2).mean())), r2(om, pm), r2(oi, pi), float(ob.max()), float(pb.max())))
    return res
rows = []
for j, k in enumerate(PK):
    for sign, lab in ((-1, 'below'), (1, 'above')):
        for d in (0.10, 0.25, 0.50):
            p = mid.copy(); p[j] = lo[j] * (1 - d) if sign < 0 else hi[j] * (1 + d)
            if k in ('theta_m', 'theta_im', 'f') and not (0 < p[j] < 1): continue
            for s, R2b, rm, R2m, R2i, pk, ppk in evaluate(p):
                rows.append(dict(set='one parameter', parameter=k, side=lab, extent=d, value=p[j], seed=s, R2_outlet=R2b, RMSE_outlet=rm, R2_mobile_interior=R2m, R2_immobile_interior=R2i, peak_fd=pk, peak_pinn=ppk))
for d in (0.0, 0.10, 0.25, 0.50):
    X = qmc.scale(qmc.LatinHypercube(d=8, seed=777).random(20), lo * (1 - d), hi * (1 + d))
    for i, p in enumerate(X):
        inside = bool(np.all((p >= lo) & (p <= hi)))
        for s, R2b, rm, R2m, R2i, pk, ppk in evaluate(p):
            rows.append(dict(set='all parameters', parameter='all', side='both', extent=d, value=i, seed=s, R2_outlet=R2b, RMSE_outlet=rm, R2_mobile_interior=R2m, R2_immobile_interior=R2i, peak_fd=pk, peak_pinn=ppk, inside_box=inside))
df = pd.DataFrame(rows); df.to_csv('../tables/extrapolation_all_runs.csv', index=False)
g = df.groupby(['set', 'parameter', 'side', 'extent'])[['R2_outlet', 'RMSE_outlet', 'R2_mobile_interior', 'R2_immobile_interior']].mean().reset_index()
g.to_csv('../tables/extrapolation_summary.csv', index=False)
print(g[g.set == 'all parameters'].to_string())
print(g[g.set == 'one parameter'].groupby('extent')[['R2_outlet', 'R2_mobile_interior', 'R2_immobile_interior']].agg(['mean', 'min']).to_string())
