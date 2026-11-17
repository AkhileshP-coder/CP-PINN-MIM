"""(A) Extrapolation test: trained CP-PINNs (five seeds, benchmark implementation) evaluated, without retraining, on 20 new
finite-difference cases whose parameters lie partly outside the training ranges. (B) Immobile and mobile concentration of the
twenty test cases at x/L = 0.25, 0.5, 0.75 against the finite-difference solution, per case and per position.
Writes ../tables/extrapolation_cases.csv, extrapolation_results.json, interior_per_case_position.csv and ../tables/fig13_data.json."""
import json, numpy as np, pandas as pd, torch
from scipy.stats import qmc
from fd_mim import solve_mim
from bench import Net, make_cases, PK, RNG, L, T, TP, TT
torch.set_num_threads(1)
SEEDS = [1234, 1334, 2024, 1434, 1534]
XF = np.array([0.25, 0.5, 0.75]); TF = np.round(np.arange(0.5, 25.3001, 0.5), 1)
r2 = lambda o, p: float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())
def fd(c):
    return solve_mim(c['alpha'], c['theta_m'], c['theta_im'], c['f'], c['rho'], c['kd'], c['D_xx'], c['qx'], L, TP, T, 801, 0.005, 'U', 1.0, full=True)
def load(seed):
    net = Net(); net.load_state_dict(torch.load(f'results/cp_s{seed}_e2000.pt')); net.eval()
    nm = json.load(open(f'results/cp_s{seed}_e2000_norm.json')); return net, np.array(nm['mu']), np.array(nm['sd'])
def predict(net, mu, sd, c, x, t):
    pz = torch.tensor((np.array([c[k] for k in PK]) - mu) / (sd + 1e-10), dtype=torch.float32).expand(len(x), 8)
    inp = torch.cat([torch.tensor(x / L, dtype=torch.float32).reshape(-1, 1), torch.tensor(t / T, dtype=torch.float32).reshape(-1, 1), pz], 1)
    with torch.no_grad(): a, b = net(inp)
    return a.squeeze().numpy(), b.squeeze().numpy()
def evaluate(net, mu, sd, c, Fm, Fi):
    o = Fm[np.round(TT / 0.005).astype(int), -1]
    pb, _ = predict(net, mu, sd, c, np.full(len(TT), L), TT)
    xs = np.repeat(XF, len(TF)); ts = np.tile(TF, len(XF)); xi = np.round(xs * 800).astype(int); ti = np.round(ts / 0.005).astype(int)
    pm, pi = predict(net, mu, sd, c, xs * L, ts); om, oi = Fm[ti, xi], Fi[ti, xi]
    res = dict(R2_btc=r2(o, pb), btc_obs=o, btc_pred=pb, R2_m=r2(om, pm), R2_im=r2(oi, pi))
    for j, x in enumerate(XF):
        s = slice(j * len(TF), (j + 1) * len(TF)); res[f'R2_m_{x}'] = r2(om[s], pm[s]); res[f'R2_im_{x}'] = r2(oi[s], pi[s])
    return res
models = {s: load(s) for s in SEEDS}
# ---------- (B) twenty test cases: per case and position
test = [c for c in make_cases() if c['split'] == 'test']
rows = []; store = {}
for c in test:
    _, _, Fm, Fi = fd(c)
    for s in SEEDS:
        r = evaluate(*models[s], c, Fm, Fi)
        rows.append(dict(case=c['name'], seed=s, **{k: v for k, v in r.items() if k.startswith('R2')}))
    if c['name'] in ('case33', 'case35', 'case36', 'case42', 'case47'):
        tf = np.round(np.arange(0, 25.3001, 0.25), 2); xs = np.repeat(XF, len(tf)); ts = np.tile(tf, len(XF))
        pm, pi = predict(*models[1234], c, xs * L, ts)
        xi = np.round(xs * 800).astype(int); ti = np.round(ts / 0.005).astype(int)
        store[c['name']] = dict(t=tf.tolist(), x=XF.tolist(), Cm_fd=Fm[ti, xi].tolist(), Cim_fd=Fi[ti, xi].tolist(), Cm=pm.tolist(), Cim=pi.tolist())
df = pd.DataFrame(rows); g = df.groupby('case').mean(numeric_only=True).drop(columns='seed')
g = g.loc[sorted(g.index, key=lambda s: int(s[4:]))]; g.to_csv('../tables/interior_per_case_position.csv', float_format='%.4f')
json.dump(store, open('../tables/fig13_data.json', 'w'))
print('mean over cases/seeds:', g.mean().round(3).to_dict())
# ---------- (A) extrapolation cases: ranges widened by 25 % of the width on each side; at least one parameter outside the training range
keys = list(RNG); lo = np.array([RNG[k][0] for k in keys]); hi = np.array([RNG[k][1] for k in keys]); w = hi - lo
u = qmc.LatinHypercube(d=8, seed=20261116).random(200); X = (lo - .25 * w) + u * 1.5 * w
out = [(x) for x in X if ((x < lo) | (x > hi)).any()][:20]
ecases = []
for i, x in enumerate(out):
    c = dict(zip(PK, x)); c['name'] = f'ext{i+1}'
    c['dist'] = float(np.max(np.maximum(lo - x, x - hi).clip(0) / w))   # largest excess over the training range, in units of the range width
    c['n_out'] = int(((x < lo) | (x > hi)).sum()); ecases.append(c)
rows = []; E = {}
for c in ecases:
    _, _, Fm, Fi = fd(c)
    per = [evaluate(*models[s], c, Fm, Fi) for s in SEEDS]
    E[c['name']] = dict(params={k: c[k] for k in PK}, dist=c['dist'], n_out=c['n_out'], R2_btc=[p['R2_btc'] for p in per], R2_m=[p['R2_m'] for p in per], R2_im=[p['R2_im'] for p in per],
                        peak_obs=float(per[0]['btc_obs'].max()), peak_pred=float(np.mean([p['btc_pred'].max() for p in per])))
    rows.append(dict(case=c['name'], **{k: c[k] for k in PK}, n_params_outside=c['n_out'], max_excess_over_range=c['dist'],
                     R2_btc=np.mean(E[c['name']]['R2_btc']), R2_mobile_interior=np.mean(E[c['name']]['R2_m']), R2_immobile_interior=np.mean(E[c['name']]['R2_im'])))
pd.DataFrame(rows).to_csv('../tables/extrapolation_cases.csv', index=False, float_format='%.6g')
r = pd.DataFrame(rows); print(r[['case', 'n_params_outside', 'max_excess_over_range', 'R2_btc', 'R2_mobile_interior', 'R2_immobile_interior']].round(3).to_string())
allb = np.array([E[k]['R2_btc'] for k in E]); print('extrap BTC R2 mean over cases per seed', allb.mean(0).round(4), 'min', allb.min().round(3))
json.dump(E, open('../tables/extrapolation_results.json', 'w'), indent=1)
