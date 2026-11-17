"""Generate the data file ../data/BTC_spatial_Cm_Cim.csv: 50 cases (case1-case50) of synthetic mobile-immobile
transport. The eight transport parameters are drawn by Latin hypercube sampling (seed 20261115) inside the
ranges of the study; each case is solved with the Crank-Nicolson finite-difference solver (801 nodes,
dt = 0.005 d, C0 = 10 mg/L, pulse duration 7.708 d). Mobile and immobile concentrations are stored at
11 positions (x/L = 0, 0.1, ..., 1) and at t = 0, 0.1, ..., 25.3 d.
usage: python generate_dataset.py [n_cases]"""
import sys, numpy as np, pandas as pd
from scipy.stats import qmc
from fd_mim import solve_mim
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
L, TP, TMAX, NX, DT, C0 = 0.299625, 7.708, 25.3, 801, 0.005, 10.0
RNG = {'alpha': (0.1011, 0.1582), 'thetam': (0.3524, 0.4761), 'thetaim': (0.0409, 0.0899), 'f': (0.3028, 0.4498),
       'rho': (1.2074, 1.5995), 'kd': (0.2037, 0.4260), 'Dxx': (1.17e-4, 9.99e-4), 'qx': (0.0302, 0.0459)}
keys = list(RNG)
u = qmc.LatinHypercube(d=len(keys), seed=20261115).random(N)
P = qmc.scale(u, [RNG[k][0] for k in keys], [RNG[k][1] for k in keys])
rows = []
for i in range(N):
    p = dict(zip(keys, P[i]))
    x, t, Cm, Cim = solve_mim(p['alpha'], p['thetam'], p['thetaim'], p['f'], p['rho'], p['kd'], p['Dxx'], p['qx'],
                              L, TP, TMAX, NX, DT, 'U', C0, full=True)
    it = np.round(np.arange(0, TMAX + 1e-9, 0.1) / DT).astype(int)
    for j in range(11):
        ix = int(round(j * (NX - 1) / 10))
        for k in it:
            rows.append((f'case{i+1}', *[round(float(p[q]), 6) if q != 'Dxx' else round(float(p[q]), 8) for q in keys],
                         j / 10, x[ix], round(k * DT, 1), Cm[k, ix], Cim[k, ix]))
    print(i + 1, flush=True)
cols = ['case', 'alpha', 'thetam', 'thetaim', 'f', 'rho', 'kd', 'Dxx', 'qx', 'x_over_L', 'x', 'time', 'Cm', 'Cim']
pd.DataFrame(rows, columns=cols).to_csv('../data/BTC_spatial_Cm_Cim.csv', index=False)
