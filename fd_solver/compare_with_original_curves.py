"""Compare the finite-difference solution with the ten breakthrough curves of the original data file
(C0 = 10, 801 nodes, dt = 0.005 d). Writes ../tables/fd_vs_original_curves.csv."""
import pandas as pd, numpy as np
from fd_mim import solve_mim
d = pd.read_csv('../data/BTC_original_10cases.csv')
rows = []
for c, g in d.groupby('case'):
    r = g.iloc[0]; o = g.concentration.values
    x, t, u = solve_mim(r.alpha, r.thetam, r.thetaim, r.f, r.rho, r.kd, r.Dxx, r.qx, 0.299625, 7.708, 25.3, 801, 0.005, 'U', 10.0)
    p = np.interp(g.time.values, t, u[:, -1])
    rows.append((c, 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum(), np.sqrt(((o - p) ** 2).mean())))
pd.DataFrame(rows, columns=['case', 'R2', 'RMSE']).to_csv('../tables/fd_vs_original_curves.csv', index=False)
print(pd.DataFrame(rows, columns=['case', 'R2', 'RMSE']).round(5))
