"""Does changing the loss weights improve the immobile concentration? Reads the existing runs benchmark/results/cp_s<seed>_e500[_<weight><mult>].json
(seeds 1234, 1334, 2024; 500 epochs; the weight sweep of Section 3.3) and cp_s<seed>_e2000.json, and writes ../tables/immobile_weight_check.csv.
R2 of the mobile and immobile concentrations inside the column (x/L = 0.25, 0.5, 0.75; t = 0.5-25.3 d), mean over the 20 test cases and the seeds."""
import json, numpy as np, pandas as pd
S = [1234, 1334, 2024]; rows = []
for tag, nm, ep in [('', 'adopted weights', 500), ('_pde0.1', 'PDE weight x 0.1', 500), ('_pde10', 'PDE weight x 10', 500), ('_data0.1', 'data weight x 0.1', 500), ('_data10', 'data weight x 10', 500), ('', 'adopted weights', 2000)]:
    v = []
    for s in S:
        d = json.load(open(f'results/cp_s{s}_e{ep}{tag}.json'))
        v.append((np.mean([e['R2_m'] for e in d['interior']]), np.mean([e['R2_im'] for e in d['interior']]), d['R2']))
    v = np.array(v); rows.append(dict(configuration=nm, epochs=ep, seeds=len(S), R2_mobile_field=v[:, 0].mean(), R2_immobile_field=v[:, 1].mean(), R2_outlet=v[:, 2].mean()))
df = pd.DataFrame(rows); df.to_csv('../tables/immobile_weight_check.csv', index=False); print(df.round(3).to_string())
