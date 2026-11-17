"""Write tables/benchmark_results_per_run.csv from benchmark/results/*.json (one row per run, per-case test R2 included)."""
import json, glob, os, pandas as pd
rows = []
for f in sorted(glob.glob('results/*_s*_e*.json')):
    if f.endswith('_norm.json'): continue
    r = json.load(open(f))
    d = dict(model=r['variant'], seed=r['seed'], epochs=r['epochs'], weight_multipliers=json.dumps(r['mult'] or {}), R2=r['R2'], RMSE=r['RMSE'])
    for c in r['per_case']: d['R2_' + c['case']] = c['R2']
    d['interior_R2_mobile'] = sum(x['R2_m'] for x in r['interior']) / len(r['interior'])
    d['interior_R2_immobile'] = sum(x['R2_im'] for x in r['interior']) / len(r['interior'])
    d['seconds'] = r['seconds']; rows.append(d)
os.makedirs('../tables', exist_ok=True)
pd.DataFrame(rows).to_csv('../tables/benchmark_results_per_run.csv', index=False); print(len(rows), 'runs')
