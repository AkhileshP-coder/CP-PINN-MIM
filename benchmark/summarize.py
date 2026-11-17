"""Collect all benchmark results (results/*.json) into summary_five_seed.json and print the numbers of Tables 6-8."""
import json, glob, collections, numpy as np
R = [json.load(open(f)) for f in glob.glob('results/*_s*_e*.json') if not f.endswith('_norm.json')]
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
def sel(variant, epochs, mult=None, seeds=None):
    return [r for r in R if r['variant'] == variant and r['epochs'] == epochs and (r['mult'] or {}) == (mult or {}) and (seeds is None or r['seed'] in seeds)]
def row(rs):
    return dict(n=len(rs), R2=ms([r['R2'] for r in rs]), minR2=min(r['R2'] for r in rs), maxR2=max(r['R2'] for r in rs), RMSE=ms([r['RMSE'] for r in rs]),
                seconds=float(np.mean([r['seconds'] for r in rs])),
                R2_m=ms([np.mean([x['R2_m'] for x in r['interior']]) for r in rs]), R2_im=ms([np.mean([x['R2_im'] for x in r['interior']]) for r in rs]),
                percase={c['case']: ms([next(x['R2'] for x in r['per_case'] if x['case'] == c['case']) for r in rs]) for c in rs[0]['per_case']})
S = {}
for v in ['cp', 'nopde', 'single', 'dataonly', 'physonly']: S[v] = row(sel(v, 2000))
S['cp3'] = row(sel('cp', 2000, seeds=(1234, 1334, 2024)))
S['phys_w2000'] = row(sel('physonly', 2000, dict(bci=20., ic=20., bco=20.))) if sel('physonly', 2000, dict(bci=20., ic=20., bco=20.)) else None
S['phys_w6000'] = row(sel('physonly', 6000, dict(bci=20., ic=20., bco=20.))) if sel('physonly', 6000, dict(bci=20., ic=20., bco=20.)) else None
try: S['gp'] = json.load(open('results/gp.json'))
except Exception: pass
base = sel('cp', 500)
S['sweep'] = {'base500': row(base)}
for k in ['data', 'pde', 'ic', 'bci', 'bco']:
    for m in (0.1, 10.0):
        rs = sel('cp', 500, {k: m}); S['sweep'][f'{k}_{m:g}'] = row(rs) if rs else None
for m in (0.1, 10.0):
    rs = sel('cp', 2000, {'ic': m}); S['sweep'][f'ic2000_{m:g}'] = row(rs) if rs else None
json.dump(S, open('summary_five_seed.json', 'w'), indent=1)
if __name__ == '__main__':
    for k, v in S.items():
        if k in ('sweep', 'gp') or v is None: continue
        print(k, v['n'], 'R2 %.4f±%.4f [%.4f,%.4f] RMSE %.4f±%.4f' % (*v['R2'], v['minR2'], v['maxR2'], *v['RMSE']))
    print('gp', S['gp']['kernel'], round(S['gp']['R2'], 4), round(S['gp']['RMSE'], 4))
    for k, v in S['sweep'].items():
        if v: print(k, v['n'], 'R2 %.3f±%.3f RMSE %.4f' % (*v['R2'], v['RMSE'][0]))
