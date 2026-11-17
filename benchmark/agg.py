import json,glob,numpy as np,collections
R=[json.load(open(f)) for f in [f for f in glob.glob('results/*_s*_e*.json') if not f.endswith('_norm.json')]]
def ms(v): return float(np.mean(v)),float(np.std(v,ddof=1)) if len(v)>1 else float('nan')
main=collections.defaultdict(list); sweep=collections.defaultdict(list); sweep2k=collections.defaultdict(list)
for r in R:
    if r['epochs']==2000 and not r['mult']:
        main[r['variant']].append(r)
        if r['variant']=='cp' and r['seed'] in (1234,1334,2024): main['cp3'].append(r)
    if r['epochs']==2000 and r['mult']: sweep2k[(list(r['mult'])[0],list(r['mult'].values())[0])].append(r)
    if r['epochs']==500 and r['variant']=='cp':
        key=('base',1.0) if not r['mult'] else (list(r['mult'])[0],list(r['mult'].values())[0]); sweep[key].append(r)
S={'main':{},'sweep':{}}
for v,l in main.items():
    S['main'][v]=dict(minR2=min(x['R2'] for x in l),maxR2=max(x['R2'] for x in l),n=len(l),R2=ms([x['R2'] for x in l]),RMSE=ms([x['RMSE'] for x in l]),seeds=sorted(x['seed'] for x in l),
        per_case={c:ms([x['per_case'][i]['R2'] for x in l]) for i,c in enumerate([p['case'] for p in l[0]['per_case']])},sec=float(np.mean([x['seconds'] for x in l])))
for k,l in sweep.items():
    S['sweep'][f'{k[0]}_{k[1]:g}']=dict(n=len(l),R2=ms([x['R2'] for x in l]),RMSE=ms([x['RMSE'] for x in l]))
S['sweep2000']={f'{k[0]}_{k[1]:g}':dict(n=len(l),R2=ms([x['R2'] for x in l]),RMSE=ms([x['RMSE'] for x in l])) for k,l in sweep2k.items()}
try: S['main']['gp']=json.load(open('results/gp.json'))
except Exception: pass
json.dump(S,open('summary.json','w'),indent=1)
for v,d in S['main'].items(): print(v,{k:(np.round(x,4).tolist() if isinstance(x,(list,tuple)) else x) for k,x in d.items() if k in('n','R2','RMSE')})
for k,d in sorted(S['sweep'].items()): print(k,d['n'],np.round(d['R2'],4).tolist(),np.round(d['RMSE'],4).tolist())
