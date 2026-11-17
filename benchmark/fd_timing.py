import json, time, numpy as np
from fd_mim import solve_mim
L=0.299625; TP=7.708; T=25.3
cs=[c for c in __import__('bench').make_cases() if c['split']=='test']
out={}
for nx,dt in [(101,0.04),(201,0.02),(401,0.01),(801,0.005)]:
    ts=[];err=[]
    for c in cs:
        best=1e9
        for _ in range(3):
            t0=time.perf_counter(); x,t,u=solve_mim(c['alpha'],c['theta_m'],c['theta_im'],c['f'],c['rho'],c['kd'],c['D_xx'],c['qx'],L,TP,T,nx,dt,'U',1.0); best=min(best,time.perf_counter()-t0)
        ts.append(best)
    out[f'{nx}']=dict(nx=nx,dt=dt,sec_mean=float(np.mean(ts)),sec_per_case=ts)
json.dump(out,open('fd_timing.json','w'),indent=1); print({k:round(v['sec_mean'],3) for k,v in out.items()})
