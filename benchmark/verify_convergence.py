import numpy as np,pandas as pd,time
from fd_mim import solve_mim
d=pd.read_csv('../data/BTC_spatial_Cm_Cim.csv'); d=d[d.x_over_L==1]
tt=np.arange(0,25.3001,0.1)
rows=[]
for c in ['case1','case2','case9']:
    r=d[d.case==c].iloc[0]; prev=None
    for nx,dt in [(101,0.04),(201,0.02),(401,0.01),(801,0.005),(1601,0.0025)]:
        t0=time.time()
        x,t,o=solve_mim(r.alpha,r.thetam,r.thetaim,r.f,r.rho,r.kd,r.Dxx,r.qx,0.299625,7.708,25.3,nx,dt,'U',10.0)
        el=time.time()-t0; p=np.interp(tt,t,o[:,-1])
        dif=np.sqrt(((p-prev)**2).mean())/p.max() if prev is not None else np.nan
        rows.append(dict(case=c,nx=nx,dt=dt,rel_rms_change_vs_previous=dif,peak=p.max(),runtime_s=el)); prev=p
df=pd.DataFrame(rows); df.to_csv('convergence_3cases.csv',index=False); print(df.to_string())
