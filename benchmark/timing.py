"""Inference timing of the trained CP-PINN (seed 1234): first call, then 10 warm-up and 200 timed repeats; forward pass and end-to-end (input build + forward + numpy)."""
import json,time,numpy as np,torch
from bench import Net,make_cases,PK,L,T,TT
torch.set_num_threads(1)
net=Net(); net.load_state_dict(torch.load('results/cp_s1234_e2000.pt')); net.eval()
nm=json.load(open('results/cp_s1234_e2000_norm.json')); mu,sd=np.array(nm['mu']),np.array(nm['sd'])
cases=[c for c in make_cases() if c['split']=='test']
tt=torch.tensor(TT,dtype=torch.float32); xo=torch.full_like(tt,L)
def build(c):
    pz=torch.tensor((np.array([c[k] for k in PK])-mu)/(sd+1e-10),dtype=torch.float32).expand(len(tt),8)
    return torch.cat([(xo/L).reshape(-1,1),(tt/T).reshape(-1,1),pz],1)
out={}
with torch.no_grad():
    for i,c in enumerate(cases):
        t0=time.perf_counter(); x=build(c); net(x)[0].numpy(); first=(time.perf_counter()-t0)*1e3
        for _ in range(10): net(build(c))
        fw=[];ee=[]
        for _ in range(200):
            x=build(c); t0=time.perf_counter(); net(x); fw.append((time.perf_counter()-t0)*1e3)
            t0=time.perf_counter(); y=net(build(c))[0].numpy(); ee.append((time.perf_counter()-t0)*1e3)
        out[f'TC-{i+1}']=dict(case=c['name'],first_ms=first,fwd_mean=float(np.mean(fw)),fwd_std=float(np.std(fw)),e2e_mean=float(np.mean(ee)),e2e_std=float(np.std(ee)),throughput=float(254/(np.mean(ee)/1e3)))
json.dump(out,open('timing.json','w'),indent=1)
for k,v in out.items(): print(k,{a:round(b,3) if isinstance(b,float) else b for a,b in v.items()})
