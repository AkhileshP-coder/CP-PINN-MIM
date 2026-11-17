"""Final PDE residual (mean squared, Eq. 26) of the trained CP-PINN (seed 1234) on 5,000 random interior points per case."""
import json,numpy as np,torch
from bench import Net,make_cases,PK,L,T
torch.set_num_threads(1); torch.manual_seed(0)
net=Net(); net.load_state_dict(torch.load('results/cp_s1234_e2000.pt')); net.eval()
nm=json.load(open('results/cp_s1234_e2000_norm.json')); mu,sd=np.array(nm['mu']),np.array(nm['sd'])
out={}
for c in make_cases():
    n=5000; x=(torch.rand(n)*L).requires_grad_(True); t=(torch.rand(n)*T).requires_grad_(True)
    pz=torch.tensor((np.array([c[k] for k in PK])-mu)/(sd+1e-10),dtype=torch.float32).expand(n,8)
    Ci,Si=net(torch.cat([(x/L).reshape(-1,1),(t/T).reshape(-1,1),pz],1))
    g=lambda y,v: torch.autograd.grad(y.sum(),v,create_graph=True)[0]
    Ct=g(Ci,t); Cx=g(Ci,x); Cxx=g(Cx,x); St=g(Si,t)
    Rm=c['theta_m']+c['f']*c['rho']*c['kd']; Ri=c['theta_im']+(1-c['f'])*c['rho']*c['kd']
    r1=Rm*Ct-c['theta_m']*c['D_xx']*Cxx+c['qx']*Cx+c['alpha']*(Ci-Si); r2=Ri*St-c['alpha']*(Ci-Si)
    out[c['name']]=dict(split=c['split'],pde=float((r1**2).mean()+(r2**2).mean()))
tr=[v['pde'] for v in out.values() if v['split']=='train']; te=[v['pde'] for v in out.values() if v['split']=='test']
out['summary']=dict(train_mean=float(np.mean(tr)),test_mean=float(np.mean(te)))
json.dump(out,open('resid.json','w'),indent=1); print(out['summary'])
