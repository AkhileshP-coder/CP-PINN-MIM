"""CP-PINN benchmark on an FD-regenerated dataset: variants, seeds, loss-weight sweeps, timing.
usage: python bench.py <variant> <seed> <epochs> [key=mult ...]   (variants: cp, nopde, dataonly, physonly, single)"""
import sys, json, time, os, numpy as np, pandas as pd, torch, torch.nn as nn
from fd_mim import solve_mim
torch.set_num_threads(1)
L=0.299625; T=25.3; TP=7.708
PK=['alpha','theta_m','theta_im','f','rho','kd','D_xx','qx']
RNG={'alpha':(0.1011,0.1582),'theta_m':(0.3524,0.4761),'theta_im':(0.0409,0.0899),'f':(0.3028,0.4498),'rho':(1.2074,1.5995),'kd':(0.2037,0.4260),'D_xx':(1.17e-4,9.99e-4),'qx':(0.0302,0.0459)}
TT=np.round(np.arange(0,25.3001,0.1),1)
def make_cases():
    """Cases from the single data file ../data/BTC_spatial_Cm_Cim.csv: the first ten cases (sorted by name) are
    training cases, the next five test cases; the breakthrough curve is the outlet slice x/L = 1 of Cm divided by C0."""
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'BTC_spatial_Cm_Cim.csv')
    df = pd.read_csv(f)
    C0 = float(df[np.isclose(df['x_over_L'], 0.0)]['Cm'].max().round(6))
    names = sorted(df['case'].unique(), key=lambda s: int(s[4:]))   # case1 ... case50
    cases = []
    for k, n in enumerate(names):
        g = df[df['case'] == n]; r = g.iloc[0]
        o = g[np.isclose(g['x_over_L'], 1.0)].sort_values('time')
        assert len(o) == len(TT) and np.allclose(o['time'].values, TT)
        cases.append(dict(name=n, split='train' if k < 30 else 'test', alpha=float(r.alpha), theta_m=float(r.thetam), theta_im=float(r.thetaim),
                          f=float(r.f), rho=float(r.rho), kd=float(r.kd), D_xx=float(r.Dxx), qx=float(r.qx), C=(o['Cm'].values / C0).tolist()))
    return cases
class Net(nn.Module):
    def __init__(s,single=False,h=64):
        super().__init__(); s.single=single
        enc=lambda: nn.Sequential(nn.Linear(10,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        dec=lambda o: nn.Sequential(nn.Linear(h,h//2),nn.Tanh(),nn.Linear(h//2,o),nn.Sigmoid())
        if single: s.e=enc(); s.d=dec(2)
        else: s.em,s.dm,s.ei,s.di=enc(),dec(1),enc(),dec(1)
        for m in s.modules():
            if isinstance(m,nn.Linear): nn.init.orthogonal_(m.weight,gain=0.8); nn.init.zeros_(m.bias)
    def forward(s,inp):
        if s.single: o=s.d(s.e(inp)); return o[:,0:1],o[:,1:2]
        return s.dm(s.em(inp)),s.di(s.ei(inp))
def run(variant,seed,epochs,mult):
    cases=make_cases(); tr=[c for c in cases if c['split']=='train']; te=[c for c in cases if c['split']=='test']
    torch.manual_seed(seed); np.random.seed(seed)
    P=np.array([[c[k] for k in PK] for c in tr]); mu,sd=P.mean(0),P.std(0)
    W=dict(data=10.,pde=1.,ic=.5,bci=.5,bco=.5)
    if variant=='nopde': W['pde']=0
    if variant=='dataonly': W.update(pde=0,ic=0,bci=0,bco=0)
    if variant=='physonly': W['data']=0
    for k,v in mult.items(): W[k]*=v
    net=Net(single=(variant=='single')); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    def inp(x,t,c):
        pz=torch.tensor((np.array([c[k] for k in PK])-mu)/(sd+1e-10),dtype=torch.float32).expand(len(x),8)
        return torch.cat([(x/L).reshape(-1,1),(t/T).reshape(-1,1),pz],1)
    tt=torch.tensor(TT,dtype=torch.float32); xo=torch.full_like(tt,L)
    hist=[]; r2hist=[]
    def test_r2():
        """pooled test R2 of the outlet curves in units of Cm/C0 (the R2 reported in Table 7), forward pass only"""
        with torch.no_grad():
            ao_,ap_=[],[]
            for c_ in te:
                p_=net(inp(xo,tt,c_))[0].squeeze().numpy(); o_=np.array(c_['C']); ao_.append(o_); ap_.append(p_)
            ao_,ap_=np.concatenate(ao_),np.concatenate(ap_)
            return float(1-((ao_-ap_)**2).sum()/((ao_-ao_.mean())**2).sum())
    for ep in range(epochs):
        tot=0
        for c in np.random.permutation(len(tr)):
            c=tr[c]; opt.zero_grad()
            cd=torch.tensor(c['C'],dtype=torch.float32)
            Cm,_=net(inp(xo,tt,c)); ld=((Cm.squeeze()-cd)**2).mean()
            xi=(torch.rand(100)*L).requires_grad_(True); ti=(torch.rand(100)*T).requires_grad_(True)
            Ci,Si=net(inp(xi,ti,c))
            g=lambda y,v: torch.autograd.grad(y.sum(),v,create_graph=True)[0]
            Ct=g(Ci,ti); Cx=g(Ci,xi); Cxx=g(Cx,xi); St=g(Si,ti)
            Rm=c['theta_m']+c['f']*c['rho']*c['kd']; Ri=c['theta_im']+(1-c['f'])*c['rho']*c['kd']
            r1=Rm*Ct-c['theta_m']*c['D_xx']*Cxx+c['qx']*Cx+c['alpha']*(Ci-Si); r2=Ri*St-c['alpha']*(Ci-Si)
            lp=(r1**2).mean()+(r2**2).mean()
            xb=torch.zeros(50); tb=torch.cat([torch.rand(25)*TP+1e-4,TP+torch.rand(25)*(T-TP)])
            Cb,_=net(inp(xb,tb,c)); lbi=((Cb.squeeze()-(tb<=TP).float())**2).mean()
            xe=torch.full((50,),L).requires_grad_(True); te_=torch.rand(50)*T
            Ce,_=net(inp(xe,te_,c)); lbo=(g(Ce,xe)**2).mean()
            x0=torch.rand(50)*L; Cz,Sz=net(inp(x0,torch.zeros(50),c)); lic=(Cz**2).mean()+(Sz**2).mean()
            loss=W['data']*ld+W['pde']*lp+W['ic']*lic+W['bci']*lbi+W['bco']*lbo
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),0.1); opt.step(); tot+=loss.item()
        hist.append(tot/len(tr))
        r2hist.append(test_r2())
    # evaluation on test cases
    res=[];allo=[];allp=[]
    with torch.no_grad():
        for c in te:
            p=net(inp(xo,tt,c))[0].squeeze().numpy(); o=np.array(c['C']); cm=o.max()
            r2=1-((o-p)**2).sum()/((o-o.mean())**2).sum(); rm=np.sqrt((((o-p)/cm)**2).mean())
            res.append(dict(case=c['name'],R2=float(r2),RMSE=float(rm),pred=p.tolist())); allo.append(o); allp.append(p)
    # ---- interior (spatial) evaluation against the finite-difference fields, mobile and immobile
    XF=np.array([0.25,0.5,0.75]); TF=np.round(np.arange(0.5,25.3001,0.5),1)
    SNAP=[5.0,7.7,12.0,18.0]; XG=np.linspace(0,1,21)
    inter=[]; prof={}
    with torch.no_grad():
        for c in te:
            _,_,Fm,Fi=solve_mim(c['alpha'],c['theta_m'],c['theta_im'],c['f'],c['rho'],c['kd'],c['D_xx'],c['qx'],L,TP,T,801,0.005,'U',1.0,full=True)
            xs=np.repeat(XF,len(TF)); ts=np.tile(TF,len(XF))
            xi_=np.round(xs*800).astype(int); ti_=np.round(ts/0.005).astype(int)
            Cm_,Ci_=net(inp(torch.tensor(xs*L,dtype=torch.float32),torch.tensor(ts,dtype=torch.float32),c))
            om,oi=Fm[ti_,xi_],Fi[ti_,xi_]; pm,pi=Cm_.squeeze().numpy(),Ci_.squeeze().numpy()
            r2=lambda o,p: float(1-((o-p)**2).sum()/((o-o.mean())**2).sum())
            inter.append(dict(case=c['name'],R2_m=r2(om,pm),R2_im=r2(oi,pi),RMSE_m=float(np.sqrt(((om-pm)**2).mean())/om.max()),RMSE_im=float(np.sqrt(((oi-pi)**2).mean())/oi.max()),om=om.tolist(),oi=oi.tolist(),pm=pm.tolist(),pi=pi.tolist()))
            pr={}
            for ts_ in SNAP:
                xg=XG*L; tg=np.full_like(xg,ts_); a,b_=net(inp(torch.tensor(xg,dtype=torch.float32),torch.tensor(tg,dtype=torch.float32),c))
                ix=np.round(XG*800).astype(int); it=int(round(ts_/0.005))
                pr[str(ts_)]=dict(x=xg.tolist(),Cm_fd=Fm[it,ix].tolist(),Cim_fd=Fi[it,ix].tolist(),Cm=a.squeeze().numpy().tolist(),Cim=b_.squeeze().numpy().tolist())
            prof[c['name']]=pr
    ao,ap=np.concatenate(allo),np.concatenate(allp)
    out=dict(variant=variant,seed=seed,epochs=epochs,mult=mult,R2=float(1-((ao-ap)**2).sum()/((ao-ao.mean())**2).sum()),RMSE=float(np.sqrt(((ao-ap)**2).mean())),per_case=res,interior=inter,profiles=prof,final_loss=hist[-1],loss=hist[::max(1,epochs//100)],loss_full=hist,r2_full=r2hist)
    return out,net,mu,sd
if __name__=='__main__':
    variant,seed,epochs=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); mult={a.split('=')[0]:float(a.split('=')[1]) for a in sys.argv[4:]}
    t0=time.time(); out,net,mu,sd=run(variant,seed,epochs,mult); out['seconds']=time.time()-t0
    tag=f"{variant}_s{seed}_e{epochs}"+''.join(f"_{k}{v:g}" for k,v in mult.items())
    os.makedirs('results',exist_ok=True); json.dump(out,open(f'results/{tag}.json','w'))
    if variant=='cp' and not mult and epochs>=2000: torch.save(net.state_dict(),f'results/{tag}.pt'); json.dump(dict(mu=mu.tolist(),sd=sd.tolist()),open(f'results/{tag}_norm.json','w'))
    print(tag,'R2=%.4f RMSE=%.4f time=%.0fs'%(out['R2'],out['RMSE'],out['seconds']))
