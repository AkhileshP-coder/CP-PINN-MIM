"""Test accuracy of the Gaussian-process baseline (Matern-5/2 ARD kernel) as a function of the number of training cases (random subsets of the 30 training cases; 20 test cases)."""
import json,numpy as np,warnings
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern,WhiteKernel,ConstantKernel as CK
from bench import make_cases,PK,T,TT
warnings.filterwarnings('ignore')
STEP=8
cs=make_cases(); tr=[c for c in cs if c['split']=='train']; te=[c for c in cs if c['split']=='test']
def XY(cl,step,mu,sd):
    X=np.vstack([np.column_stack([TT[::step]/T,np.tile((np.array([c[k] for k in PK])-mu)/(sd+1e-10),(len(TT[::step]),1))]) for c in cl]); Y=np.concatenate([np.array(c['C'])[::step] for c in cl]); return X,Y
out={}
for n in [5,10,15,20,25,30]:
    r2s=[];rm=[]
    for rep in range(3 if n<30 else 1):
        rng=np.random.RandomState(100*n+rep); sub=[tr[i] for i in rng.choice(30,n,replace=False)]
        P=np.array([[c[k] for k in PK] for c in sub]); mu,sd=P.mean(0),P.std(0)
        X,Y=XY(sub,STEP,mu,sd); g=GPR(CK(1.0)*Matern(np.ones(9),(1e-2,1e3),nu=2.5)+WhiteKernel(1e-4,(1e-8,1e-1)),normalize_y=True,random_state=0).fit(X,Y)
        Xt,Yt=XY(te,1,mu,sd); p=g.predict(Xt); ao=[];ap=[]
        for i,c in enumerate(te):
            o=np.array(c['C']); q=p[i*len(TT):(i+1)*len(TT)]; ao.append(o/o.max()); ap.append(q/o.max())
        ao,ap=np.concatenate(ao),np.concatenate(ap); r2s.append(float(1-((ao-ap)**2).sum()/((ao-ao.mean())**2).sum())); rm.append(float(np.sqrt(((ao-ap)**2).mean())))
    out[n]=dict(R2_mean=float(np.mean(r2s)),R2_min=float(np.min(r2s)),R2_max=float(np.max(r2s)),RMSE_mean=float(np.mean(rm)),runs=len(r2s)); print(n,out[n],flush=True)
json.dump(out,open('results/gp_learning_curve.json','w'),indent=1)
