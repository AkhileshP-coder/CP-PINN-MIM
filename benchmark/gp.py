"""Gaussian-process baseline on the same 30 training curves (inputs: t and the 8 parameters; target C/C0 at x = L).
The kernel is chosen by 5-fold cross-validation over training cases (two cases held out per fold)."""
import json,numpy as np,warnings
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF,Matern,WhiteKernel,ConstantKernel as CK
from bench import make_cases,PK,T,TT
warnings.filterwarnings('ignore')
STEP=8   # training curves are sub-sampled to every 8th point (0.8 d) to keep the 30-curve fit affordable
cs=make_cases(); tr=[c for c in cs if c['split']=='train']; te=[c for c in cs if c['split']=='test']
P=np.array([[c[k] for k in PK] for c in tr]); mu,sd=P.mean(0),P.std(0)
def XY(cl,step):
    X=np.vstack([np.column_stack([TT[::step]/T,np.tile((np.array([c[k] for k in PK])-mu)/(sd+1e-10),(len(TT[::step]),1))]) for c in cl]); Y=np.concatenate([np.array(c['C'])[::step] for c in cl]); return X,Y
kernels={
 'rbf_ard':lambda: CK(1.0)*RBF(np.ones(9),(1e-2,1e3))+WhiteKernel(1e-4,(1e-8,1e-1)),
 'matern_ard':lambda: CK(1.0)*Matern(np.ones(9),(1e-2,1e3),nu=2.5)+WhiteKernel(1e-4,(1e-8,1e-1)),
 'rbf_iso_params':lambda: CK(1.0)*RBF([1.0]+[3.0]*1,(1e-2,1e3)) if False else CK(1.0)*RBF(np.r_[1.0,np.full(8,3.0)],(1e-2,1e3))+WhiteKernel(1e-4,(1e-8,1e-1)),
}
def fit(name,cl,step): X,Y=XY(cl,step); return GPR(kernels[name](),normalize_y=True,random_state=0).fit(X,Y)
folds=[list(range(6*i,6*i+6)) for i in range(5)]; cv={}
for name in kernels:
    err=[]
    for f in folds:
        trn=[c for i,c in enumerate(tr) if i not in f]; val=[tr[i] for i in f]
        g=fit(name,trn,STEP); Xv,Yv=XY(val,STEP); err.append(np.sqrt(((g.predict(Xv)-Yv)**2).mean()))
    cv[name]=float(np.mean(err)); print(name,cv[name],flush=True)
best=min(cv,key=cv.get); gp=fit(best,tr,STEP)   # same sub-sampling (every 8th point, 0.8 d) as in the cross-validation
Xt,Yt=XY(te,1); p=gp.predict(Xt); res=[];ao=[];ap=[]
for i,c in enumerate(te):
    o=np.array(c['C']); q=p[i*len(TT):(i+1)*len(TT)]; cm=o.max()
    res.append(dict(case=c['name'],R2=float(1-((o-q)**2).sum()/((o-o.mean())**2).sum()),RMSE=float(np.sqrt((((o-q)/cm)**2).mean())),pred=q.tolist())); ao.append(o/cm); ap.append(q/cm)
ao,ap=np.concatenate(ao),np.concatenate(ap)
out=dict(variant='gp',kernel=best,cv_rmse=cv,R2=float(1-((ao-ap)**2).sum()/((ao-ao.mean())**2).sum()),RMSE=float(np.sqrt(((ao-ap)**2).mean())),per_case=res)
json.dump(out,open('results/gp.json','w')); print(best,out['R2'],out['RMSE'])
