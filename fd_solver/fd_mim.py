import numpy as np
from scipy.linalg import solve_banded
def solve_mim(alpha,thm,thim,f,rho,kd,D,U,L,tp,tmax,nx,dt,adv='thm_U',C0=1.0,xout=None,full=False):
    """Crank-Nicolson FD for mobile-immobile transport. Returns x, t, Cm(t,x), Cim.
    adv='thm_U': advective coeff = thm*U ; 'U': coeff = U."""
    x=np.linspace(0,L,nx); dx=x[1]-x[0]
    Rm=thm+rho*kd*f; Rim=thim+rho*kd*(1-f)
    A=thm*D; V=thm*U if adv=='thm_U' else U
    nt=int(round(tmax/dt)); t=np.arange(nt+1)*dt
    # operator on mobile: Rm dC/dt = A Cxx - V Cx - alpha(Cm-Cim)
    n=nx
    # build L matrix (interior, central diff); node0 Dirichlet; node n-1 zero-gradient (ghost)
    main=np.zeros(n); lo=np.zeros(n); up=np.zeros(n)
    a=A/dx**2; b=V/(2*dx)
    main[1:]=-2*a; lo[1:]=a+b; up[1:]=a-b   # lo: coefficient of C_{i-1}, up: C_{i+1}
    # outlet: C_{n} = C_{n-2}  => up+lo on lo
    lo[-1]=2*a
    # theta: dC/dt = (1/Rm)[Lm C - alpha C + alpha Cim]
    # Cim eq: Rim dCim/dt = alpha (C - Cim). Solve with CN on both, eliminate Cim analytically per node.
    # Cim^{n+1} = (Cim^n(1 - g/2) + g/2*(C^n+C^{n+1}))/(1+g/2), g=alpha*dt/Rim
    g=alpha*dt/Rim; p=(1-g/2)/(1+g/2); q=(g/2)/(1+g/2)
    # mobile CN: Rm(C1-C0)/dt = .5*(Lc0+Lc1) - alpha*.5*(C0+C1) + alpha*.5*(Cim0+Cim1)
    # substitute Cim1 = p Cim0 + q(C0+C1)
    # => LHS: Rm/dt C1 - .5 L C1 + .5 alpha C1 - .5 alpha q C1 ; define
    kk=0.5*alpha*(1-q)
    ab=np.zeros((3,n))
    ab[1,:]=Rm/dt+kk-0.5*main
    ab[0,1:]=-0.5*up[:-1]  # superdiag
    ab[2,:-1]=-0.5*lo[1:]  # subdiag
    ab[1,0]=1; ab[0,1]=0; 
    # row0 dirichlet: superdiag entry ab[0,1] belongs to row 0 -> zero
    C=np.zeros(n); Cim=np.zeros(n)
    out=np.zeros((nt+1,n)); out[0]=C; outim=np.zeros((nt+1,n))
    def Lop(c):
        r=main*c
        r[1:]+=lo[1:]*c[:-1]
        r[:-1]+=up[:-1]*c[1:]
        r[0]=0
        return r
    for k in range(nt):
        tn1=t[k+1]
        bc=C0 if (0<tn1<=tp) else 0.0   # inlet at new time
        rhs=Rm/dt*C+0.5*Lop(C)-0.5*alpha*C+0.5*alpha*(Cim+p*Cim+q*C)
        rhs[0]=bc
        Cn=solve_banded((1,1),ab,rhs)
        Cim=p*Cim+q*(C+Cn); C=Cn; out[k+1]=C; outim[k+1]=Cim
    if full: return x,t,out,outim
    return x,t,out
