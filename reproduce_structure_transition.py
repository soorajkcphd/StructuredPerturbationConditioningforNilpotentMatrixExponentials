"""Reproduce the structure-transition figure and validate its statements.

Run with Python 3, NumPy, SciPy, mpmath and Matplotlib. Outputs are beside
this script. Integer/dyadic coefficient cancellations are checked before
floating-point norms. SVD/error-bound evaluations are not interval arithmetic.
"""
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "etna-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "etna-cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import expm, expm_frechet, svd, svdvals

from reproduce_markov import model, basis, exact_maps

OUT = Path(__file__).resolve().parent


def opnorm(M):
    return float(svdvals(M)[0])


def units(d, pairs):
    result = []
    for i, j in pairs:
        E = np.zeros((d, d))
        E[i, j] = 1
        result.append(E)
    return result


def coefficient_maps(X, B):
    # All inputs to this function are small integer or Gaussian-integer
    # matrices; powers and sums are exactly representable before division.
    d = len(X)
    powers = [np.eye(d, dtype=X.dtype)]
    for k in range(1, d + 1):
        powers.append(powers[-1] @ X)
        if not np.any(powers[-1]):
            m = k
            powers.pop()
            break
    else:
        raise AssertionError("Input must be nilpotent")
    C = []
    for n in range(2 * m - 1):
        cols = []
        for E in B:
            Y = sum((powers[p] @ E @ powers[n-p]
                     for p in range(m) if 0 <= n-p < m), np.zeros_like(X))
            cols.append(Y.ravel() / math.factorial(n+1))
        C.append(np.column_stack(cols))
    r = max(n for n, M in enumerate(C) if np.any(M))
    return powers, C[:r+1]


def data(X, powers, CV, CU, BV=None, BU=None):
    m = len(powers)
    scale = math.factorial(m-1) * np.linalg.norm(X) / np.linalg.norm(powers[-1])
    return dict(X=X, powers=powers, CV=CV, CU=CU,
                BV=BV, BU=BU, rhoV=len(CV)-m+1, rhoU=len(CU)-m+1,
                AV=scale*CV[-1], AU=scale*CU[-1])


def shift_data(d, diagonal=False):
    X = np.diag(np.ones(d-1), 1)
    BV = units(d, [(i,j) for i in range(d) for j in range(d) if i < j])
    BU = units(d, ([(i,i) for i in range(d)] if diagonal else
                   [(i,j) for i in range(d) for j in range(d) if i >= j]))
    powers, CV = coefficient_maps(X, BV)
    _, CU = coefficient_maps(X, BU)
    return data(X, powers, CV, CU, BV, BU)


def markov_data(H=6, variant=0):
    A, branches = model(H, variant)
    BV = basis(A, branches, True)
    BU = []
    for i, j, k in branches:
        E = np.zeros_like(A)
        E[i,j] = E[i,k] = 1
        BU.append(E)
    powers, CV, rv, _ = exact_maps(A, H, BV, True)
    _, CU, ru, _ = exact_maps(A, H, BU, True)
    BFV = [np.asarray(E,dtype=float)/np.sqrt(2) for E in BV]
    BFU = [np.asarray(E,dtype=float)/np.sqrt(2) for E in BU]
    return data(np.asarray(A,dtype=float)/8, powers, CV[:rv+1], CU[:ru+1], BFV, BFU)


def scaled_operators(D, t):
    """Return t^{-rho_J} L_J without forming high powers of t."""
    m = len(D['powers'])
    F = sum(t**(k-m+1)*P/math.factorial(k) for k,P in enumerate(D['powers']))
    scale = np.linalg.norm(D['X'])/np.linalg.norm(F)
    result = []
    for C in (D['CV'], D['CU']):
        r = len(C)-1
        result.append(scale*sum(t**(k-r)*M for k,M in enumerate(C)))
    return result


def errors(D, t):
    m = len(D['powers'])
    a = np.linalg.norm(D['powers'][-1])/math.factorial(m-1)
    beta = sum(t**(k-m+1)*np.linalg.norm(P)/math.factorial(k)
               for k,P in enumerate(D['powers'][:-1]))/a
    e = []
    for C in (D['CV'], D['CU']):
        r = len(C)-1
        alpha = sum(t**(k-r)*opnorm(M) for k,M in enumerate(C[:-1]))/opnorm(C[-1])
        e.append((alpha+beta)/(1-beta) if beta < 1 else None)
    return beta, e


def check_exact_bounds():
    rng = np.random.default_rng(20260924)
    count, discrepancy = 0, 0.
    for d in (2,3,4):
        for complex_case in (False, True):
            for _ in range(4):
                X = rng.normal(size=(d,d))/3
                Q = rng.normal(size=(d*d, min(5,d*d)))
                if complex_case:
                    X = X + 1j*rng.normal(size=X.shape)/3
                    Q = Q + 1j*rng.normal(size=Q.shape)
                Q = np.linalg.qr(Q)[0]
                L = np.column_stack([expm_frechet(X,Q[:,j].reshape(d,d),
                         compute_expm=False).ravel() for j in range(Q.shape[1])])
                L *= np.linalg.norm(X)/np.linalg.norm(expm(X))
                V,U = L[:,:2],L[:,2:]
                kv,ku=opnorm(V),opnorm(U)
                for eta in np.r_[0.,np.logspace(-4,4,9)]:
                    k=opnorm(np.column_stack((V,eta*U)))
                    gram=V@V.conj().T+eta**2*U@U.conj().T
                    independent=math.sqrt(max(0,float(np.linalg.eigvalsh(gram)[-1])))
                    discrepancy=max(discrepancy,abs(independent/k-1))
                    assert max(kv,eta*ku) <= k*(1+3e-13)
                    assert k <= np.hypot(kv,eta*ku)*(1+3e-13)
                    assert abs(independent/k-1)<3e-13
                    count+=1
    return dict(cases=count,max_relative_gram_discrepancy=discrepancy)


def check_angle():
    count=0
    for complex_case in (False,True):
        for c in (0.,.2,.8,1.):
            u=np.array([1,0],dtype=complex)
            z=np.array([c,np.sqrt(1-c*c)],dtype=complex)
            if complex_case:
                z[0]*=1j
            for a in (0.,.1,1.,10.):
                kv,ku=2.,3.
                B=np.column_stack((kv*u,a*ku*z))
                f2=(kv*kv+(a*ku)**2+np.hypot(kv*kv-(a*ku)**2,2*a*kv*ku*c))/2
                assert abs(opnorm(B)**2/f2-1)<3e-14
                count+=1
    return count


def verify_transitions():
    results=[]
    families=[]
    for d in (2,3,4,6):
        for diagonal in (False,True):
            D=shift_data(d,diagonal)
            ku=(np.sqrt((d-1)/d) if diagonal else
                 math.factorial(d-1)*np.sqrt(d-1)/math.factorial(2*d-1))
            np.testing.assert_allclose(opnorm(D['AV']),d-1,rtol=1e-14)
            np.testing.assert_allclose(opnorm(D['AU']),ku,rtol=1e-14)
            assert np.linalg.matrix_rank(D['AV'])==np.linalg.matrix_rank(D['AU'])==1
            assert D['rhoV']==0 and D['rhoU']==(1 if diagonal else d)
            families.append((f'shift_d{d}_'+('diagonal' if diagonal else 'full'),D))
    markov_checks=[]
    for H in range(2,9):
        for variant in range(3):
            D=markov_data(H,variant)
            assert D['rhoV']==-1 and D['rhoU']==0
            # Entry supports of the two leading ranges are disjoint.
            assert not np.any(np.any(D['AV']!=0,axis=1) & np.any(D['AU']!=0,axis=1))
            assert np.linalg.norm(D['AV'].T@D['AU'])==0
            markov_checks.append(dict(H=H,variant=variant))
    families.append(('markov_H6',markov_data()))
    rng=np.random.default_rng(919)
    for d in (3,4,5):
        for complex_case in (False,True):
            X=np.triu(rng.integers(-2,3,size=(d,d)),1).astype(complex if complex_case else float)
            for i in range(d-1):
                X[i,i+1]=1+1j if complex_case else 1
            BV=units(d,[(i,j) for i in range(d) for j in range(d) if i<j])
            BU=units(d,[(i,j) for i in range(d) for j in range(d) if i>=j])
            powers,CV=coefficient_maps(X,BV)
            _,CU=coefficient_maps(X,BU)
            families.append((f'random_d{d}_complex{complex_case}',data(X,powers,CV,CU)))
    count=0
    for name,D in families:
        kv,ku=opnorm(D['AV']),opnorm(D['AU'])
        local=[]
        for t in (10.,100.,1000.):
            V,U=scaled_operators(D,t)
            beta,e=errors(D,t)
            for factor in (0.,.01,.3,1.,3.,100.):
                a=factor*kv/ku
                actual=opnorm(np.column_stack((V,a*U)))
                target=opnorm(np.column_stack((D['AV'],a*D['AU'])))
                relative=abs(actual/target-1)
                bound=np.sqrt(2)*max(e) if beta<1 else None
                if bound is not None:
                    assert relative <= bound+3e-13
                    count+=1
                local.append(dict(t=t,balance_factor=factor,relative_error=relative,bound=bound))
        results.append(dict(name=name,rhoV=D['rhoV'],rhoU=D['rhoU'],KV=kv,KU=ku,checks=local))
    return dict(finite_bound_checks=count,markov_exact_support_checks=markov_checks,cases=results)


def independent_frechet(D,t):
    V,U=scaled_operators(D,t)
    normexp=np.linalg.norm(expm(t*D['X']))
    discrepancy=[]
    for scaled,rho,B in ((V,D['rhoV'],D['BV']),(U,D['rhoU'],D['BU'])):
        L=np.column_stack([expm_frechet(t*D['X'],E,compute_expm=False).ravel() for E in B])
        L*=t*np.linalg.norm(D['X'])/normexp/t**rho
        e=np.linalg.norm(L-scaled)/np.linalg.norm(scaled)
        assert e<2e-12
        discrepancy.append(float(e))
    return discrepancy


def verify_exact_five_state():
    """Check the all-t maximum law independently of the rank-one formula."""
    cases=[]
    for pnum,qnum in [(3,8),(2,7),(5,5),(9,1)]:
        A=np.zeros((5,5),dtype=object)  # A=10P, integral.
        A[0,2],A[0,3]=pnum,10-pnum
        A[1,2],A[1,3]=qnum,10-qnum
        A[2,4]=A[3,4]=10
        Eu=[np.asarray(E,dtype=object) for E in
            units(5,[(0,2),(0,3),(1,2),(1,3),(2,4),(3,4)])]
        rawV=[Eu[0]-Eu[1],Eu[2]-Eu[3]]
        rawU=[Eu[0]+Eu[1],Eu[2]+Eu[3],Eu[4],Eu[5]]
        for Ev in rawV:
            assert not np.any(A@Ev) and not np.any(Ev@A)
            for E in rawU:
                assert sum((Ev*E).ravel())==0
                assert sum((Ev*(A@E+E@A)).ravel())==0
        for E in rawU:
            assert not np.any(A@A@E+A@E@A+E@A@A)
        P=np.asarray(A,dtype=float)/10
        BV=[np.asarray(E,dtype=float)/np.sqrt(2) for E in rawV]
        BU=[np.asarray(E,dtype=float)/(np.sqrt(2) if i<2 else 1.)
            for i,E in enumerate(rawU)]
        CV=[np.column_stack([E.ravel() for E in BV])]
        CU=[np.column_stack([E.ravel() for E in BU]),
            np.column_stack([((P@E+E@P)/2).ravel() for E in BU])]
        powers=[np.eye(5),P,np.asarray(A@A,dtype=float)/100]
        D=data(P,powers,CV,CU,BV,BU)
        assert np.linalg.matrix_rank(D['AV'])==np.linalg.matrix_rank(D['AU'])==2
        pn,qn=pnum/10,qnum/10
        M=np.array([[np.sqrt(2),0,pn,1-pn],
                    [0,np.sqrt(2),qn,1-qn]])/2
        gamma=opnorm(M)
        np.testing.assert_allclose(opnorm(CU[-1]),gamma,rtol=2e-14)
        measurements=[]
        for t in (1.,10.,100.,1000.):
            V,U=scaled_operators(D,t)
            lv=t**D['rhoV']*V;lu=t**D['rhoU']*U
            kv,ku=opnorm(lv),opnorm(lu)
            eta_star=kv/ku
            np.testing.assert_allclose(eta_star,1/np.sqrt(1+t*t*gamma*gamma),rtol=2e-14)
            assert not np.any(lv.T@lu)
            for eta in (0.,eta_star/100,eta_star,3*eta_star,1.):
                mixed=opnorm(np.column_stack([lv,eta*lu]))
                gap=abs(mixed/max(kv,eta*ku)-1)
                assert gap<3e-14
                measurements.append(dict(t=t,eta=eta,relative_max_gap=gap))
        cases.append(dict(p=pn,q=qn,rankV=2,rankU=2,asymptotic_balance_constant=1/gamma,
                          exact_frechet_checks=independent_frechet(D,10.),measurements=measurements))
    return cases


def slow_convergence_check():
    D=shift_data(6)
    kv,ku=opnorm(D['AV']),opnorm(D['AU'])
    rows=[]
    for t in (10.,100.,1000.):
        V,U=scaled_operators(D,t)
        beta,e=errors(D,t)
        finite=opnorm(V)/(t**6*opnorm(U))
        asymptotic=kv/(t**6*ku)
        rows.append(dict(t=t,beta=float(beta),certificate_applicable=bool(beta<1),
            certificate_bound=float(np.sqrt(2)*max(e)) if beta<1 else None,
            eta_star_finite=finite,eta_star_asymptotic=asymptotic,
            ratio_asymptotic_to_finite=asymptotic/finite,
            mixed_at_asymptotic_balance=opnorm(np.column_stack([V,(kv/ku)*U])),
            limiting_mixed_value=float(np.sqrt(2)*kv)))
    assert rows[0]['beta']>1 and rows[0]['certificate_bound'] is None
    return rows


def mp_gain(D,t,eta,epsilon,dps):
    V,U=scaled_operators(D,t)
    L=np.column_stack((t**D['rhoV']*V,eta*t**D['rhoU']*U))
    _,s,vh=svd(L,full_matrices=False)
    z=vh[0].conj()
    E=sum((z[i]*B for i,B in enumerate(D['BV'])),np.zeros_like(D['X']))
    E+=eta*sum((z[len(D['BV'])+i]*B for i,B in enumerate(D['BU'])),np.zeros_like(D['X']))
    with mp.workdps(dps):
        X=mp.matrix(D['X'].tolist());Em=mp.matrix(E.tolist())
        norm=lambda M:mp.sqrt(sum(abs(x)**2 for x in M))
        delta=mp.mpf(str(epsilon))*t*norm(X)*Em
        F=mp.expm(t*X)
        gain=norm(mp.expm(t*X+delta)-F)/(mp.mpf(str(epsilon))*norm(F))
        # Preserve digits for the independent 80/100-digit comparison.
        return mp.nstr(gain,70),float(s[0])


def check_exponential_differences():
    result=[]
    for name,D in [('shift_d4',shift_data(4)),('markov_H6',markov_data())]:
        for t in (10.,100.):
            discrepancies=independent_frechet(D,t)
            V,U=scaled_operators(D,t)
            eta_star=(t**D['rhoV']*opnorm(V))/(t**D['rhoU']*opnorm(U))
            for factor in (.3,1.,3.):
                eta=factor*eta_star
                gain,k=mp_gain(D,t,eta,1e-12,80)
                half,_=mp_gain(D,t,eta,5e-13,80)
                higher,_=mp_gain(D,t,eta,1e-12,100)
                with mp.workdps(80):
                    precision=float(abs(mp.mpf(gain)/mp.mpf(higher)-1))
                    halving=float(abs(mp.mpf(gain)/mp.mpf(half)-1))
                    error=float(abs(mp.mpf(gain)/k-1))
                assert precision<1e-35
                assert error<2e-7 and halving<1e-7
                result.append(dict(case=name,t=t,factor=factor,eta=eta,gain=gain,kappa=k,
                    relative_error=error,halving_change=halving,precision_change=precision,
                    frechet_discrepancies=discrepancies))
    return result


def make_mixed_table():
    """Generate the finite-t mixed-amplification table in the manuscript."""
    rows=[]
    tex=[]
    for name,D,label in [('shift_d4',shift_data(4),r'$S$, $d=4$'),
                         ('markov_H6',markov_data(),r'Markov, $d=12$')]:
        t=100.
        V,U=scaled_operators(D,t)
        LV=t**D['rhoV']*V
        LU=t**D['rhoU']*U
        eta_star=opnorm(LV)/opnorm(LU)
        split=LV.shape[1]
        for factor in (.3,1.,3.):
            eta=factor*eta_star
            block=np.column_stack((LV,eta*LU))
            _,singular_values,vh=svd(block,full_matrices=False)
            z=vh[0].conj()
            weight_v=float(np.linalg.norm(z[:split])**2)
            weight_u=float(np.linalg.norm(z[split:])**2)
            np.testing.assert_allclose(weight_v+weight_u,1.,rtol=0,atol=3e-14)
            gain,kappa=mp_gain(D,t,eta,1e-12,80)
            half,_=mp_gain(D,t,eta,5e-13,80)
            higher,_=mp_gain(D,t,eta,1e-12,100)
            with mp.workdps(80):
                ratio=float(mp.mpf(gain)/kappa)
                precision=float(abs(mp.mpf(gain)/mp.mpf(higher)-1))
                halving=float(abs(mp.mpf(gain)/mp.mpf(half)-1))
            assert abs(kappa/singular_values[0]-1)<3e-14
            assert abs(ratio-1)<2e-7 and precision<1e-35 and halving<1e-7
            row=dict(case=name,t=t,factor=factor,eta_star=eta_star,eta=eta,
                     kappa=kappa,gain=gain,ratio=ratio,
                     weight_V=weight_v,weight_U=weight_u,
                     halving_change=halving,precision_change=precision)
            rows.append(row)
            tex.append(f"{label} & {factor:.1f} & {kappa:.5f} & "
                       f"{float(gain):.5f} & {ratio:.5f} & "
                       f"{weight_v:.6f} & {weight_u:.6f} \\\\")
    table=r"""% Generated by reproduce_structure_transition.py.
\begin{table}[tbp]
\caption{Mixed-perturbation amplification at $t=100$ and
$\varepsilon=10^{-12}$. Here $\eta_*=\kappa_V(tX)/\kappa_U(tX)$,
$G_\varepsilon$ uses the top right singular vector of
$[L_V(t),\eta L_U(t)]$, and $w_J=\lVert z_J\rVert_2^2$.
The exponentials were evaluated at 80 decimal digits.}
\label{tab:mixed}
\centering
\begin{tabular}{lrrrrrr}
\hline
$X$ & $\eta/\eta_*$ & $\kappa_\eta$ & $G_\varepsilon$ &
$G_\varepsilon/\kappa_\eta$ & $w_V$ & $w_U$ \\
\hline
"""+'\n'.join(tex)+r"""
\hline
\end{tabular}
\end{table}
"""
    (OUT/'mixed_transition_table.tex').write_text(table)
    return rows


def make_figure():
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,
                         'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    q=np.unique(np.r_[np.logspace(-2,np.log10(4),91),1.])
    fig,axes=plt.subplots(1,2,figsize=(6.35,2.55),layout='constrained')
    records=[]
    for ax,(name,D,title) in zip(axes,[('shift_d4',shift_data(4),'Shift, $d=4$'),
                                      ('markov_H6',markov_data(),'Layered Markov model, $d=12$')]):
        target=np.sqrt(1+q*q) if name=='shift_d4' else np.maximum(1,q)
        other=np.maximum(1,q) if name=='shift_d4' else np.sqrt(1+q*q)
        ax.plot(q,other,':',color='.65',lw=1,label='Other bound')
        ax.plot(q,target,'k--',lw=1.2,label='Limiting curve')
        for t,color,marker in [(10.,'#0072B2','o'),(100.,'#D55E00','s'),(1000.,'#009E73','^')]:
            V,U=scaled_operators(D,t)
            kv,ku=opnorm(V),opnorm(U)
            y=[]
            for x in q:
                y.append(opnorm(np.column_stack((V/kv,x*U/ku))))
            ax.plot(q,y,color=color,lw=.8,marker=marker,ms=3,markevery=13,
                    fillstyle='none',label=f'$t={t:g}$')
            records.append(dict(case=name,t=t,q=q.tolist(),normalized_condition=y,
                eta_star=float(t**(D['rhoV']-D['rhoU'])*kv/ku),
                ratio_at_balance=float(y[list(q).index(1.)])))
        ax.set_xscale('log');ax.set_xlim(.01,4);ax.set_ylim(.94,4.25)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel(r'$q=\eta\kappa_U(tX)/\kappa_V(tX)$')
        ax.grid(True,alpha=.2,lw=.5)
    axes[0].set_ylabel(r'$\kappa_\eta(tX)/\kappa_V(tX)$')
    axes[0].legend(loc='upper left',frameon=False)
    fig.savefig(OUT/'structure_transition.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    return records


def main():
    results=dict(numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,
                 matplotlib=matplotlib.__version__)
    results['exact_block_checks']=check_exact_bounds()
    results['rank_one_angle_checks']=check_angle()
    results['transitions']=verify_transitions()
    results['exact_five_state']=verify_exact_five_state()
    results['slow_convergence_d6']=slow_convergence_check()
    results['exponential_differences']=check_exponential_differences()
    results['mixed_table']=make_mixed_table()
    results['figure']=make_figure()
    D=markov_data()
    results['markov_constants']=dict(KV=opnorm(D['AV']),KU=opnorm(D['AU']),
        balance_constant=opnorm(D['AV'])/opnorm(D['AU']),
        leading_rankV=int(np.linalg.matrix_rank(D['AV'])),
        leading_rankU=int(np.linalg.matrix_rank(D['AU'])))
    (OUT/'structure_transition_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(dict(exact_block=results['exact_block_checks'],
        rank_one_angle_checks=results['rank_one_angle_checks'],
        finite_bound_checks=results['transitions']['finite_bound_checks'],
        markov_exact_support_checks=len(results['transitions']['markov_exact_support_checks']),
        exact_five_state_checks=sum(len(x['measurements']) for x in results['exact_five_state']),
        exponential_checks=len(results['exponential_differences']),
        max_forward_discrepancy=max(x['relative_error'] for x in results['exponential_differences']),
        markov=results['markov_constants']),indent=2))


if __name__=='__main__':
    main()
