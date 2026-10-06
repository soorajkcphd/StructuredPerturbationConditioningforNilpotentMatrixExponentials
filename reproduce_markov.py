"""Reproduce the Markov table and check the new propositions independently.

Python 3, NumPy, SciPy. Run python reproduce_markov.py.
Integer arithmetic identifies zero coefficients exactly. SVDs and printed
error bounds are floating-point evaluations, not interval certificates.
The script is self-contained and does not use external data.
"""
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm, expm_frechet, svd, svdvals

OUT = Path(__file__).resolve().parent


def model(H=6, variant=0):
    sizes = [1] + [2]*(H-1) + [1]
    offsets = np.cumsum([0]+sizes)
    n = int(offsets[-1])
    A = np.zeros((n, n), dtype=object)  # A=8P, exactly integral.
    branching = []
    for h in range(H):
        for i in range(offsets[h], offsets[h+1]):
            if h == H-1:
                A[i, offsets[h+1]] = 8
            else:
                numerator = 2+((h+i+variant) % 5)
                j, k = offsets[h+1], offsets[h+1]+1
                A[i, j], A[i, k] = 8-numerator, numerator
                branching.append((i, j, k))
    return A, branching


def basis(A, branching, conserving):
    B = []
    for i, j, k in branching:
        E = np.zeros_like(A)
        E[i, j] = 1
        if conserving:
            E[i, k] = -1
            B.append(E)
        else:
            B.append(E)
            E = np.zeros_like(A)
            E[i, k] = 1
            B.append(E)
    return B


def exact_maps(A, H, B, conserving):
    n = len(A)
    AP = [np.eye(n, dtype=object)]
    for _ in range(H+1):
        AP.append(AP[-1]@A)
    assert np.any(AP[H]) and not np.any(AP[H+1])
    C = []
    zero = []
    for k in range(H):
        columns = []
        all_zero = True
        for E in B:
            num = sum((AP[p]@E@AP[k-p] for p in range(k+1)), np.zeros_like(A))
            all_zero = all_zero and not np.any(num)
            col = np.array(num, dtype=float).ravel()/(8**k*math.factorial(k+1))
            if conserving:
                col /= np.sqrt(2)
            columns.append(col)
        C.append(np.column_stack(columns))
        zero.append(bool(all_zero))
    r = max(k for k in range(H) if not zero[k])
    PP = [np.array(x, dtype=float)/(8**k) for k, x in enumerate(AP[:H+1])]
    return PP, C, r, zero


def relative_bound(t, PP, C, r):
    H = len(PP)-1
    norms = [float(svdvals(c)[0]) for c in C]
    gamma = norms[r]
    a = np.linalg.norm(PP[-1])/math.factorial(H)
    alpha = sum(t**(k-r)*norms[k]/gamma for k in range(r))
    beta = sum(t**(k-H)*np.linalg.norm(PP[k])/math.factorial(k)/a for k in range(H))
    scaled_D = sum(t**(k-r)*C[k] for k in range(r+1))
    scaled_F = sum(t**(k-H)*PP[k]/math.factorial(k) for k in range(H+1))
    ratio = (svdvals(scaled_D)[0]/gamma)/(np.linalg.norm(scaled_F)/a)
    bound = (alpha+beta)/(1-beta) if beta < 1 else None
    if bound is not None:
        assert abs(ratio-1) <= bound+1e-12
    return dict(t=t, alpha=float(alpha), beta=float(beta), bound=bound,
                ratio=float(ratio), observed_relative_error=float(abs(ratio-1)))


def validate_derivatives(P, B, C, conserving):
    normal = np.sqrt(2) if conserving else 1.
    BF = [np.array(b, dtype=float)/normal for b in B]
    discrepancies = []
    centered = []
    for t in (0.5, 2., 6.):
        operator = sum(t**k*c for k, c in enumerate(C))
        independent = np.column_stack([
            expm_frechet(t*P, E, compute_expm=False).ravel() for E in BF])
        err = np.linalg.norm(operator-independent)/np.linalg.norm(operator)
        assert err < 2e-13
        discrepancies.append(dict(t=t, relative_error=float(err)))
        if conserving:
            _, _, vt = svd(operator, full_matrices=False)
            E = sum(z*b for z, b in zip(vt[0], BF))
            np.testing.assert_allclose(np.linalg.norm(E), 1., atol=1e-14)
            np.testing.assert_allclose(E.sum(axis=1), 0., atol=1e-14)
            target = t*np.exp(-t)*(operator@vt[0]).reshape(P.shape)
            for eps in (1e-3, 1e-4, 1e-5):
                assert np.min(P+eps*E) >= 0 and np.min(P-eps*E) >= 0
                finite = (expm(t*(P+eps*E-np.eye(len(P))))
                          -expm(t*(P-eps*E-np.eye(len(P)))))/(2*eps)
                error = np.linalg.norm(finite-target)/np.linalg.norm(target)
                assert error < 1e-6
                centered.append(dict(t=t, epsilon=eps, relative_error=float(error)))
    return dict(frechet=discrepancies, centered_differences=centered)


def verify_family():
    results = []
    for H in range(2, 9):
        for variant in range(3):
            A, branches = model(H, variant)
            B = basis(A, branches, True)
            _, _, r, zero = exact_maps(A, H, B, True)
            assert r == H-2 and zero[H-1]
            results.append(dict(H=H, variant=variant, r=r, rho=-1))
    return results


def verify_general_bound():
    """Check the bound independently outside the probabilistic application."""
    rng = np.random.default_rng(20260923)
    count = 0
    max_error_over_bound = 0.
    for n in (3, 4, 5):
        for complex_case in (False, True):
            X = np.triu(rng.integers(-2, 3, (n,n)), 1).astype(complex if complex_case else float)
            for i in range(n-1):
                X[i,i+1] = 1+1j if complex_case else 1.
            PP = [np.linalg.matrix_power(X,k) for k in range(n)]
            for kind in ("strict", "full", "corner"):
                pairs = ([(0,n-1)] if kind == "corner" else
                         [(i,j) for i in range(n) for j in range(n) if kind=="full" or i<j])
                C = []
                for k in range(2*n-1):
                    cols = []
                    for i,j in pairs:
                        E = np.zeros_like(X)
                        E[i,j] = 1
                        s = sum((PP[p]@E@PP[k-p] for p in range(n) if 0<=k-p<n),
                                np.zeros_like(X))/math.factorial(k+1)
                        cols.append(s.ravel())
                    C.append(np.column_stack(cols))
                r = max(k for k, c in enumerate(C) if np.any(c))
                for t in (10.,100.,1000.):
                    x = relative_bound(t, PP, C, r)
                    if x["bound"] is not None:
                        count += 1
                        max_error_over_bound = max(max_error_over_bound,
                                                  x["observed_relative_error"]/x["bound"])
    return dict(successful_bound_checks=count, max_error_divided_by_bound=max_error_over_bound)


def main():
    H=6
    A, branching = model(H)
    P = np.array(A,dtype=float)/8
    results = dict(H=H, n=len(P), seed=20260923, numpy=np.__version__, scipy=scipy.__version__,
                   P=P.tolist(), spaces={}, exact_family=verify_family(), general_bound=verify_general_bound())
    rows = []
    for name, conserving in (("W0",False),("V",True)):
        B = basis(A, branching, conserving)
        PP,C,r,zeros = exact_maps(A,H,B,conserving)
        rho = r-H+1
        gamma = float(svdvals(C[r])[0])
        K = math.factorial(H)*np.linalg.norm(P)*gamma/np.linalg.norm(PP[-1])
        validation=validate_derivatives(P,B,C,conserving)
        measurements=[relative_bound(t,PP,C,r) for t in (10.,100.,1000.)]
        results["spaces"][name]=dict(dimension=len(B), r=r, rho=rho, K=float(K),
            gamma=gamma, coefficient_norms=[float(svdvals(c)[0]) for c in C],
            exact_zero_coefficients=zeros, measurements=measurements, validation=validation)
        label=r"$W_0$" if name=="W0" else r"$V$"
        rows.append(f"{label} & {r} & {rho} & {K:.5f} & "
                    + " & ".join(f"{v['ratio']:.5f}" for v in measurements)+r" \\")
    B=basis(A,branching,True)
    PP,C,r,_=exact_maps(A,H,B,True)
    results['regime_limits']=[dict(**relative_bound(t,PP,C,r),
        survival=float(math.exp(-t)*sum(t**k/math.factorial(k) for k in range(H+1))))
        for t in (10.,20.,30.,50.,100.)]
    policy=results["spaces"]["V"]["measurements"][1]
    assert policy["bound"] is not None
    table=r"""% Generated by reproduce_markov.py; ratios rounded to five decimals.
\begin{table}[h!]
\caption{Degrees, leading constants, and normalized condition numbers
for the layered Markov example. For $J\in\{W_0,V\}$, set
$R_J(t):=\kappa_J(tP)/(K_J(P)t^{\rho_J(P)})$.
Only $V$ enforces row conservation.}
\label{tab:markov}
\centering
\begin{tabular}{lrrrrrr}
$J$ & $r_J$ & $\rho_J$ & $K_J$ & $R_J(10)$ & $R_J(100)$ & $R_J(1000)$ \\
\hline
"""+"\n".join(rows)+r"""
\end{tabular}
\end{table}
"""
    (OUT/'markov_table.tex').write_text(table)
    (OUT/'markov_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({"spaces":results["spaces"], "exact_family_checks":len(results["exact_family"]),
                      "general_bound":results["general_bound"]},indent=2))


if __name__ == '__main__':
    main()
