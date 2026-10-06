"""Check observed backward and forward errors of SciPy expm.
High-precision logarithms are numerical, not exact certificates.
Run with the library versions in requirements.txt. Outputs are observations,
not a stability theorem for expm. No random cases depend on global RNG state.
"""
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import expm
import mpmath as mp
OUT=Path(__file__).resolve().parent

def evaluate(A,Y,dps):
    with mp.workdps(dps):
        A=mp.matrix(A.tolist());Y=mp.matrix(Y.tolist());d=A.rows
        norm=lambda B:mp.sqrt(sum(abs(x)**2 for x in B))
        F=mp.expm(A); E=mp.logm(Y)-A
        D=mp.diag([E[i,i] for i in range(d)]); N=E-D
        powers=[mp.eye(d)]
        for _ in range(1,d):powers.append(powers[-1]*A)
        def derivative(B):
            result=mp.zeros(d)
            for p in range(d):
                for q in range(d):result+=powers[p]*B*powers[q]/mp.factorial(p+q+1)
            return result
        LN=derivative(N);LD=derivative(D); L=LN+LD
        forward=norm(Y-F)/norm(F)
        vals=dict(backward=float(norm(E)/norm(A)),diagonal_backward=float(norm(D)/norm(A)),
            strict_upper_backward=float(norm(N)/norm(A)),lower_backward=float(mp.sqrt(sum(abs(E[i,j])**2 for i in range(d) for j in range(i)))/norm(A)),
            forward=float(forward),linearized_forward=float(norm(L)/norm(F)),
            strict_upper_forward=float(norm(LN)/norm(F)),diagonal_forward=float(norm(LD)/norm(F)),
            relative_linearization_residual=float(norm(Y-F-L)/norm(Y-F)),
            logarithm_reconstruction_residual=float(norm(mp.expm(A+E)-Y)/norm(Y)))
        return vals

def main():
    rng=np.random.default_rng(0)
    families=[('shift4',np.diag(np.ones(3),1)),('shift6',np.diag(np.ones(5),1)),
              ('random6',np.triu(rng.standard_normal((6,6)),1))]
    rows=[]
    for name,X in families:
        for t in (1.,10.,100.):
            A=t*X; Y=expm(A)
            assert np.count_nonzero(np.tril(Y,-1))==0
            a=evaluate(A,Y,60);b=evaluate(A,Y,100)
            for k in ('backward','diagonal_backward','forward','linearized_forward','strict_upper_forward','diagonal_forward'):
                assert abs(a[k]-b[k])<=1e-9*max(abs(b[k]),1e-100),(name,t,k)
            assert b['relative_linearization_residual']<1e-5
            assert b['logarithm_reconstruction_residual']<1e-85
            rows.append(dict(case=name,t=t,input_norm=float(np.linalg.norm(A)),**b))
    result=dict(numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,seed=0,precision_checks=[60,100],results=rows)
    (OUT/'backward_error_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
