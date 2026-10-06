"""Regenerate Section 6.1 from two fully specified matrices (12 cases).

Run python reproduce_expcompare.py. The original supplied CSV is archived
separately and is not an input. Stored binary64 matrices permit diagnostics
to be checked independently of platform-dependent expm rounding.
"""
import csv
import json
import math
import platform
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import expm, svdvals

from reproduce_backward_error import evaluate
from reproduce_markov import model
from render_expcompare import render

OUT = Path(__file__).resolve().parent


def condition_numbers(A):
    d = len(A)
    powers = [np.eye(d)]
    for _ in range(1, d):
        powers.append(powers[-1] @ A)
    F = sum((p / math.factorial(k) for k, p in enumerate(powers)), np.zeros_like(A))
    columns = []
    for i in range(d):
        for j in range(d):
            # A^p E_ij A^q = outer(A^p[:,i], A^q[j,:]).
            col = sum((np.outer(powers[p][:, i], powers[q][j, :])
                       / math.factorial(p+q+1)
                       for p in range(d) for q in range(d)), np.zeros_like(A))
            columns.append(col.ravel())
    M = np.column_stack(columns)
    factor = np.linalg.norm(A) / np.linalg.norm(F)
    tests = {'n': lambda i,j: i<j, 'b': lambda i,j: i<=j,
             'd': lambda i,j: i==j, 'full': lambda i,j: True}
    return {name: float(factor * svdvals(M[:, [i*d+j for i in range(d)
                   for j in range(d) if test(i,j)]])[0]) for name,test in tests.items()}


def polynomial(A):
    # Sequential term recurrence fixes the evaluation order explicitly.
    term = np.eye(len(A)); Y = term.copy()
    for k in range(1, len(A)):
        term = (term @ A) / k
        Y = Y + term
    return Y


def main():
    raw, _ = model(6)
    matrices = {'S8': np.diag(np.ones(7), 1), 'L12': np.array(raw, dtype=float)/8}
    rows, saved = [], []
    for name,X in matrices.items():
        for t in (1,10,100):
            A = t*X
            kappas = condition_numbers(A)
            for method, Y in [('poly', polynomial(A)), ('expm', expm(A))]:
                assert np.count_nonzero(np.tril(Y, -1)) == 0
                assert np.all(np.diag(Y)>0)  # principal logarithm exists
                low, high = evaluate(A,Y,60), evaluate(A,Y,100)
                keys = ('backward','strict_upper_backward','diagonal_backward',
                        'forward','linearized_forward','strict_upper_forward','diagonal_forward')
                precision_change = max(abs(low[k]-high[k])/max(abs(high[k]),1e-100) for k in keys)
                assert precision_change < 1e-12, (name,t,method,precision_change)
                assert abs(low['relative_linearization_residual']-high['relative_linearization_residual']) < 1e-20
                assert high['logarithm_reconstruction_residual'] < 1e-80
                assert high['relative_linearization_residual'] < 1e-7
                if method=='poly':
                    assert high['diagonal_backward']==high['lower_backward']==0
                row = dict(family=name,d=len(X),m=8 if name=='S8' else 7,t=t,method=method,
                           **high,precision_relative_change=precision_change)
                row.update({'k_'+k:v for k,v in kappas.items()})
                row['p_n'] = kappas['n']*high['strict_upper_backward']
                row['p_b'] = kappas['b']*high['backward']
                row['p_full'] = kappas['full']*high['backward']
                row['p_split'] = row['p_n']+kappas['d']*high['diagonal_backward']
                assert high['strict_upper_forward'] <= row['p_n']*(1+1e-10)
                assert high['diagonal_forward'] <= kappas['d']*high['diagonal_backward']*(1+1e-10)
                assert high['linearized_forward'] <= min(row['p_b'],row['p_full'],row['p_split'])*(1+1e-10)
                rows.append(row)
                saved.append(dict(family=name,t=t,method=method,
                                  A_hex=[[float(x).hex() for x in r] for r in A],
                                  Y_hex=[[float(x).hex() for x in r] for r in Y]))
    metadata = dict(python=platform.python_version(),platform=platform.platform(),
                    numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,
                    precision_digits=[60,100],
                    matrices={k:v.tolist() for k,v in matrices.items()},computed_matrices=saved,results=rows)
    (OUT/'expcompare_results.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (OUT/'expcompare.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    render(rows)
    print(json.dumps(dict(cases=len(rows),max_residual=max(r['relative_linearization_residual'] for r in rows),
                         max_precision_change=max(r['precision_relative_change'] for r in rows),
                         headline=next(r for r in rows if r['family']=='S8' and r['method']=='expm' and r['t']==100)),indent=2))


if __name__=='__main__': main()
