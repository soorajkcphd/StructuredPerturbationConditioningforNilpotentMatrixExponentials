"""Check stored Section 6.1 outputs and an independent block-exponential derivative."""
import csv
import json
from pathlib import Path
import numpy as np
import mpmath as mp
from reproduce_backward_error import evaluate

OUT=Path(__file__).resolve().parent

def main():
    data=json.loads((OUT/'expcompare_results.json').read_text())
    rows=data['results']
    with (OUT/'expcompare.csv').open() as f: csv_rows=list(csv.DictReader(f))
    assert len(rows)==len(csv_rows)==len(data['computed_matrices'])==12
    max_block=0.
    for r,c,s in zip(rows,csv_rows,data['computed_matrices']):
        for k,v in r.items():
            assert c[k]==v if isinstance(v,str) else float(c[k])==v
        A=np.array([[float.fromhex(x) for x in row] for row in s['A_hex']])
        Y=np.array([[float.fromhex(x) for x in row] for row in s['Y_hex']])
        check=evaluate(A,Y,80)
        for k,v in check.items():
            # Reconstruction residual depends on precision; all are negligible.
            if k=='logarithm_reconstruction_residual':
                assert v<1e-60
            else:
                assert abs(v-r[k])<1e-20*max(abs(r[k]),1e-20),(s['family'],k,v,r[k])
        assert r['forward']>0 and r['relative_linearization_residual']<6.3e-12
        np.testing.assert_allclose(r['p_split'],r['k_n']*r['strict_upper_backward']+
                                   r['k_d']*r['diagonal_backward'],rtol=1e-15)
        assert r['linearized_forward'] <= min(r['p_split'],r['p_b'],r['p_full'])*(1+1e-10)
        # Independent derivative evaluation via the upper-right exponential block.
        if r['method']=='expm' and r['t']==100:
            with mp.workdps(80):
                AA=mp.matrix(A.tolist()); YY=mp.matrix(Y.tolist()); d=len(A)
                E=mp.logm(YY)-AA; B=mp.zeros(2*d)
                for i in range(d):
                    for j in range(d):
                        B[i,j]=B[i+d,j+d]=AA[i,j];B[i,j+d]=E[i,j]
                Z=mp.expm(B); L=mp.matrix([[Z[i,j+d] for j in range(d)] for i in range(d)])
                norm=lambda M:mp.sqrt(sum(abs(x)**2 for x in M))
                residual=float(norm(YY-mp.expm(AA)-L)/norm(YY-mp.expm(AA)))
                err=abs(residual-r['relative_linearization_residual'])
                assert err<1e-20
                max_block=max(max_block,err)
    print('PASS: 12 stored outputs and CSV checked at 80 digits; two independent block-derivative checks.')
    print('Maximum matrix linearization residual:',max(r['relative_linearization_residual'] for r in rows))
    print('Maximum absolute discrepancy in independent residual checks:',max_block)

if __name__=='__main__':main()
