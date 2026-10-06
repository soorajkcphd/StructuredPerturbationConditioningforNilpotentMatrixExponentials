# Structured Perturbation Conditioning for Nilpotent Matrix Exponentials

Reproducibility material for the article *Exact Fréchet-derivative
degrees and structured conditioning of the matrix exponential on strictly
upper-triangular matrices* by Sooraj K.C and Vivek Mishra.

The saved data and Python scripts reproduce Tables 6.1--6.3 and 7.1 and
the bandwise diagnostic discussed in Section 6.1. All mathematical results
and proofs are contained in the article.

Install requirements.txt in an isolated Python environment. The saved
data use Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, and mpmath 1.3.0.

Commands and the current manuscript tables:

    python verify_core.py
    python verify_expcompare.py
    python reproduce_expcompare.py       # Table 6.1
    python superdiag_decomp.py            # Section 6.1 diagnostic
    python reproduce_amplification.py    # Table 6.2
    python reproduce_weak_link.py        # Table 6.3
    python reproduce_markov.py           # Table 7.1

verify_expcompare.py verifies the saved binary64 outputs at high precision;
run it before reproducing Table 6.1 and the bandwise diagnostic. The
superdiagonal script writes all seven rows for both recorded methods to
superdiag_results.json.
Work in a copy of the extracted
archive, since generators overwrite the saved result files and new expm
runs can differ by platform. The above generators write TeX
fragments beside the scripts; they do not rewrite the single-file source.

reproduce_structure_transition.py supplies additional checks of the
uniform-in-eta bound, block identities, and output-range geometry. Its
figure and mixed-perturbation table are not printed in the short manuscript.
reproduce_table.py and reproduce_backward_error.py also supply imported
helpers; their standalone tables are additional checks. All mathematical
proofs needed by the article are in its text.
