"""Reproduce the forward-amplification experiment in Section 6.2.

Run: python3 reproduce_amplification.py
Dependencies: NumPy, SciPy, mpmath (tested with 2.1.3, 1.15.3, 1.3.0).
The experiment uses actual exponential differences evaluated by mpmath,
not a derivative in place of the perturbed exponential. The derivative
provides a separate linearization check. All base matrices and unnormalized
directions have exact integer entries. Precision and epsilon checks are
recorded in amplification_results.json; the manuscript table is generated
as amplification_table.tex. No random numbers are used.
"""

import json
from math import factorial
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import svdvals

from reproduce_table import derivative_matrix


def unit(d, i, j):
    """Matrix unit with mathematical (one-based) indices."""
    a = np.zeros((d, d), dtype=int)
    a[i - 1, j - 1] = 1
    return a


def cases():
    result = []
    for d in (4, 6):
        s = np.diag(np.ones(d - 1, dtype=int), 1)
        for space, direction, rho in (
            ("strict", s, 0),
            ("upper", np.eye(d, dtype=int), 1),
            ("full", unit(d, d, 1), d),
        ):
            result.append(dict(name=f"S{d}_{space}", label="S", d=d,
                               space=space, x=s, e=direction, m=d, rho=rho))
    d = 6
    x0 = unit(d, 1, 2) + unit(d, 2, 6)
    result.append(dict(name="X0", label="X_0", d=d, space="strict",
                       x=x0, e=x0, m=3, rho=0))
    for ell in (1, 2, 3):
        x = unit(d, 1, 2) + unit(d, 2, 3)
        for i in range(4, 3 + ell):
            x += unit(d, i, i + 1)
        result.append(dict(name=f"X{ell}", label=f"X_{ell}", d=d,
                           space="strict", x=x, e=unit(d, 3, 4),
                           m=3, rho=ell))
    return result


def exact_matrix(a):
    return mp.matrix([[int(value) for value in row] for row in a])


def fro(a):
    return mp.sqrt(mp.fsum(abs(value) ** 2 for value in a))


def asymptotic_constant(case):
    d, rho = case["d"], case["rho"]
    if case["label"] == "S":
        return {"strict": mp.mpf(d - 1),
                "upper": mp.sqrt(mp.mpf(d - 1) / d),
                "full": mp.factorial(d - 1) * mp.sqrt(d - 1)
                        / mp.factorial(2 * d - 1)}[case["space"]]
    if rho == 0:
        return mp.mpf(2)
    return (2 * mp.sqrt(rho + 1)
            / (mp.factorial(rho + 2) * mp.sqrt(1 + int(rho == 3))))


def finite_series(a, d):
    powers = [mp.eye(d)]
    for _ in range(1, d):
        powers.append(powers[-1] * a)
    f = mp.zeros(d)
    for k in range(d):
        f += powers[k] / mp.factorial(k)
    return f, powers


def measure(case, t, epsilon, precision):
    with mp.workdps(precision):
        d = case["d"]
        a = t * exact_matrix(case["x"])
        e = exact_matrix(case["e"])
        e /= fro(e)
        eps = mp.mpf(epsilon)
        f, powers = finite_series(a, d)
        assert fro(f - mp.expm(a)) / fro(f) < mp.mpf("1e-60")
        perturbation = eps * fro(a) * e
        perturbed = mp.expm(a + perturbation)
        gain = fro(perturbed - f) / (eps * fro(f))
        derivative = mp.zeros(d)
        for p in range(d):
            for q in range(d):
                derivative += powers[p] * e * powers[q] / mp.factorial(p + q + 1)
        linear_gain = fro(a) * fro(derivative) / fro(f)
        ratio = gain / (asymptotic_constant(case) * t ** case["rho"])
        return dict(gain=mp.nstr(gain, 70), ratio=mp.nstr(ratio, 70),
                    linear_gain=mp.nstr(linear_gain, 70),
                    relative_linearization_error=float(abs(gain / linear_gain - 1)))


def check_leading_map(case):
    """Independently assemble all leading columns and verify K and E."""
    x = case["x"].astype(float)
    d, m = case["d"], case["m"]
    r = case["rho"] + m - 2
    powers = [np.linalg.matrix_power(x, k) for k in range(d)]
    pairs = [(i, j) for i in range(d) for j in range(d)
             if (case["space"] == "full"
                 or (case["space"] == "upper" and i <= j)
                 or (case["space"] == "strict" and i < j))]
    columns = []
    for i, j in pairs:
        e = np.zeros_like(x)
        e[i, j] = 1
        lead = sum((powers[p] @ e @ powers[r - p]
                    for p in range(d) if 0 <= r - p < d), np.zeros_like(x))
        columns.append(lead.ravel(order="F") / factorial(r + 1))
    matrix = np.column_stack(columns)
    gamma = svdvals(matrix)[0]
    k = factorial(m - 1) * np.linalg.norm(x) * gamma / np.linalg.norm(powers[m - 1])
    np.testing.assert_allclose(k, float(asymptotic_constant(case)), rtol=2e-14)
    coefficients = np.array([case["e"][i, j] for i, j in pairs], dtype=float)
    coefficients /= np.linalg.norm(coefficients)
    np.testing.assert_allclose(np.linalg.norm(matrix @ coefficients), gamma, rtol=2e-14)
    return pairs


def main():
    mp.mp.dps = 80
    all_results, table_rows = [], []
    max_linear_error = max_precision_error = max_half_epsilon_change = 0.0
    max_derivative_error = 0.0
    for case in cases():
        pairs = check_leading_map(case)
        row = {key: value for key, value in case.items() if key not in ("x", "e")}
        row["base_matrix"] = case["x"].tolist()
        row["unnormalized_direction"] = case["e"].tolist()
        row["K"] = mp.nstr(asymptotic_constant(case), 45)
        row["measurements"] = []
        ratios, gains = [], []
        for t in (10, 100):
            baseline = measure(case, t, "1e-12", 80)
            higher = measure(case, t, "1e-12", 100)
            half = measure(case, t, "5e-13", 80)
            precision_error = float(abs(mp.mpf(baseline["gain"]) / mp.mpf(higher["gain"]) - 1))
            half_change = float(abs(mp.mpf(baseline["gain"]) / mp.mpf(half["gain"]) - 1))
            assert precision_error < 1e-35
            assert baseline["relative_linearization_error"] < 1e-6
            assert half_change < 1e-6
            max_linear_error = max(max_linear_error, baseline["relative_linearization_error"])
            max_precision_error = max(max_precision_error, precision_error)
            max_half_epsilon_change = max(max_half_epsilon_change, half_change)
            a = t * case["x"].astype(float)
            matrix, error = derivative_matrix(a, pairs)
            max_derivative_error = max(max_derivative_error, error)
            e = case["e"].astype(float)
            e /= np.linalg.norm(e)
            coefficients = np.array([e[i, j] for i, j in pairs])
            f = sum((np.linalg.matrix_power(a, k) / factorial(k)
                     for k in range(case["d"])), np.zeros_like(a))
            double_linear_gain = np.linalg.norm(a) * np.linalg.norm(matrix @ coefficients) / np.linalg.norm(f)
            np.testing.assert_allclose(double_linear_gain, float(baseline["linear_gain"]), rtol=3e-14)
            row["measurements"].append(dict(t=t, epsilon="1e-12", precision=80,
                                              **baseline, at_100_digits=higher,
                                              at_half_epsilon=half,
                                              relative_precision_change=precision_error,
                                              relative_half_epsilon_change=half_change,
                                              derivative_matrix_check_error=error))
            ratios.append(float(baseline["ratio"]))
            gains.append(mp.mpf(baseline["gain"]))
        slope = float(mp.log(gains[1] / gains[0]) / mp.log(10))
        row["observed_slope"] = slope
        all_results.append(row)
        space_label = {"strict": r"\nfrak(d)", "upper": r"\bfrak(d)", "full": r"\mathrm{full}"}[case["space"]]
        table_rows.append(f" ${case['label']}$ & {case['d']} & ${space_label}$ & {case['rho']} & "
                          f"{ratios[0]:.5f} & {ratios[1]:.5f} & {slope:.4f}" + r" \\")
        print(f"{case['name']}: rho={case['rho']}; ratios={ratios}; slope={slope:.6f}", flush=True)
    summary = dict(max_relative_linearization_error=max_linear_error,
                   max_relative_precision_change=max_precision_error,
                   max_relative_half_epsilon_change=max_half_epsilon_change,
                   max_derivative_matrix_check_error=max_derivative_error)
    print(json.dumps(summary, indent=2))
    metadata = dict(numpy=np.__version__, scipy=scipy.__version__, mpmath=mp.__version__,
                    epsilon="1e-12", baseline_decimal_digits=80, validation_decimal_digits=100)
    destination = Path(__file__).parent
    (destination / "amplification_results.json").write_text(
        json.dumps(dict(environment=metadata, validation=summary, cases=all_results), indent=2) + "\n")
    table = r"""% Generated by reproduce_amplification.py from actual exponential differences.
\begin{table}[tb]
\caption{Measured forward amplification with $\varepsilon=10^{-12}$ in
the asymptotically optimal directions from Section~\ref{sec:amplification}.
The normalized gains $R(t)$ and observed slopes $\widehat\rho$ are defined there.}
\label{tab:amplification}
\centering
\begin{tabular}{crcrrrr}
 $X$ & $d$ & $V$ & $\rho$ & $R(10)$ & $R(100)$ & $\widehat\rho$ \\
\hline
""" + "\n".join(table_rows[:6]) + "\n\\hline\n" + "\n".join(table_rows[6:]) + r"""
\end{tabular}
\end{table}
"""
    (destination / "amplification_table.tex").write_text(table)


if __name__ == "__main__":
    main()
