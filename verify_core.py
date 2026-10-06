"""Independent regression checks for the shortened manuscript's formulas."""
from math import factorial
import numpy as np

rng = np.random.default_rng(20260922)


def profile(x):
    d = len(x)
    powers = [np.eye(d, dtype=object)]
    for _ in range(d):
        powers.append(powers[-1] @ x)
    m = next(k for k in range(1, d + 1) if not np.any(powers[k]))
    left = [max(k for k in range(m) if np.any(powers[k][:, a]))
            for a in range(d)]
    right = [max(k for k in range(m) if np.any(powers[k][b, :]))
             for b in range(d)]
    r = max(left[a] + right[b] for a in range(d) for b in range(a + 1, d))
    return m, r, left, right, powers


directions = 0
for d in range(2, 7):
    for _ in range(24):
        x = np.triu(rng.integers(-2, 3, size=(d, d)), 1).astype(object)
        m, r, left, right, powers = profile(x)
        assert r <= min(d - 2, 2 * m - 2)
        if m >= 2:
            assert r >= m - 2
        for a in range(d):
            for b in range(d):
                e = np.zeros((d, d), dtype=object)
                e[a, b] = 1
                degree = 0
                for n in range(2 * m - 1):
                    coefficient = sum(
                        (powers[p] @ e @ powers[n - p]
                         for p in range(m) if 0 <= n - p < m),
                        np.zeros((d, d), dtype=object),
                    )
                    if np.any(coefficient):
                        degree = n
                assert degree == left[a] + right[b]
                directions += 1

realizations = 0
for d in range(2, 13):
    for m in range(2, d + 1):
        for ell in range(min(m, d - m) + 1):
            x = np.zeros((d, d), dtype=object)
            if ell == 0:
                x[m - 2, d - 1] = 1
                for k in range(2, m):
                    x[k - 2, k - 1] = 1
            else:
                for i in range(1, m):
                    x[i - 1, i] = 1
                for i in range(m + 1, m + ell):
                    x[i - 1, i] = 1
            actual_m, r, *_ = profile(x)
            assert actual_m == m and r - m + 2 == ell
            realizations += 1

constants = 0
for d in range(2, 8):
    for complex_field in (False, True):
        x = np.triu(rng.normal(size=(d, d)), 1)
        if complex_field:
            x = x + 1j * np.triu(rng.normal(size=(d, d)), 1)
        weights = np.diag(x, 1)
        powers = [np.linalg.matrix_power(x, k) for k in range(d)]
        columns = []
        for a in range(d):
            for b in range(a + 1, d):
                e = np.zeros_like(x)
                e[a, b] = 1
                lead = sum((powers[p] @ e @ powers[d - 2 - p]
                            for p in range(d - 1)), np.zeros_like(x))
                columns.append(lead.ravel() / factorial(d - 1))
        mat = np.column_stack(columns)
        gamma = np.linalg.svd(mat, compute_uv=False)[0]
        k_direct = (factorial(d - 1) * np.linalg.norm(x) * gamma
                    / np.linalg.norm(powers[d - 1]))
        k_formula = np.linalg.norm(x) * np.linalg.norm(1 / weights)
        np.testing.assert_allclose(k_direct, k_formula, rtol=5e-13)
        assert k_formula >= d - 1 - 1e-12
        e = np.diag(1 / np.conj(weights), 1)
        lead_e = sum((powers[p] @ e @ powers[d - 2 - p]
                      for p in range(d - 1)), np.zeros_like(x))
        np.testing.assert_allclose(
            np.linalg.norm(lead_e) / factorial(d - 1),
            gamma * np.linalg.norm(e), rtol=5e-13,
        )
        constants += 1

bands = 0
for d in range(2, 13):
    s = np.diag(np.ones(d - 1), 1)
    powers = [np.linalg.matrix_power(s, k) for k in range(d)]
    for nu in range(d):
        degree = d - 1 - nu
        columns = []
        for i in range(d - nu):
            e = np.zeros((d, d))
            e[i, i + nu] = 1
            lead = sum(
                (powers[p] @ e @ powers[degree - p]
                 for p in range(degree + 1)),
                np.zeros((d, d)),
            ) / factorial(degree + 1)
            columns.append(lead.ravel())
        matrix = np.column_stack(columns)
        gamma = np.linalg.svd(matrix, compute_uv=False)[0]
        expected_gamma = np.sqrt(d - nu) / factorial(d - nu)
        np.testing.assert_allclose(gamma, expected_gamma, rtol=5e-14)
        optimal = np.ones(d - nu) / np.sqrt(d - nu)
        np.testing.assert_allclose(
            np.linalg.norm(matrix @ optimal), gamma, rtol=5e-14
        )
        expected_k = (
            factorial(d - 1)
            / factorial(d - nu)
            * np.sqrt((d - 1) * (d - nu))
        )
        np.testing.assert_allclose(
            factorial(d - 1) * np.linalg.norm(s) * gamma,
            expected_k,
            rtol=5e-14,
        )
        bands += 1

print(f"Exact signed-integer directional checks passed: {directions}")
print(f"All exponent constructions, dimensions 2 through 12: {realizations}")
print(f"Maximal-index constants and real/complex extremizers: {constants}")
print(f"Superdiagonal leading maps and optimal directions: {bands}")
print("These finite regression checks supplement, and do not replace, proofs.")
