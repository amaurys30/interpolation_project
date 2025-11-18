import numpy as np

def lagrange_evaluator(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n = len(xs)
    def L(xq):
        xq = np.asarray(xq)
        out = np.zeros_like(xq, dtype=float)
        for i in range(n):
            Li = np.ones_like(xq, dtype=float)
            for j in range(n):
                if j==i: continue
                Li *= (xq - xs[j])/(xs[i]-xs[j])
            out += ys[i]*Li
        return out
    return L

def lagrange_coeffs(xs, ys):
    # returns coefficients in descending powers using Vandermonde solve
    V = np.vander(xs, increasing=False)
    coeffs = np.linalg.solve(V, ys)
    return np.asarray(coeffs)  # highest->lowest
