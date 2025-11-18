import numpy as np

def divided_differences(xs, ys):
    n = len(xs)
    dd = np.zeros((n, n))
    dd[:,0] = ys
    for j in range(1, n):
        for i in range(n-j):
            dd[i,j] = (dd[i+1,j-1] - dd[i,j-1]) / (xs[i+j] - xs[i])
    coeffs = dd[0,:]
    return coeffs  # a0, a1, a2...

def newton_evaluator(xs, coeffs):
    xs = np.asarray(xs)
    def P(xq):
        xq = np.asarray(xq)
        n = len(coeffs)
        y = np.full_like(xq, coeffs[0], dtype=float)
        for i in range(1,n):
            term = coeffs[i]
            for j in range(i):
                term = term * (xq - xs[j])
            y = y + term
        return y
    return P
