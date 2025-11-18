import numpy as np

def polyfit_model(xs, ys, degree):
    coeffs = np.polyfit(xs, ys, degree)
    p = np.poly1d(coeffs)
    def pred(xq):
        return p(xq)
    return {'coeffs': coeffs, 'poly': p, 'predict': pred}
