import numpy as np
from scipy.interpolate import CubicSpline

def cubic_spline_model(xs, ys, bc_type='natural'):
    # bc_type: 'natural' or 'clamped' (clamped requires first derivatives as tuple)
    cs = CubicSpline(xs, ys, bc_type='natural')
    return cs  # cs(xq) evalúa

