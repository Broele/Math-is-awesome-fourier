"""
This module contain functions to compute Fourier coefficients of different types of curves.

Notes
-----
The functions in this package are designed to handle piecewise defined functions.
For this reason, each function can compute the contribution to the Fourier coefficient if the curve is just a piece of
a bigger curve.
"""

import numpy as np


def transform_straight_line(p1, p2, lam, a=0, b=1):
    """
    Computes the contribution of a straight line to the Fourier coefficients.

    Parameters
    ----------
    p1: complex
        Starting point of the line. Either as complex number or as vector
    p2: complex
        Endpoint of the line. Either as complex number or as vector
    lam: float or array_like
        One or multiple lambda values
    a: float
        Start of the interval
    b: float
        End of the interval

    Returns
    -------
    complex
        Contribution to the Fourier coefficients
    """
    i = 1j  # Just for shorter notations

    lam = np.asarray(lam, dtype=float)
    result = np.zeros(shape=lam.shape, dtype=np.complex)

    # Handle Case lambda != 0
    l_ = lam[lam != 0]
    result[lam != 0] = i * np.exp(-i * l_ * b) * p2 / l_ \
                       - i * np.exp(-i * l_ * a) * p1 / l_ \
                       + (p2 - p1) * (np.exp(-i * l_ * b) - np.exp(-i * l_ * a)) / (l_ * l_ * (b - a))

    # Handle case lambda = 0
    result[lam == 0] = (p1 + p2) * (b - a) / 2

    # Return results
    return result
