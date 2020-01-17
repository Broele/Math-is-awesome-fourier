"""
This module contain functions to compute Fourier coefficients of different types of curves.

Notes
-----
The functions in this package are designed to handle piecewise defined functions.
For this reason, each function can compute the contribution to the Fourier coefficient if the curve is just a piece of
a bigger curve.
"""

import numpy as np
from scipy.special import comb, perm


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


def _exp_integral(x, a, b):
    """
    Computes the integral of an exponential function ``exp(x*t)`` for ``t`` from `a` to `b`

    Notes
    -----
    This is a helper function to compute

    Parameters
    ----------
    x: float or complex or array_like
        Factor in the exponent of the exponential. It's possible to provide multiple values of `x` at once.
    a: float
        Start of the interval
    b: float
        End of the interval

    Returns
    -------
    ndarray
        The integral for all values of `x`
    """
    x = np.asarray(x)
    result = np.zeros(shape=x.shape, dtype=np.complex)

    # Handle Case x != 0
    x_ = x[x != 0]
    result[x != 0] = (np.exp(x_ * b) - np.exp(x_ * a)) / x_

    # Handle case x = 0
    result[x == 0] = b - a

    # Return results
    return result


def transform_arc(p, r1, r2, phi, theta1, theta2, lam, a, b):
    """
    Computes the contribution of an ellipse arc to the Fourier coefficients.

    Parameters
    ----------
    p: complex
        Center point of the ellipse
    r1: float
        First radius of the ellipse
    r2: float
        Second radius of the ellipse
    phi: float
        Rotation of the ellipse (in radians)
    theta1: float
        Start angle of the arc (in radians)
    theta2
        End angle of the arc (in radians)
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

    lam = np.asarray(lam)

    # Compute summand 1
    s1 = p * _exp_integral(-i * lam, a, b)

    # Compute summand 2
    e2 = i * (phi + theta1 - (theta2 - theta1) / (b - a) * a)
    x2 = i * (theta2 - theta1) / (b - a) - i * lam
    s2 = (r1 + r2) / 2 * np.exp(e2) * _exp_integral(x2, a, b)

    # Compute summand 3
    e3 = i * (phi - theta1 + (theta2 - theta1) / (b - a) * a)
    x3 = - i * (theta2 - theta1) / (b - a) - i * lam
    s3 = (r1 - r2) / 2 * np.exp(e3) * _exp_integral(x3, a, b)

    # Return the sum of all three
    return s1 + s2 + s3


def transform_bezier(p, lam, a=0, b=1):
    """
    Computes the contribution of a Bézier curve to the Fourier coefficients.

    Parameters
    ----------
    p: array_like
        Control points of the Bézier curve
    lam: float or array_like
        One or multiple lambda values
    a: float
        Start of the interval
    b: float
        End of the interval

    Returns
    -------

    """
    i = 1j  # Just for shorter notations

    # Reshape lambda
    l = np.asarray(lam)
    s = l.shape
    l = np.reshape(l, (-1))

    # Create array for the results
    result = np.zeros(shape=l.shape, dtype=np.complex)

    # Reshape points
    p = np.asarray(p)
    p = np.reshape(p, (-1))
    n = np.prod(p.shape) - 1

    # Handle Case k != 0
    l_ = np.reshape(l[l != 0], (-1))

    # Iterate over r:
    for r in range(n + 1):
        j = np.reshape(np.arange(r + 1, dtype=int), (1, -1))

        # j as row vector, lambda as a column vector --> broadcasting gives a matrix
        u = comb(r, j) * np.power(-1, r - j) * (p[n - r + j] * np.exp(-i * (b - a) * np.reshape(l_, (-1, 1))) - p[j])
        result[l != 0] = result[l != 0] + np.sum(u, axis=1) * perm(n, r) / np.power(i * (b - a) * l_, r + 1)

    # Add factor
    result[l != 0] = -(b - a) * np.exp(-i * l_ * a) * result[l != 0]

    # Handle case lambda = 0
    result[l == 0] = (b - a) * np.sum(p) / (n + 1)

    result = np.reshape(result, s)

    # Return results
    return result

