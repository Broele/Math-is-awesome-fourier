"""
This module contains functions to transform svg paths
"""

import numpy as np
from svgpathtools import svg2paths, Path, Line, Arc, QuadraticBezier, CubicBezier

from .transform import transform_straight_line, transform_arc, transform_bezier


def transform_path_segment(segment, k, a=0, b=2 * np.pi):
    """
    Computes the contribution of the segment to the Fourier coefficients.

    The segment can be any of the following elements from the `svgpathtools` package
    - a Path
    - a straight line
    - an ellipse arc
    - a quadratic Bézier curve
    - a cubic Bézier curve

    Parameters
    ----------
    segment: Path or Line Or Arc or QuadraticBezier or CubicBezier
        Curve segment of which the Fourier coefficient's contribution will be computed
    k: int or array_like
        Index of the Fourier coefficients
    a: float
        Defines the start of the interval of this segment
    b: float
        Defines the end of the interval of this segment

    Returns
    -------
    c: array_like
        Complex array of Fourier coefficients corresponding to `k`
    """
    if isinstance(segment, (Path, list)):
        # Path: Iterate of path segments

        # Use length of path to compute borders
        L = [0] + [s.length() for s in segment]
        L = np.cumsum(L) / np.sum(L)
        L = a + (b - a) * L

        # Compute contribution for the segments of the path
        # and sum them to get the total coefficients / contribution of the path
        c = np.sum([
            transform_path_segment(s, k, a=L[i], b=L[i + 1])
            for i, s in enumerate(segment)
        ], axis=0)

    elif isinstance(segment, Line):
        # Straight line

        p1 = segment.start
        p2 = segment.end
        c = transform_straight_line(p1, p2, k, a, b)

    elif isinstance(segment, Arc):
        # Ellipse arc: Although this is parametrized in svg in a different way, the arc class allows
        # to extract the required attributes

        p = segment.center
        r1 = np.real(segment.radius)
        r2 = np.imag(segment.radius)
        phi = segment.phi
        theta1 = np.deg2rad(segment.theta)
        theta2 = np.deg2rad(segment.delta) + theta1

        c = transform_arc(p, r1, r2, phi, theta1, theta2, k, a, b)
    elif isinstance(segment, QuadraticBezier):
        # Bézier curve of degree 2
        p = [
            segment.start,
            segment.control,
            segment.end
        ]

        c = transform_bezier(p, k, a, b)

    elif isinstance(segment, CubicBezier):
        # Bézier curve of degree 3
        p = [
            segment.start,
            segment.control1,
            segment.control2,
            segment.end
        ]

        c = transform_bezier(p, k, a, b)
    else:
        raise TypeError(f"Unsupported svg path segment class '{type(segment)}'")

    return c
