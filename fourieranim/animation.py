import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


def _extract_center(c,k):
    """

    Parameters
    ----------
    c: array_like
        Fourier coefficients, complex type
    k: array_like
        Corresponding indices, int values

    Returns
    -------
    c_new: ndarray
        Fourier coefficients without center (k = 0)
    k_new: ndarray
        Corresponding indices without k = 0
    p: complex
        Center point. Either c[k==0] or 0, if k==0 was not provided
    """
    c = np.asarray(c)
    k = np.asarray(k)

    p = np.sum(c[k == 0])    # Handle multiple coefficients for k = 0 # TODO: is this a good idea?

    # Remove Fourier coefficient c_0, if given
    c = c[k != 0]
    k = k[k != 0]

    return c, k, p


def _compute_bounding_box(c, k, p=0, zoom_factor=1, method='tight'):
    """
    Computes the bounding box of a Fourier animation.

    This method simply adds the radius of each circle around.
    Parameters
    ----------
    c: array_like
        Fourier coefficients, complex type
    k: array_like
        Corresponding indices, int values
    p: complex
        Center point. 0 by default
    zoom_factor: float
        A factor that defines how much the bounding_box is zoomed in / out.
        Value 1 corresponds to the normal view. Factor 2 means, bounding box has only half
        of the originally computed size (in each direction)
    method: str
        Defines the method how to compute the bounding box. Options are:
        "radius" - sums up the circles radii to determine the maximal length of the bounding box
        "tight" - simulates the circles movement to compute the exact bounding box to always show all circles completely

    Returns
    -------
    x_min: float
        Minimal x-value
    y_min: float
        Minimal y-value
    x_max: float
        Maximal x-value
    y_max: float
        Maximal y-value
    """
    if method is "radius":
        # The sum of all radii is half of the bounding box length / height
        L = np.sum(np.abs(c)) / zoom_factor

        x0 = np.real(p)
        y0 = np.imag(p)

        x_min = x0 - L
        y_min = y0 - L
        x_max = x0 + L
        y_max = y0 + L

    elif method is "tight":
        # Number of evaluation steps
        n = 1000
        t = np.arange(0, 2 * np.pi, 2 * np.pi / 1000)

        # Computes the centers of all circles over all evaluation steps
        # q.shape = (evaluation_steps, number of circles + 1)
        fct = _get_cumulative_fourier_function(c,k,p)
        q = np.asarray([fct(2*np.pi*t / n) for t in range(n)])

        # Remove the point on the last circle - we are only interested in the circles centers
        q = q[:,:-1]

        # Radius of the circles
        r = np.reshape(np.abs(c), (1,-1))

        # Compute the bounding box by adding / substracting r to the circle centers
        # and finding maximal / minimal values
        x_min = np.min(np.real(q) - r)
        x_max = np.max(np.real(q) + r)
        y_min = np.min(np.imag(q) - r)
        y_max = np.max(np.imag(q) + r)
    else:
        raise ValueError(f"'{method}' is not a valid method. Currently only 'radius' and 'tight' are supported. ")

    return x_min, y_min, x_max, y_max


def _get_fourier_function(c, k, p = 0):
    """
    Return a function, that computes for a a number of `t` inbetween `0` and `2*pi` the fourier function.

    Parameters
    ----------
    c: array_like
        Fourier coefficients, complex type
    k: array_like
        Corresponding indices, int values
    p: complex
        Center point. 0 by default

    Returns
    -------
    callable
        A function, that takes an array `t`and return a corresponding array `a` for which the Fourier function was computed
        for each value of `t`
    """
    def fourier_function(t):
        """
        A function, that takes a parameter `t`and return an array `a`, where `a[i]` contains the fourier approximation,
        if only the first `i` coefficients of `c` are used

        Parameters
        ----------
        t: array_like
            The Fourier parameters between 0 and 2 pi

        Returns
        -------
        a: array_like
            The Fourier function evaluated at `t`
        """
        a = c * np.exp(1j * np.reshape(k, (1,-1)) * np.reshape(t, (-1,1)))
        a = np.sum(a, axis = 1) + p
        a = np.reshape(a, np.shape(t))

        return a

    return fourier_function


def _get_cumulative_fourier_function(c, k, p = 0):
    """
    Return a function, that computes for a given `t` inbetween `0` and `2*pi` the fourier approximations
    if only a part of the fourier coefficients is used.

    Parameters
    ----------
    c: array_like
        Fourier coefficients, complex type
    k: array_like
        Corresponding indices, int values
    p: complex
        Center point. 0 by default

    Returns
    -------
    callable
        A function, that takes a parameter `t`and return an array `a`, where `a[i]` contains the fourier approximation,
        if only the first `i` coefficients of `c` are used
    """
    def cumulative_fourier_function(t):
        """
        A function, that takes a parameter `t`and return an array `a`, where `a[i]` contains the fourier approximation,
        if only the first `i` coefficients of `c` are used

        Parameters
        ----------
        t: float
            The fourier parameter between 0 and 2 pi

        Returns
        -------
        a: array_like
            array of length ``len(c)+1``. a[i]` contains the fourier approximation,
            if only the first `i` coefficients of `c` are used
        """
        a = c * np.exp(1j * k * t)
        a = np.concatenate([[p], a])
        a = np.cumsum(a)

        return a

    return cumulative_fourier_function


def animate_fourier(c, k,
                    frames=100, interval=200,
                    figsize=(12, 12),
                    bounding_box='tight', zoom_factor=1,
                    circle_style={}, curve_style={}):
    """
    Creates an animation of a fourier series.

    Parameters
    ----------
    c: array_like
        Fourier coefficients, complex type
    k: array_like
        Corresponding indices, int values
    frames : int, optional
        Number of frames
    interval : number, optional
        Delay between frames in milliseconds. Defaults to 200.
    figsize: tuple
        Size of the figure in inches
    bounding_box: str or array_like
        Either a concrete bounding box as (x_min, y_min, x_max, y_max) tuple or a
        string that specifies the bounding_box algorithm. Possible values are
        `radius` and `tight`.
    zoom_factor: float
        A factor that defines the zoom into the figure. Ignored, if `bounding_box`
        specifies an explicit bounding box.
    curve_style: dict
        A list of styling parameters for the curve. See also `matplotlib.pyplot.plot`.
    circle_style: dict
        A list of styling parameters for the circles. See also `matplotlib.patches.Circle`.

    Returns
    -------
    anim: FuncAnimation
        Animation of the Fourier series
    """

    #
    # 1. Preprocessing
    #

    # Separate constant term (c_0)
    c, k, p = _extract_center(c, k)

    # Circle radius
    r = np.abs(c)

    #
    # 2. Prepare Figure
    #

    # Create Figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.axis('off')

    # Set limits
    if isinstance(bounding_box, str):
        x_min, y_min, x_max, y_max = _compute_bounding_box(c, k, p, zoom_factor=zoom_factor, method=bounding_box)
    else:
        x_min, y_min, x_max, y_max = bounding_box
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_aspect('equal')

    #
    # 3. Create Elements
    #

    # Styles
    ci_s = {
        'fill': False,
        'color': [0.5, 0.5, 0.5],
        'lw': 1.5
    }
    ci_s.update(circle_style)

    cu_s = {
        'lw': 3
    }
    cu_s.update(curve_style)

    # Circles
    circles = [Circle((0, 0), radius=r[i], **ci_s) for i in range(len(c))]
    for circle in circles:
        ax.add_artist(circle)

    # dot = Circle((np.real(pos_dot), np.imag(pos_dot)), radius=L / 100)
    # ax.add_artist(dot)

    line, = ax.plot([], [], **cu_s)

    #
    # 4. Fourier Computation
    #

    fourier_fct = _get_fourier_function(c, k, p)
    cum_fourier_fct = _get_cumulative_fourier_function(c, k, p)

    #
    # 5. Animate
    #

    def animate(i):
        t = i * 2 * np.pi / frames

        # Compute Circle position
        pos = cum_fourier_fct(t)
        pos = pos[:-1]

        for j in range(len(pos)):
            circles[j].set_center((np.real(pos[j]), np.imag(pos[j])))

        t = np.arange(i+1) * 2 * np.pi / frames
        points = fourier_fct(t)

        x, y = np.real(points), np.imag(points)
        line.set_data(x, y)

        return circles + [line]

    anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)
    return anim
