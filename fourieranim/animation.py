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


def _compute_bounding_box(c, k, p=0, zoom_factor=1):
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
    # TODO: One could look at the circles to find the effective bounding box!
    L = np.sum(np.abs(c)) / zoom_factor

    x0 = np.real(p)
    y0 = np.imag(p)

    x_min = x0 - L
    y_min = y0 - L
    x_max = x0 + L
    y_max = y0 + L

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


def animate_fourier(c, k, frames = 100, interval = 200, figsize=(12,12), zoom_factor=1):
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
    figsize
        Size of the figure in inches
    zoom_factor: float
        A factor that defines the soom into the figure

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
    x_min, y_min, x_max, y_max = _compute_bounding_box(c, k, p, zoom_factor=zoom_factor)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    #
    # 3. Create Elements
    #

    # Circles
    circles = [Circle((0, 0), radius=r[i], fill=False) for i in range(len(c))]
    for circle in circles:
        ax.add_artist(circle)

    # dot = Circle((np.real(pos_dot), np.imag(pos_dot)), radius=L / 100)
    # ax.add_artist(dot)

    line, = ax.plot([], [], lw=3)


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
