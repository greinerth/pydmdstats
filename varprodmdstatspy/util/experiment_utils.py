""" utility functions for OptDMD experiments
"""
from typing import Tuple, Dict, Any

import numpy as np
from pydmd.bopdmd import BOPDMD
from pydmd.varprodmd import VarProDMD
from skimage.metrics import structural_similarity as ssim

import varprodmdstatspy.util.stats as stats

OPT_ARGS: Dict[str, Any] = {"method": 'trf', "tr_solver": 'exact', "x_scale": 'jac'}
# OPT_ARGS = {"method": 'lm'}


def signal2d(x_in: np.ndarray,  # pylint: disable=unused-variable
             y_in: np.ndarray,
             x_0: np.ndarray,
             y_0: np.ndarray,
             velocity: float,
             time: float,
             sigma: float = 0.1) -> np.ndarray:
    """Create 2D signal (2 moving points with different velocities)

    Args:
        x_in (np.ndarray): Corrdinates
        x_0 (np.ndarray): x-center coordinate
        y_0 (np.ndarray): y-center coordinate
        velocity (float): Speed in pixel/s
        time (float): Time steps
        sigma (float, optional): Size of dots. Defaults to 0.1.

    Returns:
        np.ndarray: 2D signal
    """
    x_diff_square_fast = np.square(x_in - 10 * velocity * time - x_0[0])
    y_diff_square_fast = np.square(y_in - y_0[0])
    x_diff_square_slow = np.square(x_in - x_0[1])
    y_diff_square_slow = np.square(y_in - velocity * time - y_0[1])
    return np.exp(-sigma * (x_diff_square_fast + y_diff_square_fast)) + \
        np.exp(-sigma*(x_diff_square_slow + y_diff_square_slow))


def signal(x_loc: np.ndarray,  # pylint: disable=unused-variable
           time: np.ndarray) -> np.ndarray:
    """Create high dimensional (complex) signal

    Args:
        x_loc (np.ndarray): location spread along axis
        time (np.ndarray): Time value

    Returns:
        np.ndarray: Complex surface
    """
    __f_1 = 1. / np.cosh(x_loc + 3) * np.exp(1j*2.3*time)
    __f_2 = 2. / np.cosh(x_loc) * np.tanh(x_loc) * np.exp(1j*2.8*time)
    return __f_1 + __f_2


def ssim_multi_images(imgs_baseline: np.ndarray,  # pylint: disable=unused-variable
                      imgs_reconst: np.ndarray) -> Tuple[float, float]:
    """Compute SSIM on a collection of images

    Args:
        imgs_baseline (np.ndarray): Base line images of shape [N, H, W, C]
        imgs_reconst (np.ndarray): Reconstructed images of shape [N, H, W, C]

    Raises:
        ValueError: If shape missmatches

    Returns:
        float: Mean and Variance of SSIM
    """
    mean: float = 0
    prev_mean: float = 0
    var_hat: float = 0
    i: int = 0

    if len(imgs_baseline.shape) != len(imgs_baseline.shape):
        raise ValueError("Invalid dimensions of images")

    for __i, __j in zip(imgs_baseline.shape, imgs_reconst.shape):
        if __i != __j:
            raise ValueError("Invalid dimensions of images")

    channels = imgs_baseline.shape[-1]
    weight = 1. / float(channels)

    for i in range(imgs_baseline.shape[0]):
        __res: float = 0
        for __c in range(channels):
            __res += weight * ssim(imgs_baseline[i, :, :, __c],
                                   imgs_reconst[i, :, :, __c],
                                   data_range=imgs_reconst[i, :, :, __c].max()
                                   - imgs_reconst[i, :, :, __c].min(),
                                   sigma=1.5,
                                   use_sample_covariance=False,
                                   gaussian_weights=True)
        delta = (__res - mean)
        mean = prev_mean + delta / float(i + 1)
        var_hat = delta * (__res - prev_mean)
        prev_mean = mean

    var = var_hat / float(i) if i > 0 else 0

    return mean, var


def std_checker(x_in: any) -> float:  # pylint: disable=unused-variable
    """Check standard deviatons as input

    Args:
        x_in (any): Std encoded es string

    Returns:
        float: Absolute value of std.
    """
    __x = float(x_in)
    return abs(__x)


def comp_checker(x_in: any) -> float:  # pylint: disable=unused-variable
    """Argparser checking instance for compression values

    Args:
        x_in (any): Compression value encoded as string

    Raises:
        ValueError: If absolute value of compression is not within (0, 1)

    Returns:
        float: Absolute value of compression
    """
    __comp = abs(float(x_in))
    if not 0 <= __comp < 1:
        raise ValueError(
            "Invalid Compression. Values needs to be between [0, 1)!")
    return __comp


def bopdmd_wrapper(data, time):  # pylint: disable=unused-variable
    """ Dummy function to force optimization execution of BOPDMD class

    Args:
        data (np.ndarray): Measurements for OptDMD
        time (np.ndarray): Timestamps of measurements
    """
    __bop_dmd = BOPDMD()
    __bop_dmd.fit(data, time)


def varprodmd_wrapper(data: np.ndarray,
                   time: np.ndarray,
                   eps: float = 1e-2):  # pylint: disable=unused-variable
    """ Dummy function to force optimization execution of VarProDMD class

    Args:
        data (np.ndarray): Measurements for VarProDMD
        time (np.ndarray): Timestamps of measurements
        eps (float, optional): Compression. Defaults to 1e-2.

    """
    __varpro_dmd = VarProDMD(compression=eps, optargs=OPT_ARGS)
    __varpro_dmd.fit(data, time)


def exec_times_varpro_dmd(data: np.ndarray,  # pylint: disable=unused-variable
                       time: np.ndarray,
                       comp: float = 0.1,
                       optargs: Dict[str, Any] = OPT_ARGS,
                       n_iter: int = 100) -> stats.ExecutionStats:
    """Wrapper functions for timing purposes

    Args:
        data (np.ndarray): Input for OptDMD
        time (np.ndarray): time stamps, where mesaurements where taken
        comp (float, optional): Compression. Defaults to 0.1.
        n_iter (int, optional): Number of iterations to execute. Defaults to 100.

    Returns:
        stats.ExecutionStats: _description_
    """

    wrapper = stats.runtime_stats(False)(varprodmd_wrapper)
    for __ in range(n_iter):
        wrapper(data, time, comp)
    return wrapper


def exec_times_bop_dmd(data: np.ndarray,  # pylint: disable=unused-variable
                       time: np.ndarray,
                       n_iter: int = 100) -> stats.ExecutionStats:
    """Wrapper function for timing purposes

    Args:
        data (np.ndarray): Input data for DOPDMD
        time (np.ndarray): time stamps, where mesaurements where taken
        n_iter (int, optional): Number of iterations to execute. Defaults to 100.

    Returns:
        stats.ExecutionStats: Return the execution statistics
    """
    __wrapper = stats.runtime_stats(False)(bopdmd_wrapper)
    for __ in range(n_iter):
        __wrapper(data, time)
    return __wrapper
