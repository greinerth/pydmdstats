"""utility functions for OptDMD experiments"""

import timeit
from typing import Any, Dict, Tuple

import numpy as np
from pydmd.bopdmd import BOPDMD
from pydmd.varprodmd import VarProDMD
from skimage.metrics import structural_similarity as ssim

import varprodmdstatspy.util.stats as stats

OPT_ARGS: Dict[str, Any] = {  # pylint: disable=unused-variable
    "method": "trf",
    "tr_solver": "exact",
    # "x_scale": 'jac',
}


def signal2d(
    x_in: np.ndarray,  # pylint: disable=unused-variable
    y_in: np.ndarray,
    x_0: np.ndarray,
    y_0: np.ndarray,
    velocity: float,
    time: float,
    sigma: float = 0.1,
) -> np.ndarray:
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
    return np.exp(-sigma * (x_diff_square_fast + y_diff_square_fast)) + np.exp(
        -sigma * (x_diff_square_slow + y_diff_square_slow)
    )


def signal(
    x_loc: np.ndarray,  # pylint: disable=unused-variable
    time: np.ndarray,
) -> np.ndarray:
    """Create high dimensional (complex) signal

    Args:
        x_loc (np.ndarray): location spread along axis
        time (np.ndarray): Time value

    Returns:
        np.ndarray: Complex surface
    """
    __f_1 = 1.0 / np.cosh(x_loc + 3) * np.exp(1j * 2.3 * time)
    __f_2 = 2.0 / np.cosh(x_loc) * np.tanh(x_loc) * np.exp(1j * 2.8 * time)
    return __f_1 + __f_2


def ssim_multi_images(
    imgs_baseline: np.ndarray,  # pylint: disable=unused-variable
    imgs_reconst: np.ndarray,
) -> Tuple[float, float]:
    """Compute SSIM on a collection of images

    Args:
        imgs_baseline (np.ndarray): Base line images of shape [N, H, W, C]
        imgs_reconst (np.ndarray): Reconstructed images of shape [N, H, W, C]

    Raises:
        ValueError: If shape missmatches

    Returns:
        float: Mean and Variance of SSIM
    """
    if len(imgs_baseline.shape) != len(imgs_baseline.shape):
        raise ValueError("Invalid dimensions of images")

    for i, j in zip(imgs_baseline.shape, imgs_reconst.shape):
        if i != j:
            raise ValueError("Invalid dimensions of images")

    channels = imgs_baseline.shape[-1]
    weight = 1.0 / float(channels)
    ssim_stats = stats.Stats()

    for i in range(imgs_baseline.shape[0]):
        res: float = 0
        for channel in range(channels):
            res += weight * ssim(
                imgs_baseline[i, :, :, channel],
                imgs_reconst[i, :, :, channel],
                data_range=imgs_reconst[i, :, :, channel].max()
                - imgs_reconst[i, :, :, channel].min(),
                sigma=1.5,
                use_sample_covariance=False,
                gaussian_weights=True,
            )
        ssim_stats.push(res)

    return ssim_stats.mean, ssim_stats.var


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
        raise ValueError("Invalid Compression. Values needs to be between [0, 1)!")
    return __comp


def bopdmd_wrapper(data, time):  # pylint: disable=unused-variable
    """Dummy function to force optimization execution of BOPDMD class

    Args:
        data (np.ndarray): Measurements for OptDMD
        time (np.ndarray): Timestamps of measurements
    """
    __bop_dmd = BOPDMD()
    __bop_dmd.fit(data, time)


def varprodmd_wrapper(
    data: np.ndarray, time: np.ndarray, optargs: Dict[str, Any], eps: float = 1e-2
):  # pylint: disable=unused-variable
    """Dummy function to force optimization execution of VarProDMD class

    Args:
        data (np.ndarray): Measurements for VarProDMD
        time (np.ndarray): Timestamps of measurements
        eps (float, optional): Compression. Defaults to 1e-2.

    """
    __varpro_dmd = VarProDMD(compression=eps, optargs=optargs)
    __varpro_dmd.fit(data, time)


def _corrupt_data(data: np.ndarray, std: float) -> np.ndarray:
    _data = data.copy()

    if std > 0:
        if np.iscomplexobj(data):
            _data.real += np.random.normal(0.0, std, size=data.real.shape)
            _data.imag += np.random.normal(0.0, std, size=data.imag.shape)
        else:
            _data += np.random.normal(0.0, std, size=data.shape)

    return _data


def _flatten(data: np.ndarray) -> np.ndarray:
    flat = np.zeros((np.prod(data.shape[1:]), data.shape[0]), dtype=data.dtype)
    for i in range(data.shape[0]):
        flat[:, i] = np.ravel(data[i])
    return flat


def _flat2images(flat: np.ndarray, shape: np.shape) -> np.ndarray:
    out = []
    is_complex = np.iscomplexobj(flat)

    for i in range(flat.shape[-1]):
        img = flat[:, i].reshape(shape[1:])

        if is_complex:
            img = np.concatenate([img.real, img.imag], axis=-1)
        img = np.expand_dims(img, axis=0)
        out.append(img)

    return np.concatenate(out, axis=0)


def _complex2realimgs(imgs: np.ndarray) -> np.ndarray:
    out = []

    for i in range(imgs.shape[0]):
        real_img = imgs[i].real
        imag_img = imgs[i].imag
        img = np.concatenate([real_img, imag_img], axis=-1)
        out.append(np.expand_dims(img, axis=0))

    return np.concatenate(out, axis=0)


def dmd_stats(
    method: str,
    data: np.ndarray,  # pylint: disable=unused-variable
    time: np.ndarray,
    std: float,
    optargs: dict,
    compression: float,
    n_iter: int = 100,
) -> tuple[float]:
    """Wrapper functions for timing purposes

    Args:
        data (np.ndarray): Input for OptDMD
        time (np.ndarray): time stamps, where mesaurements where taken
        n_iter (int, optional): Number of iterations to execute. Defaults to 100.

    Returns:
        stats.ExecutionStats: Runtime statistics (Mean and variance)
    """

    # wrapper = stats.runtime_stats(False)(varprodmd_wrapper)
    # timestats = stats.Stats()
    # error_stats = stats.Stats()
    is_complex = np.iscomplexobj(data)
    is_img = len(data.shape) > 2
    calc_error = True
    experiment_stats = np.zeros((2, n_iter))

    for i in range(n_iter):
        _data = _corrupt_data(data, std)

        if is_img:
            _data = _flatten(data)

        if method == "BOPDMD":
            dmd = BOPDMD(trial_size=_data.shape[-1] - 1)
        elif method == "VarProDMD":
            dmd = VarProDMD(compression=compression, optargs=optargs)
        else:
            raise ValueError(f"{method} not implemented")

        t0 = timeit.default_timer()
        dmd.fit(_data, time)
        dt = timeit.default_timer() - t0
        # timestats.push(dt)
        experiment_stats[1, i] = dt

        if calc_error:
            pred = dmd.forecast(time)
            error = 0.0

            if is_img:
                pred_in = _flat2images(pred if is_complex else pred.real, data.shape)
                data_in = data if not is_complex else _complex2realimgs(data)
                error = ssim_multi_images(data_in, pred_in)[0]
            else:
                error = np.linalg.norm(
                    data - (pred if is_complex else pred.real)
                ) / np.sqrt(data.shape[-1])
            experiment_stats[0, i] = error
            # error_stats.push(error)
            calc_error = std > 0.0
        # wrapper(data, time, optargs, comp)
    if std == 0:
        experiment_stats[0, 1:] = experiment_stats[0, 0]
    mean = np.mean(experiment_stats, axis=-1)
    zero_mean = experiment_stats - mean[:, None]
    cov = (zero_mean @ zero_mean.T) / float(n_iter)

    return mean[0], mean[1], cov[0, 0], cov[0, 1], cov[1, 1]


def dmd_stats_global_temp(
    method: str,
    data: np.ndarray,  # pylint: disable=unused-variable
    time: np.ndarray,
    msk_valid: np.ndarray,
    std: float,
    optargs: dict,
    compression: float,
    n_iter: int = 100,
) -> tuple[float]:
    """Wrapper functions for timing purposes

    Args:
        data (np.ndarray): Input for OptDMD
        time (np.ndarray): time stamps, where mesaurements where taken
        n_iter (int, optional): Number of iterations to execute. Defaults to 100.

    Returns:
        stats.ExecutionStats: Runtime statistics (Mean and variance)
    """

    # wrapper = stats.runtime_stats(False)(varprodmd_wrapper)
    calc_error = True
    rows, cols = np.where(~msk_valid)
    msk_flat = np.ravel(msk_valid)
    data[:, rows, cols] = 0.0
    experiment_stats = np.zeros((2, n_iter))

    for i in range(n_iter):
        if method == "BOPDMD":
            dmd = BOPDMD(trial_size=data.shape[0] - 1)
        elif method == "VarProDMD":
            dmd = VarProDMD(compression=compression, optargs=optargs)
        else:
            raise ValueError(f"{method} not implemented")

        _data = _corrupt_data(data, std)
        _data = _flatten(_data)[msk_flat, :]

        t0 = timeit.default_timer()
        dmd.fit(_data, time)
        dt = timeit.default_timer() - t0
        experiment_stats[1, i] = dt

        if calc_error:
            pred = dmd.forecast(time)
            flat_in = np.zeros((msk_flat.shape[-1], pred.shape[-1]))
            for j in range(pred.shape[-1]):
                flat_in[msk_flat, j] = pred[:, j].real
            pred_in = _flat2images(flat_in, data.shape)
            error = ssim_multi_images(
                np.expand_dims(data, axis=-1), np.expand_dims(pred_in, axis=-1)
            )[0]

            calc_error = std > 0.0
            experiment_stats[0, i] = error
    if std == 0:
        experiment_stats[0, 1:] = experiment_stats[0, 0]

    mean = np.mean(experiment_stats, axis=-1)
    zero_mean = experiment_stats - mean[:, None]
    cov = (zero_mean @ zero_mean.T) / float(n_iter)
    return mean[0], mean[1], cov[0, 0], cov[0, 1], cov[1, 1]


def exec_times_bop_dmd(
    data: np.ndarray,  # pylint: disable=unused-variable
    time: np.ndarray,
    n_iter: int = 100,
) -> stats.Stats:
    """Wrapper function for timing purposes

    Args:
        data (np.ndarray): Input data for DOPDMD
        time (np.ndarray): time stamps, where mesaurements where taken
        n_iter (int, optional): Number of iterations to execute. Defaults to 100.

    Returns:
        stats.ExecutionStats: Return the execution statistics
    """
    # __wrapper = stats.runtime_stats(False)(bopdmd_wrapper)
    timestats = stats.Stats()
    for _ in range(n_iter):
        # __wrapper(data, time)
        dmd = BOPDMD()
        t0 = timeit.default_timer()
        dmd.fit(data, time)
        dt = timeit.default_timer() - t0
        timestats.push(dt)

    return timestats
