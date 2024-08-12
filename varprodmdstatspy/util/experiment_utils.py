"""utility functions for OptDMD experiments"""
from __future__ import annotations

import timeit
from typing import Any

import numpy as np
from pydmd.bopdmd import BOPDMD
from pydmd.varprodmd import VarProDMD
from skimage.metrics import structural_similarity as ssim

from varprodmdstatspy.util import stats

OPT_ARGS: dict[str, Any] = {  # pylint: disable=unused-variable
    "method": "trf",
    "tr_solver": "exact",
    # "x_scale": 'jac',
}


def signal2d(
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_0: np.ndarray,
    y_0: np.ndarray,
    velocity: float,
    time: float,
    sigma: float = 0.1,
) -> np.ndarray:
    """Create 2D signal (2 moving points with different velocities)

    :param x_in: x-coordinates
    :type x_in: np.ndarray
    :param y_in: y-coordinates
    :type y_in: np.ndarray
    :param x_0: x-center coordinate
    :type x_0: np.ndarray
    :param y_0: y-center coordinate
    :type y_0: np.ndarray
    :param velocity: Speed in pixel/s
    :type velocity: float
    :param time: Time steps
    :type time: float
    :param sigma: Size of dots, defaults to 0.1
    :type sigma: float, optional
    :return: 2D signal
    :rtype: np.ndarray
    """
    x_diff_square_fast = np.square(x_in - 10 * velocity * time - x_0[0])
    y_diff_square_fast = np.square(y_in - y_0[0])
    x_diff_square_slow = np.square(x_in - x_0[1])
    y_diff_square_slow = np.square(y_in - velocity * time - y_0[1])
    return np.exp(-sigma * (x_diff_square_fast + y_diff_square_fast)) + np.exp(
        -sigma * (x_diff_square_slow + y_diff_square_slow)
    )


def signal(
    x_loc: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Create high dimensional (complex) signal

    :param x_loc: Location spread along axis
    :type x_loc: np.ndarray
    :param time: Time value
    :type time: np.ndarray
    :return: Complex surface
    :rtype: np.ndarray
    """
    f1 = 1.0 / np.cosh(x_loc + 3) * np.exp(1j * 2.3 * time)
    f2 = 2.0 / np.cosh(x_loc) * np.tanh(x_loc) * np.exp(1j * 2.8 * time)
    return f1 + f2


def ssim_multi_images(
    imgs_baseline: np.ndarray,  # pylint: disable=unused-variable
    imgs_reconst: np.ndarray,
) -> tuple[float, float]:
    """Compute SSIM on a collection of images

    :param imgs_baseline: Base line images of shape [N, H, W, C]
    :type imgs_baseline: np.ndarray
    :param imgs_reconstr: Reconstructed images of shape [N, H, W, C]
    :raises ValueError: If shape of `imgs_baseline` and `imgs_reconst` disagree
    :raises ValueError: If dimensions of images are invalid
    :return: _description_
    :rtype: tuple[float, float]
    """
    if len(imgs_baseline.shape) != len(imgs_baseline.shape):
        msg = "Invalid shape of images!"
        raise ValueError(msg)

    for i, j in zip(imgs_baseline.shape, imgs_reconst.shape):
        if i != j:
            msg = "Invalid dimensions of images!"
            raise ValueError(msg)

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


def std_checker(x_in: Any) -> float:  # pylint: disable=unused-variable
    """Check standard deviatons as input

    :param x_in: Std encoded es string
    :type x_in: Any
    :return: Absolute value of std.
    :rtype: float
    """
    out = float(x_in)
    return abs(out)


def comp_checker(x_in: Any) -> float:  # pylint: disable=unused-variable
    """Argparser checking instance for compression values

    :param x_in: Compression value
    :type x_in: Any
    :raises ValueError: If absolute value of compression is not within (0, 1)
    :return: Absolute value of compression
    :rtype: float
    """
    comp = abs(float(x_in))
    if not 0 <= comp < 1:
        msg = "Invalid Compression. Values needs to be between [0, 1)!"
        raise ValueError(msg)
    return comp


def _corrupt_data(data: np.ndarray, std: float) -> np.ndarray:
    _data = data.copy()
    generator = np.random.Generator(np.random.PCG64())
    if std > 0:
        if np.iscomplexobj(data):
            _data.real += generator.normal(0.0, std, size=data.real.shape)
            _data.imag += generator.normal(0.0, std, size=data.imag.shape)
        else:
            _data += generator.normal(0.0, std, size=data.shape)

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
    optargs: dict[str, Any],
    compression: float,
    n_iter: int = 100,
) -> tuple[Any, Any, Any, Any, Any]:
    """DMD runtime- and reconstruction statistics

    :param method: Which DMD method to use ("VarProDMD" or "BOPDMD")
    :type method: str
    :param data: Input data
    :type data: np.ndarray
    :param std: Standard deviation for artificial noise corruption of data
    :type std: float
    :param optargs: Additional optimization arguments for VarProDMD
    :type optargs: dict[str, Any]
    :param compression: Compression for VarProDMD. Must lie in [0, 1)
    :type compression: float
    :param n_iter: Number of iterations, defaults to 100
    :type n_iter: int, optional
    :raises ValueError: If `method` is not "VarProDMD" or "BOPDMD"
    :return: Runtime statistics
        (Mean Error, Mean execution time, Error variance, covariance and time variance)
    :rtype: tuple[Any, Any, Any, Any, Any]
    """
    is_complex = np.iscomplexobj(data)
    is_img = len(data.shape) > 2
    calc_error = True
    experiment_stats = np.zeros((2, n_iter))

    for i in range(n_iter):
        _data = _corrupt_data(data, std)

        if is_img:
            _data = _flatten(data)

        if method == "BOPDMD":
            dmd = BOPDMD(trial_size=_data.shape[-1])
        elif method == "VarProDMD":
            dmd = VarProDMD(compression=compression, optargs=optargs)
        else:
            msg = f"{method} not implemented"
            raise ValueError(msg)

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
                    data - (pred if is_complex else pred.real), axis=0
                ).mean()
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
    optargs: dict[str, Any],
    compression: float,
    n_iter: int = 100,
) -> tuple[Any, Any, Any, Any, Any]:
    """Global temperature runtime statistics and reconstruction performance

    :param method: Method to use ("VarProDMD" or "BOPDMD")
    :type method: str
    :param data: Input for VarProDMD or BOPDMD instance
    :type data: np.ndarray
    :param msk_valid: Mask of which datapoints are valid
    :type msk_valid: np.ndarray
    :param std: Standard deviation for artificial data corruption
    :type std: float
    :param optargs: Additional optimization arguments for VarProDMD
    :type optargs: dict[str, Any]
    :param compression: Compression for VarProDMD. Must lie in [0, 1)
    :type compression: float
    :param n_iter: Number of iterations, defaults to 100
    :type n_iter: int, optional
    :raises ValueError: If `method` is not "VarProDMD" or "BOPDMD"
    :return: Runtime statistics
        (Mean Error, Mean execution time, Error variance, covariance and time variance)
    :rtype: tuple[Any, Any, Any, Any, Any]
    """
    # wrapper = stats.runtime_stats(False)(varprodmd_wrapper)
    calc_error = True
    rows, cols = np.where(~msk_valid)
    msk_flat = np.ravel(msk_valid)
    data[:, rows, cols] = 0.0
    experiment_stats = np.zeros((2, n_iter))

    for i in range(n_iter):
        if method == "BOPDMD":
            dmd = BOPDMD(trial_size=data.shape[0])
        elif method == "VarProDMD":
            dmd = VarProDMD(compression=compression, optargs=optargs)
        else:
            msg = f"{method} not implemented"
            raise ValueError(msg)

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
