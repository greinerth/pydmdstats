from __future__ import annotations

import numpy as np
from varprodmdstatspy.util.experiment_utils import ssim_multi_images

generator = np.random.Generator(np.random.PCG64())


def generate_complex2d(
    std: float = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate damped oscillating signal

    :param std: Standard deviation for data corruption, defaults to -1
    :type std: float, optional
    :return: snapshots, timestamps, data
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    timestamps = np.linspace(0, 6, 16)
    x_1 = np.linspace(-3, 3, 128)
    x_2 = np.linspace(-3, 3, 128)
    x1grid, x2grid = np.meshgrid(x_1, x_2)

    data = [
        np.expand_dims(2 / np.cosh(x1grid) / np.cosh(x2grid) * (1.2j**-t), axis=0)
        for t in timestamps
    ]
    snapshots_flat = np.zeros((np.prod(data[0].shape), len(data)), dtype=complex)
    for j, img in enumerate(data):
        __img = img.copy()
        if std > 0:
            __img += generator.normal(0, std, img.shape)
            data[j] = __img
        snapshots_flat[:, j] = np.ravel(__img)
    return snapshots_flat, timestamps, np.concatenate(data, axis=0)


def test_ssim() -> None:
    """Test structural similarity index for experiments."""
    complex_signal = generate_complex2d()[-1]
    real_imgs = np.expand_dims(complex_signal.real, axis=-1)
    mean_real, var_real = ssim_multi_images(real_imgs, real_imgs)
    assert mean_real == 1
    assert var_real == 0
