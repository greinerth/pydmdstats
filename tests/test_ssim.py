from __future__ import annotations

import numpy as np
from varprodmdstatspy.util.experiment_utils import ssim_multi_images
from varprodmdstatspy.visualize_complex2d import generate_complex2d


def test_ssim() -> None:
    """Test structural similarity index for experiments."""
    complex_signal = generate_complex2d()[-1]
    real_imgs = np.expand_dims(complex_signal.real, axis=-1)
    mean_real, var_real = ssim_multi_images(real_imgs, real_imgs)
    assert mean_real == 1
    assert var_real == 0
