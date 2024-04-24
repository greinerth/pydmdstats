from __future__ import annotations

import numpy as np
from pydmd import VarProDMD
from pydmd.utils import compute_svd

from varprodmdstatspy.util.experiment_utils import signal


def test_ortho() -> None:
    """Test if selected samples in subspace are also almost orthogornal in original space."""
    COMP = 0.6
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    x, _time = np.meshgrid(x_loc, time)
    z_signal = signal(x, _time).T

    dmd = VarProDMD(compression=COMP, exact=False)
    dmd.fit(z_signal, time)

    s_r, v_r = compute_svd(z_signal)[1:]
    data_subspace = v_r.conj().T * s_r[:, None]

    selected_samples = data_subspace[:, dmd.selected_samples]
    subspace_diag = np.diagonal(selected_samples.conj().T @ selected_samples)
    selected_samples = z_signal[:, dmd.selected_samples]
    origspace_diag = np.diagonal(selected_samples.conj().T @ selected_samples)
    np.testing.assert_allclose(subspace_diag, origspace_diag)
