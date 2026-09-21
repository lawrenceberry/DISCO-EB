"""The recombination history against CLASS, and the warmed time of its solve.

The RECFAST system is integrated on the GPU by modax (see
``discoeb.thermodynamics_recfast``). The check runs the same cosmologies as
``test_matter_power_spectrum_matches_class``: one, and a batch of 128 nearby
flat LCDM models, and compares the free-electron fraction, the matter
temperature and the baryon sound speed with CLASS run with its own recfast
implementation and no reionization.

The gates are set from the measured agreement, with margin. The electron
fraction differs from CLASS by up to ~2 percent below ``z ~ 1400`` (differences
in the fudge factors of the two recfast versions), the temperature by less than
1 percent, and ``c_b^2`` by less than 1 percent, except in the redshift window
where RECFAST switches off its tight Compton-coupling approximation for ``T_m``
(``timeTh < H_frac * timeH``): there ``dT_m/dz`` jumps momentarily, which CLASS's
full treatment does not reproduce, and ``c_b^2`` differs by up to ~10 percent
over a few grid points around ``z ~ 850``.

The benchmark times one warmed call of :func:`thermal_history_batch`. The
modax Rodas5P solve measured 156 ms (N=1) and 188 ms (N=128) on an RTX 4070
SUPER, against 449 ms and 1034 ms for the diffrax Dopri5 solve it replaced.
"""

import numpy as np
import pytest

from test_perturbations import BATCH_SIZE_CASES, _perturbed_cosmologies

Z_MIN, Z_MAX = 10.0, 5000.0
XE_GATE = 0.03
TM_GATE = 0.01
CS2_GATE = 0.01
SWITCH_WINDOW = (800.0, 900.0)
CS2_SWITCH_GATE = 0.2


def _class_thermodynamics(param):
    """Return ``(z, x_e, T_b, c_b^2)`` from CLASS with recfast and no reionization, ascending in z."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    cosmo = Class()
    cosmo.set(
        {
            "h": h,
            "omega_b": param["Omegab"] * h**2,
            "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
            "T_cmb": param["Tcmb"],
            "YHe": param["YHe"],
            "N_ur": param["Neff"],
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "recombination": "recfast",
            "reio_parametrization": "reio_none",
        }
    )
    try:
        cosmo.compute()
        thermo = cosmo.get_thermodynamics()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    order = np.argsort(thermo["z"])
    return tuple(np.asarray(thermo[key])[order] for key in ("z", "x_e", "Tb [K]", "c_b^2"))


@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_recfast_matches_class(n_cosmologies, benchmark):
    """Check the recombination history against CLASS and time one warmed batch solve."""

    from discoeb.perturbations import thermal_history_batch

    cosmologies = _perturbed_cosmologies(n_cosmologies)
    history = benchmark.pedantic(
        thermal_history_batch,
        args=(cosmologies,),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )
    z = 1.0 / history["a"] - 1.0
    inside = (z >= Z_MIN) & (z <= Z_MAX)
    switch = (z > SWITCH_WINDOW[0]) & (z < SWITCH_WINDOW[1])

    def worst(ours, theirs, mask):
        rel = np.abs(ours[mask] / theirs[mask] - 1.0)
        i = np.argmax(rel)
        return float(rel[i]), float(z[mask][i])

    for i, param in enumerate(cosmologies):
        z_class, xe_class, tb_class, cb2_class = _class_thermodynamics(param)
        xe_ref = np.interp(z, z_class, xe_class)
        tm_ref = np.interp(z, z_class, tb_class)
        cs2_ref = np.interp(z, z_class, cb2_class)
        assert np.isfinite(history["xe"][i]).all()

        err, at = worst(history["xe"][i], xe_ref, inside)
        assert err < XE_GATE, f"cosmology {i}: x_e off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["Tm"][i], tm_ref, inside)
        assert err < TM_GATE, f"cosmology {i}: T_m off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["cs2"][i], cs2_ref, inside & ~switch)
        assert err < CS2_GATE, f"cosmology {i}: c_b^2 off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["cs2"][i], cs2_ref, inside & switch)
        assert err < CS2_SWITCH_GATE, (
            f"cosmology {i}: c_b^2 off by {err:.3g} at z={at:.0f} in the tight-coupling switch window"
        )
