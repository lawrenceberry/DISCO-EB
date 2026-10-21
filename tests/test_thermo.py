"""The recombination history against CLASS, and the warmed time of its solve.

The RECFAST system is integrated on the GPU by modax (see
``discoeb.thermodynamics_recfast``). The check runs the same cosmologies as
``test_matter_power_spectrum_matches_class``: one, and a batch of 128 nearby
flat LCDM models, and compares the free-electron fraction, the matter
temperature and the baryon sound speed with CLASS run with its own recfast
implementation and no reionization.

The gates are set from the measured agreement, with margin. The electron
fraction differs from CLASS by up to ~2 percent below ``z ~ 1400`` (differences
in the fudge factors of the two recfast versions; hydrogen leaves Saha
equilibrium on redshift as in CLASS 3, since RECFAST's switch on ``x_H`` made
the stiff solver overshoot ``x_e`` by up to 5 percent for some cosmologies of
the batch), the temperature by less than
1 percent, and ``c_b^2`` by less than 1 percent, except in the redshift window
where RECFAST switches off its tight Compton-coupling approximation for ``T_m``
(``timeTh < H_frac * timeH``): there ``dT_m/dz`` jumps momentarily, which CLASS's
full treatment does not reproduce, and ``c_b^2`` differs by up to ~10 percent
over a few grid points around ``z ~ 850``.

The benchmark times one warmed call of :func:`thermal_history_batch`. The
modax Rodas5P solve measured 156 ms (N=1) and 188 ms (N=128) on an RTX 4070
SUPER, against 449 ms and 1034 ms for the diffrax Dopri5 solve it replaced.

``test_thermo_tables_successive_batches`` times what the perturbation solve
actually consumes, the device tables of :func:`build_thermo_tables_batch`
(the RECFAST solve plus the background quadrature and the resampling onto the
kernel's ``log(tau)`` grid, one vmapped program), for fresh batches of 128
issued one after another: four must cost four single fresh batches, with no
recompilation, and a fresh batch must stay within a small factor of the bare
RECFAST solve.
"""

import numpy as np
import pytest

from test_perturbations import (
    BATCH_SIZE_CASES,
    _perturbed_cosmologies,
    _successive_batches,
    _wall_time,
)

# Fresh 128-cosmology table builds issued in sequence, how far past four
# single builds the four may run, the rounds the bare RECFAST solve's time
# is the minimum over, and how far above that solve one fresh table build may
# sit (measured at 1.6x: the background and the resampling on top of RECFAST).
SUCCESSIVE_BATCHES = 4
SUCCESSIVE_BATCH_SIZE = 128
SUCCESSIVE_TOLERANCE = 0.25
TIMING_ROUNDS = 3
TABLES_OVER_RECFAST = 3.0

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


def test_thermo_tables_successive_batches(benchmark):
    """Four fresh batches of device tables cost four single ones, close to the bare RECFAST solve.

    The thermodynamics half of an inference iteration. A first batch warms up
    (compiles the vmapped background-and-resampling program); the bare RECFAST
    solve of :func:`thermal_history_batch` is then timed warm as the floor; one
    fresh batch of :func:`build_thermo_tables_batch` is the single reference;
    and four more fresh batches are built back to back. They must take no
    longer than four references within :data:`SUCCESSIVE_TOLERANCE` -- a
    recompile or a per-batch host loop would show -- and the reference must be
    within :data:`TABLES_OVER_RECFAST` of the RECFAST solve. The benchmark's
    round is the four batches again, now from the cache.
    """

    from discoeb.perturbations import build_thermo_tables_batch, thermal_history_batch

    warm, reference, *batches = _successive_batches(
        2 + SUCCESSIVE_BATCHES, SUCCESSIVE_BATCH_SIZE
    )
    build_thermo_tables_batch(warm)
    thermal_history_batch(warm)
    recfast = min(_wall_time(lambda: thermal_history_batch(warm))[0] for _ in range(TIMING_ROUNDS))

    single, tables = _wall_time(lambda: build_thermo_tables_batch(reference))
    assert len(tables) == SUCCESSIVE_BATCH_SIZE
    assert all(np.isfinite(values).all() for _, values, _, _ in tables)

    def successive():
        return [build_thermo_tables_batch(cosmologies) for cosmologies in batches]

    four, _ = _wall_time(successive)
    assert four <= (1.0 + SUCCESSIVE_TOLERANCE) * SUCCESSIVE_BATCHES * single, (
        f"{SUCCESSIVE_BATCHES} successive fresh table builds took {four:.3f} s against "
        f"{single:.3f} s for one ({four / single:.2f}x)"
    )
    assert single <= TABLES_OVER_RECFAST * recfast, (
        f"a fresh batch's tables took {single:.3f} s against {recfast:.3f} s for the "
        f"RECFAST solve alone ({single / recfast:.2f}x)"
    )
    benchmark.pedantic(successive, rounds=1, warmup_rounds=0, iterations=1)
