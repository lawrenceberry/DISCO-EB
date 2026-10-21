import math
import os
from dataclasses import dataclass
from functools import partial
from collections import OrderedDict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .component import Component
from .eb_model import (
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
    density_parameters,
    dtauda,
    evolve_background,
    layout_state_names,
    neutrino_momentum_bins,
    perturbation_model,
    perturbation_sparsity,
    setup_background_evolution,
)
from .thermodynamics_recfast import evaluate_thermo



# Hierarchy truncations. These are the settings of the components in
# :mod:`discoeb.eb_model` -- ``DISCO_LMAX_G`` etc. in the environment override
# their defaults when that module is imported -- mirrored here as module-level
# integers so the state-vector layout below and the accelerator-compiled
# right-hand side treat them as compile-time constants. A caller may rebind a
# component's class attribute (``Photons.LMAX_G = 11``) before the first solve,
# since the GPU pipeline builds its layout from the classes at solve time
# (:func:`_layout_for`). At the defaults the flat-LambdaCDM + massless-neutrino
# layout has NVAR == 50: an 11-variable densely coupled core bordered by three
# 13-variable free-streaming hierarchies, which is the structure
# :func:`discoeb.eb_model.perturbation_sparsity` declares and the sparse direct solver exploits.
LMAX_G = Photons.LMAX_G
"""Photon temperature hierarchy truncation (``Theta_0 ... Theta_LMAX_G``)."""

LMAX_POL = Photons.LMAX_POL
"""Photon E-mode polarization hierarchy truncation (``E_2 ... E_LMAX_POL``)."""

LMAX_NR = MasslessNeutrinos.LMAX_NR
"""Massless-neutrino hierarchy truncation (``N_0 ... N_LMAX_NR``)."""

LMAX_NU = MassiveNeutrinos.LMAX_NU
"""Massive-neutrino hierarchy truncation (``psi_0 ... psi_LMAX_NU`` per momentum bin)."""

IX_ETAK = 0
"""State-vector index of the metric perturbation ``etak = k eta``."""

IX_CLXC = 1
"""State-vector index of the CDM density contrast ``clxc``."""

IX_CLXB = 2
"""State-vector index of the baryon density contrast ``clxb``."""

IX_VB = 3
"""State-vector index of the baryon velocity ``vb``."""

IX_G = 4
"""Base index of the photon temperature hierarchy (``Theta_l`` at ``IX_G + l``)."""

IX_POL = IX_G + LMAX_G + 1
"""Base index of the photon polarization hierarchy (``E_l`` at ``IX_POL + (l-2)``)."""

IX_R = IX_POL + LMAX_POL - 1
"""Base index of the massless-neutrino hierarchy (``N_l`` at ``IX_R + l``)."""

NVAR = IX_R + LMAX_NR + 1
"""Total length of the perturbation state vector."""


# ============================================================
# MATTER POWER SPECTRUM
# ============================================================


def total_matter_density_contrast(
    grhoc_t: float,
    clxc: float,
    grhob_t: float,
    clxb: float,
) -> float:
    """Return the total-matter density contrast ``delta_m``.

    Density-weighted combination of the CDM and baryon contrasts,

    ``delta_m = (rho_c delta_c + rho_b delta_b) / (rho_c + rho_b)``,

    using the comoving density coefficients (the ``a`` factors cancel).
    """

    return (grhoc_t * clxc + grhob_t * clxb) / (grhoc_t + grhob_t)


def primordial_curvature_power(
    k: float,
    A_s: float,
    n_s: float,
    k_pivot: float,
) -> float:
    """Return the dimensionless primordial curvature power ``P_R(k)``.

    Power-law spectrum ``P_R(k) = A_s (k / k_pivot)^(n_s - 1)``.
    """

    return A_s * (k / k_pivot) ** (n_s - 1.0)


def matter_power_spectrum(
    k: float,
    delta_m: float,
    A_s: float,
    n_s: float,
    k_pivot: float,
) -> float:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3.

    With initial conditions normalized to unit comoving curvature, the evolved
    total-matter contrast ``delta_m(k)`` acts as a transfer function and

    ``P(k) = (2 pi^2 / k^3) P_R(k) delta_m(k)^2``,

    where ``P_R(k)`` is the dimensionless primordial curvature power. The contrast
    ``delta_m`` must be evaluated at the output time (e.g. ``a = 1``).
    """

    return (
        2.0
        * math.pi**2
        / k**3
        * primordial_curvature_power(k, A_s, n_s, k_pivot)
        * delta_m**2
    )
@dataclass(frozen=True)
class PerturbationLayout:
    """Enabled components and state-vector indices of the perturbation hierarchy.

    Attributes
    ----------
    lmaxg, lmaxpol, lmaxr : int
        Photon-temperature, E-mode-polarization, and massless-neutrino hierarchy
        truncations.
    nqmax : int
        Number of massive-neutrino momentum bins. ``0`` disables massive
        neutrinos entirely (no state variables allocated).
    lmaxnu : int
        Massive-neutrino multipole-hierarchy truncation (used only when
        ``nqmax > 0``).
    enable_dark_energy : bool
        Whether the dynamical dark-energy fluid perturbations are evolved.
    """

    lmaxg: int = 15
    lmaxpol: int = 15
    lmaxr: int = 15
    nqmax: int = 0
    lmaxnu: int = 15
    enable_dark_energy: bool = False

    def __post_init__(self):
        if self.lmaxg < 2 or self.lmaxpol < 2 or self.lmaxr < 2:
            raise ValueError("lmaxg, lmaxpol, lmaxr must each be >= 2")
        if self.nqmax < 0:
            raise ValueError("nqmax must be non-negative")
        if self.nqmax > 0 and self.lmaxnu < 2:
            raise ValueError("lmaxnu must be >= 2 when massive neutrinos are enabled")

    # --- Core (always present) -------------------------------------------------

    @property
    def ix_etak(self) -> int:
        """Index of the synchronous-gauge metric perturbation ``etak = k eta``."""
        return 0

    @property
    def ix_clxc(self) -> int:
        """Index of the CDM density contrast."""
        return 1

    @property
    def ix_clxb(self) -> int:
        """Index of the baryon density contrast."""
        return 2

    @property
    def ix_vb(self) -> int:
        """Index of the baryon velocity."""
        return 3

    @property
    def ix_g(self) -> int:
        """Base index of the photon temperature hierarchy (``Theta_l`` at ix_g + l)."""
        return 4

    @property
    def ix_pol(self) -> int:
        """Base index of the polarization hierarchy (``E_l`` at ix_pol + (l - 2))."""
        return self.ix_g + self.lmaxg + 1

    @property
    def ix_r(self) -> int:
        """Base index of the massless-neutrino hierarchy (``N_l`` at ix_r + l)."""
        return self.ix_pol + self.lmaxpol - 1

    @property
    def _ix_after_massless(self) -> int:
        """First index after the always-present core hierarchies."""
        return self.ix_r + self.lmaxr + 1

    # --- Optional: dynamical dark energy --------------------------------------

    @property
    def has_dark_energy(self) -> bool:
        return self.enable_dark_energy

    @property
    def ix_clxq(self) -> int:
        """Index of the dark-energy density contrast (only if dark energy enabled)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless

    @property
    def ix_thetaq(self) -> int:
        """Index of the dark-energy velocity (only if dark energy enabled)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless + 1

    @property
    def _ix_after_dark_energy(self) -> int:
        return self._ix_after_massless + (2 if self.enable_dark_energy else 0)

    # --- Optional: massive neutrinos ------------------------------------------

    @property
    def has_massive_neutrinos(self) -> bool:
        return self.nqmax > 0

    @property
    def ix_massive_nu(self) -> int:
        """Base index of the massive-neutrino momentum-bin hierarchies.

        The multipole ``psi_l`` of momentum bin ``q`` is stored at
        ``ix_massive_nu + q * (lmaxnu + 1) + l`` (**bin-major, multipole-minor**),
        so that each bin's free-streaming tail ``psi_3 ... psi_lmaxnu`` is
        contiguous and is one tridiagonal block of the sparsity pattern
        (:func:`discoeb.eb_model.perturbation_sparsity`).
        """
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        return self._ix_after_dark_energy

    def ix_psi_base(self, q: int) -> int:
        """Return the state index of ``psi_0`` for momentum bin ``q``."""
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        if not (0 <= q < self.nqmax):
            raise IndexError(f"momentum bin q={q} out of range [0, {self.nqmax})")
        return self.ix_massive_nu + q * (self.lmaxnu + 1)

    def ix_psi(self, l: int, q: int) -> int:
        """Return the state index of massive-neutrino multipole ``psi_l`` of bin ``q``."""
        if not (0 <= l <= self.lmaxnu):
            raise IndexError(f"multipole l={l} out of range [0, {self.lmaxnu}]")
        return self.ix_psi_base(q) + l

    # --- Total size -----------------------------------------------------------

    @property
    def nvar(self) -> int:
        """Total length of the trimmed perturbation state vector."""
        n = self._ix_after_dark_energy
        if self.nqmax > 0:
            n += self.nqmax * (self.lmaxnu + 1)
        return n

    @classmethod
    def from_cosmology(
        cls,
        cosmology,
        *,
        lmaxg: int = 15,
        lmaxpol: int = 15,
        lmaxr: int = 15,
        lmaxnu: int = 15,
        nqmax: int = 5,
    ) -> "PerturbationLayout":
        """Return the trimmed layout implied by a cosmology's enabled components.

        Dynamical dark energy is enabled when ``w_DE_0 != -1`` or ``w_DE_a != 0``;
        massive neutrinos are enabled (with ``nqmax`` momentum bins) when
        ``num_massive_neutrinos > 0``. Curvature does not affect the layout.
        """

        enable_de = (cosmology.w_DE_0 != -1.0) or (cosmology.w_DE_a != 0.0)
        nqmax_massive = nqmax if cosmology.num_massive_neutrinos > 0 else 0
        return cls(
            lmaxg=lmaxg,
            lmaxpol=lmaxpol,
            lmaxr=lmaxr,
            nqmax=nqmax_massive,
            lmaxnu=lmaxnu,
            enable_dark_energy=enable_de,
        )


FLAT_MASSLESS_LAYOUT = PerturbationLayout()
"""Default flat-LambdaCDM + massless-neutrino layout (NVAR == 50).

Reproduces the fixed index constants of :mod:`discoeb.perturbations`.
"""
class _ParamCosmology(NamedTuple):
    """Hashable view of the historical DISCO-EB parameter dictionary."""

    Omegam: float
    Omegab: float
    w_DE_0: float
    w_DE_a: float
    cs2_DE: float
    Omegak: float
    A_s: float
    n_s: float
    H0: float
    T_cmb: float
    Y_He: float
    Neff_massless: float
    num_massive_neutrinos: float
    mnu: float
    k_pivot: float
    tau_reio: float = 0.0  # reionization optical depth (0: no reionization)

    @property
    def h(self):
        return self.H0 / 100.0

    @property
    def Omegac(self):
        return self.Omegam - self.Omegab

    @property
    def omega_b_h2(self):
        return self.Omegab * self.h**2

    @property
    def omega_c_h2(self):
        return self.Omegac * self.h**2

    @property
    def tau_reion(self):
        return self.tau_reio

    @property
    def N_eff(self):
        return self.Neff_massless + self.num_massive_neutrinos

    def to_background_params(self):
        return {
            "Omegam": self.Omegam,
            "Omegab": self.Omegab,
            "w_DE_0": self.w_DE_0,
            "w_DE_a": self.w_DE_a,
            "cs2_DE": self.cs2_DE,
            "Omegak": self.Omegak,
            "A_s": self.A_s,
            "n_s": self.n_s,
            "H0": self.H0,
            "Tcmb": self.T_cmb,
            "YHe": self.Y_He,
            "Neff": self.Neff_massless,
            "Nmnu": self.num_massive_neutrinos,
            "mnu": self.mnu,
            "tau_reio": self.tau_reio,
        }


def _as_cosmology(cosmology):
    """Normalize a legacy parameter dictionary for the accelerator path."""

    if not isinstance(cosmology, dict):
        return cosmology
    return _ParamCosmology(
        Omegam=float(cosmology["Omegam"]),
        Omegab=float(cosmology["Omegab"]),
        w_DE_0=float(cosmology.get("w_DE_0", -1.0)),
        w_DE_a=float(cosmology.get("w_DE_a", 0.0)),
        cs2_DE=float(cosmology.get("cs2_DE", 1.0)),
        Omegak=float(cosmology.get("Omegak", 0.0)),
        A_s=float(cosmology["A_s"]),
        n_s=float(cosmology["n_s"]),
        H0=float(cosmology["H0"]),
        T_cmb=float(cosmology["Tcmb"]),
        Y_He=float(cosmology["YHe"]),
        Neff_massless=float(cosmology["Neff"]),
        num_massive_neutrinos=float(cosmology.get("Nmnu", 0.0)),
        mnu=float(cosmology.get("mnu", 0.0)),
        k_pivot=float(cosmology.get("k_p", 0.05)),
        tau_reio=float(cosmology.get("tau_reio", 0.0)),
    )

# Solver and grid presets for the accelerated matter-power path.
N_THERMO_GRID = 2048
N_BACKGROUND_GRID = 4096
THERMO_BATCH_CHUNK = 128
"""Cosmologies per vmapped background solve in :func:`build_thermo_tables_batch`.

One lane per cosmology, so a launch costs the same whether it carries one
cosmology or 128, and a batch is solved in as few launches as possible. A
partial chunk is padded to this size by repeating its last cosmology, so every
batch of up to 128 cosmologies runs the one compiled program.
"""
PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = 1.0e-2
PERTURB_MAX_STEPS = 20000
TAU_START = 0.1

# Packed-parameter row layout consumed by the device rhs/jac.
#   p = (grhog, grhornomass, grhoc, grhob, grhov,
#        tau_start, tau_end, k, cosmology_idx, save_index,
#        w_DE_0, w_DE_a, cs2_DE, grhok, grhor_nu, n_mnu, amnu,
#        eps, dir_grhog, dir_grhornomass, dir_grhoc, dir_grhob, dir_grhov)
# grhok = grhom * Omega_k is the (constant) comoving curvature density
# coefficient 8*pi*G*rho_K a^2; it is 0 for a flat cosmology.
# The rhs reads density ``c`` as ``p[c] + p[IX_EPS] * p[IX_DIR + c]``. ``eps``
# is zero in every primal solve, so the trajectory is what it always was; its
# derivative with respect to ``eps`` alone is the directional derivative along
# the row's ``dir`` vector. That is what lets a trajectory carry a single
# sensitivity column for a direction of its own -- see
# :func:`matter_power_spectrum_jax`.
IX_TAU_START = 5
IX_TAU_END = 6
IX_K = 7
IX_COSMOLOGY = 8
IX_SAVE_INDEX = 9
IX_W_DE_0 = 10
IX_W_DE_A = 11
IX_CS2_DE = 12
IX_GRHOK = 13
IX_GRHOR_NU = 14
IX_NMNU = 15
IX_AMNU = 16
IX_EPS = 17
IX_DIR = 18
N_PARAM = IX_DIR + 5

PARAM_COLUMNS = {
    "grhog_nominal": 0,
    "grhornomass_nominal": 1,
    "grhoc_nominal": 2,
    "grhob_nominal": 3,
    "grhov_nominal": 4,
    "tau_start": IX_TAU_START,
    "tau_end": IX_TAU_END,
    "k": IX_K,
    "cosmology_tag": IX_COSMOLOGY,
    "save_index": IX_SAVE_INDEX,
    "w_DE_0": IX_W_DE_0,
    "w_DE_a": IX_W_DE_A,
    "cs2_DE": IX_CS2_DE,
    "grhok": IX_GRHOK,
    "grhor_nu": IX_GRHOR_NU,
    "n_mnu": IX_NMNU,
    "amnu": IX_AMNU,
    "eps": IX_EPS,
    "dir_grhog": IX_DIR,
    "dir_grhornomass": IX_DIR + 1,
    "dir_grhoc": IX_DIR + 2,
    "dir_grhob": IX_DIR + 3,
    "dir_grhov": IX_DIR + 4,
}
"""The row's columns by the names the component model's providers take them under
(:mod:`discoeb.eb_model`): ``grhog_nominal`` is read into the equations as
``grhog = grhog_nominal + eps * dir_grhog``."""

THERMO_TABLE_DTYPE = np.float32
TRAJECTORIES_PER_BLOCK = 32
"""Wave modes packed into one CUDA block by modax's thread-local Rodas5P kernel."""

CUDA_MAX_REGISTERS = int(os.environ.get("DISCO_CUDA_MAX_REGISTERS", "168"))
"""Per-thread register cap for that kernel: fewer registers, more resident warps."""

NQMAX = MassiveNeutrinos.NQMAX
"""Massive-neutrino momentum bins used by both the background and the hierarchy.

The ``NQMAX`` setting of :class:`discoeb.eb_model.MassiveNeutrinos`
(``DISCO_NQMAX``). Each bin adds ``lmaxnu + 1 == 16`` state variables and the
generated dense Jacobian scales as ``nvar^2``, so this sets the numba compile
time: 3 bins give ``nvar = 98``, 5 bins give ``nvar = 130`` and compile ~4x
slower for no gain in P(k) accuracy against CLASS.
"""


_DENSITY_CACHE: "BoundedCache"


def _densities(cosmology) -> dict:
    """:func:`discoeb.eb_model.density_parameters` of a cosmology, as floats, memoised."""

    cosmology = _as_cosmology(cosmology)
    cached = _DENSITY_CACHE.get(cosmology)
    if cached is None:
        d = density_parameters(cosmology.to_background_params(), xp=np)
        cached = _DENSITY_CACHE[cosmology] = {k: float(v) for k, v in d.items()}
    return cached


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """Return the background ``grho`` density coefficients for ``cosmology``.

    ``(grhog, grhornomass, grhoc, grhob, grhov)`` from
    :func:`discoeb.eb_model.density_parameters`: ``grhornomass`` uses the
    *massless* neutrino count, and the dark-energy coefficient closes the
    density budget at ``a = 1`` with the model's own Friedmann sum, so it
    accounts for curvature and the massive-neutrino density.
    """

    d = _densities(cosmology)
    return (d["grhog"], d["grhornomass"], d["grhoc"], d["grhob"], d["grhov"])


class BoundedCache(OrderedDict):
    """A least-recently-used ``dict`` holding at most ``maxsize`` entries.

    The module memoizes per-cosmology and per-batch host work (thermodynamics
    tables, packed initial states, grid summaries). An inference loop proposes
    a fresh batch every iteration, so an unbounded memo would grow without
    limit; this one forgets the oldest entries instead.
    """

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = int(maxsize)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            self.popitem(last=False)


THERMO_TABLE_CACHE_SIZE = 4096
"""Cosmologies whose thermodynamics tables are kept: a few dozen 128-batches."""

BATCH_CACHE_SIZE = 8
"""Batches whose prepared solve and packed initial states are kept."""

_THERMO_TABLE_CACHE: BoundedCache = BoundedCache(THERMO_TABLE_CACHE_SIZE)


def build_thermo_tables(cosmology, n_grid: int = N_THERMO_GRID):
    """Cached wrapper around :func:`_build_thermo_tables`.

    Each call runs a full RECFAST recombination solve plus the background
    quadrature, so it is far too expensive to repeat per evaluation -- a CMB batch
    needs every cosmology's ``tau0`` up front *and* again inside the solve
    preparation. The tables depend only on ``(cosmology, n_grid)``, so memoize
    them. A single cosmology is the one-lane case of
    :func:`build_thermo_tables_batch`, on the same compiled program.
    """

    return build_thermo_tables_batch((_as_cosmology(cosmology),), n_grid)[0]


def build_thermo_tables_batch(cosmologies, n_grid: int = N_THERMO_GRID):
    """Return one :func:`build_thermo_tables` tuple per cosmology, batched.

    The tables of each cosmology, memoized per cosmology, with the misses solved
    together: the RECFAST recombination history, the background quadrature *and*
    the resampling onto the device tables (:func:`_thermo_tables_device`) are
    one ``jax.vmap``-ed program over a chunk of :data:`THERMO_BATCH_CHUNK`
    cosmologies, so a fresh batch of 128 costs one launch -- a few hundred
    milliseconds, against ~0.8 s per cosmology solved one at a time, and against
    ~5 s when the resampling was done on the host cosmology by cosmology.

    ``vmap`` rather than the parameter batch axis :func:`batch_dimensions`
    threads through the splines. Under ``vmap`` each cosmology keeps its own
    adaptive step sequence; a batch axis would instead put all of them under the
    single scalar error norm of the thermodynamics ``PIDController``, tying every
    cosmology's step size to the worst one in the batch.

    Not bit-identical to the serial builder. ``tau_grid`` and ``tau0`` match
    exactly, but batching re-associates the arithmetic inside the RECFAST solve,
    and at its default ``rtol = 1e-5`` that is enough to flip step-acceptance
    decisions: ``xe`` moves by up to ~2e-3 at early times, in cosmologies where
    the solve is barely converged to begin with. That is a property of the
    tolerance, not of batching -- the serial solve is itself only good to ~8e-2
    there against a converged reference. It does not reach the observable:
    ``P(k)`` shifts by at most ~7e-6, against a test gate of 5e-3.
    """

    order = []
    todo = []
    found = {}
    for cosmology in cosmologies:
        key = (_as_cosmology(cosmology), n_grid)
        if key not in found:
            cached = _THERMO_TABLE_CACHE.get(key)
            if cached is not None:
                found[key] = cached
            elif key not in todo:
                todo.append(key)
        order.append(key)

    for start in range(0, len(todo), THERMO_BATCH_CHUNK):
        chunk = todo[start : start + THERMO_BATCH_CHUNK]
        tables = _build_thermo_tables_batch([key[0] for key in chunk], n_grid)
        found.update(zip(chunk, tables))
        _THERMO_TABLE_CACHE.update(zip(chunk, tables))

    return tuple(found[key] for key in order)


_SPLINE_INVERSE_CACHE: dict[int, np.ndarray] = {}


def _not_a_knot_inverse(n: int) -> np.ndarray:
    """Inverse of the uniform-grid not-a-knot cubic-spline system for the knot second derivatives.

    On a uniform grid of spacing ``h`` the second derivatives ``M`` of the
    not-a-knot cubic spline through ``y`` solve ``A M = 6 D2 y / h**2`` with
    ``D2`` the second difference, interior rows ``(1, 4, 1)`` and the end
    rows ``(1, -2, 1)`` (continuity of the third derivative at the second and
    second-to-last knots). ``A`` depends on ``n`` alone, so its inverse is built
    once and the device forms every cosmology's ``seconds`` as one matrix
    product -- the same numbers as ``scipy.interpolate.CubicSpline(x, y)(x, 2)``
    on that grid, to 1e-13 of their scale.
    """

    cached = _SPLINE_INVERSE_CACHE.get(n)
    if cached is None:
        a = np.zeros((n, n))
        idx = np.arange(1, n - 1)
        a[idx, idx - 1] = 1.0
        a[idx, idx] = 4.0
        a[idx, idx + 1] = 1.0
        a[0, :3] = (1.0, -2.0, 1.0)
        a[-1, -3:] = (1.0, -2.0, 1.0)
        cached = np.linalg.inv(a)
        _SPLINE_INVERSE_CACHE[n] = cached
    return cached


@partial(jax.jit, static_argnames=("n_grid",))
def _thermo_tables_device(stacked, spline_inverse, n_grid: int):
    """The device tables of a stack of cosmologies, one vmap lane each.

    Runs :func:`evolve_background` and then the resampling onto the layout the
    numba-CUDA callbacks read -- the conformal-time quadrature, the uniform
    ``log(tau)`` grid to each cosmology's own ``tau0``, the spline evaluations
    of ``a``, ``kappa'``, ``c_s^2`` and ``rho_nu`` on it, and their not-a-knot
    second derivatives (:func:`_not_a_knot_inverse`) -- as one program. Returns
    ``(tau_grid, values, seconds, tau0)`` stacked over the lanes.
    """

    a_background = jnp.geomspace(1.0e-9, 1.0, N_BACKGROUND_GRID)

    def one(params):
        tau_reio = params["tau_reio"]
        background = evolve_background(
            param={k: v for k, v in params.items() if k != "tau_reio"},
            thermo_module="RECFAST", num_thermo=n_grid,
        )
        dtauda_values = jax.vmap(dtauda, in_axes=(0, None))(a_background, background["densities"])
        steps = 0.5 * (dtauda_values[1:] + dtauda_values[:-1]) * jnp.diff(a_background)
        tau_background = jnp.concatenate(
            (jnp.zeros((1,)), jnp.cumsum(steps))
        ) + jnp.reshape(background["taumin"], (-1,))[0]
        tau0 = tau_background[-1]
        log_tau_grid = jnp.linspace(jnp.log(TAU_START), jnp.log(tau0), n_grid)
        tau_grid = jnp.exp(log_tau_grid)
        a_vals = jnp.interp(tau_grid, tau_background, a_background)
        loga_eval = jnp.log(a_vals)

        xe_spline = background["xe_of_loga_spline"]
        xe_vals = xe_spline.evaluate(loga_eval)[:, 0]
        a_recfast_start = 1.0 / 3501.0
        xe_recfast_start = xe_spline.evaluate(jnp.log(jnp.asarray([a_recfast_start])))[0, 0]
        xe_vals = jnp.where(a_vals < a_recfast_start, xe_recfast_start, xe_vals)
        akthom = (
            2.3038921003709498e-9
            * (1.0 - params["YHe"])
            * params["Omegab"]
            * params["H0"] ** 2
        )
        # Reionization (CAMB's parametrization): hydrogen and HeII reionize
        # together in a tanh step of width delta_z = 0.5 in y = (1 + z)^1.5
        # about z_re, HeIII at z = 3.5; z_re is set by the optical depth
        # tau_reio = int kappa' dtau, found by bisection on the background grid.
        f_he = params["YHe"] / (4.0 * (1.0 - params["YHe"]))  # n_He / n_H
        xe_rec_background = xe_spline.evaluate(jnp.log(a_background))[:, 0]
        xe_rec_background = jnp.where(a_background < a_recfast_start, xe_recfast_start, xe_rec_background)
        dtau_background = jnp.gradient(tau_background)

        def xe_reio(a, xe_rec, z_re):
            z = 1.0 / a - 1.0
            y, y_re = (1.0 + z) ** 1.5, (1.0 + z_re) ** 1.5
            delta_y = 1.5 * jnp.sqrt(1.0 + z_re) * 0.5
            step = 0.5 * (1.0 + jnp.tanh((y_re - y) / delta_y))
            he3 = 0.5 * (1.0 + jnp.tanh((3.5 - z) / 0.5))
            return (1.0 + f_he - xe_rec) * step + f_he * he3

        def tau_of(z_re):
            return jnp.sum(xe_reio(a_background, xe_rec_background, z_re) * akthom / a_background**2 * dtau_background)

        def bisect(carry, _):
            lo, hi = carry
            mid = 0.5 * (lo + hi)
            too_much = tau_of(mid) > tau_reio
            return (jnp.where(too_much, lo, mid), jnp.where(too_much, mid, hi)), None

        (lo, hi), _ = jax.lax.scan(bisect, (jnp.asarray(1.0), jnp.asarray(40.0)), None, length=40)
        z_re = 0.5 * (lo + hi)
        xe_vals = xe_vals + jnp.where(tau_reio > 0.0, xe_reio(a_vals, xe_vals, z_re), 0.0)
        opacity_vals = xe_vals * akthom / a_vals**2
        cs2_vals = background["cs2a_of_loga_spline"].evaluate(loga_eval)[:, 0] / a_vals
        rhonu_vals = jnp.exp(
            background["logrhonu_of_loga_spline"].evaluate(loga_eval)[:, 0]
        )

        values = jnp.stack((a_vals, opacity_vals, cs2_vals, rhonu_vals))
        h = log_tau_grid[1] - log_tau_grid[0]
        second_difference = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
        rhs = jnp.zeros_like(values).at[:, 1:-1].set(6.0 * second_difference / h**2)
        seconds = rhs @ spline_inverse.T
        return tau_grid, values, seconds, tau0

    return jax.vmap(one)(stacked)


@partial(jax.jit, static_argnames=("n_grid",))
def _solve_thermal_histories(stacked, n_grid: int):
    """The RECFAST solve of ``evolve_background`` alone, one vmap lane per cosmology."""

    def one(params):
        param = setup_background_evolution(amin=1e-9, amax=1.01, param=dict(params))
        return evaluate_thermo(param=param, num_thermo=n_grid, rtol=1e-5, atol=1e-7)

    return jax.vmap(one)(stacked)


def _stack_background_params(cosmologies) -> dict:
    """The cosmologies' background parameters stacked by name, one vmap lane each."""

    rows = [c.to_background_params() for c in cosmologies]
    return {key: jnp.asarray([row[key] for row in rows], dtype=jnp.float64) for key in rows[0]}


def thermal_history_batch(cosmologies, n_grid: int = N_THERMO_GRID) -> dict:
    """Solve the recombination history of a batch of cosmologies in one launch.

    Returns host arrays: ``a`` of shape ``(n_grid + 1,)`` (the adaptive RECFAST
    grid), and ``xe``, ``Tm`` and ``cs2`` of shape ``(n_cosmologies, n_grid + 1)``:
    the free-electron fraction ``n_e / n_H``, the matter temperature in K and the
    baryon sound speed squared. It is the thermodynamics half of
    :func:`build_thermo_tables_batch`, exposed so the recombination solve can be
    checked and timed on its own.
    """

    cosmologies = tuple(_as_cosmology(cosmology) for cosmology in cosmologies)
    stacked = _stack_background_params(cosmologies)
    a, cs2, Tm, mu, xe, dxeda = _solve_thermal_histories(stacked, n_grid)
    return {
        "a": np.asarray(a)[0],
        "xe": np.asarray(xe)[:, :, 0],
        "Tm": np.asarray(Tm)[:, :, 0],
        "cs2": np.asarray(cs2)[:, :, 0],
    }


def _build_thermo_tables_batch(cosmologies, n_grid: int = N_THERMO_GRID):
    """Uncached :func:`build_thermo_tables_batch`, for one chunk of cosmologies.

    Returns one ``(tau_grid, values, seconds, tau0)`` tuple of host arrays per
    cosmology. The chunk is padded to :data:`THERMO_BATCH_CHUNK` lanes with its
    last cosmology so that every chunk runs the one compiled program.
    """

    n = len(cosmologies)
    padded = tuple(cosmologies) + (cosmologies[-1],) * (THERMO_BATCH_CHUNK - n)
    stacked = _stack_background_params(padded)
    tau_grid, values, seconds, tau0 = (
        np.asarray(x)
        for x in _thermo_tables_device(stacked, _not_a_knot_inverse(n_grid), n_grid)
    )
    return tuple(
        (tau_grid[i], values[i], seconds[i], float(tau0[i])) for i in range(n)
    )


def make_params(cosmology, k_values: np.ndarray, tau_end: float) -> np.ndarray:
    """Return per-mode packed perturbation parameters for the device rhs.

    One row per wave mode, its columns named by :data:`PARAM_COLUMNS`. The
    cosmology tag is 0 and the sensitivity direction zero: ``eps`` and ``dir``
    are zero, so the densities the rhs sees are exactly the nominal ones.
    """

    d = _densities(cosmology)
    k_values = np.asarray(k_values, dtype=np.float64)
    row = {
        "grhog_nominal": d["grhog"],
        "grhornomass_nominal": d["grhornomass"],
        "grhoc_nominal": d["grhoc"],
        "grhob_nominal": d["grhob"],
        "grhov_nominal": d["grhov"],
        "tau_start": TAU_START,
        "tau_end": tau_end,
        "save_index": 1.0,
        "w_DE_0": cosmology.w_DE_0,
        "w_DE_a": cosmology.w_DE_a,
        "cs2_DE": cosmology.cs2_DE,
        "grhok": d["grhok"],
        "grhor_nu": d["grhor_nu"],
        "n_mnu": d["n_mnu"],
        "amnu": d["amnu"],
    }
    params = np.zeros((len(k_values), N_PARAM), dtype=np.float64)
    for name, column in PARAM_COLUMNS.items():
        params[:, column] = row.get(name, 0.0)
    params[:, IX_K] = k_values
    return params


def make_initial_states(cosmology, k_values: np.ndarray, layout) -> np.ndarray:
    """Return adiabatic initial states for every wave mode, shape ``(n_k, nvar)``.

    Each state's initial condition is declared on its component
    (:mod:`discoeb.eb_model`): CAMB's adiabatic growing-mode series at
    ``k tau << 1`` for the metric, the fluids and the radiation hierarchies,
    ``clxq = (1 + w_0) clxc`` at rest for the dark-energy fluid, and the
    massless hierarchy scaled by ``dlnf0/dlnq`` for each massive-neutrino bin.
    """

    grhog, grhornomass, grhoc, grhob, grhov = density_coefficients(cosmology)
    return perturbation_model(layout).initial_state(
        k_values,
        tau=TAU_START,
        grhog=grhog,
        grhornomass=grhornomass,
        grhoc=grhoc,
        grhob=grhob,
        grhov=grhov,
        w_DE_0=cosmology.w_DE_0,
        w_DE_a=cosmology.w_DE_a,
    )


class ThermalHistory(Component):
    """``a``, ``opacity``, ``cs2`` and ``rhonu`` at ``tau``, read off device spline tables.

    The tables are stacked over cosmologies: ``values`` and ``rescaled_seconds``
    (the spline's second derivatives times ``h^2 / 6`` for the uniform step
    ``h`` in ``log(tau)``, so the evaluator is a Horner polynomial in the local
    coordinate) have shape ``(n_cosmologies, n_channels, n_grid)`` and
    ``tau_min`` / ``inv_dtau`` are ``(n_cosmologies,)``. The device evaluator
    selects a cosmology's table by the ``IX_COSMOLOGY`` tag on each
    trajectory's parameter row, converted once per right-hand side call (the
    row is a tuple in modax's JVP path, so it cannot be read back by bit
    pattern), so one compiled kernel solves an arbitrary batch of cosmologies.

    The tables may be host arrays, uploaded here, or device arrays, used in
    place. The spline reads the device arrays' *contents at launch*, not at
    build time, so a caller that keeps one set of device buffers per batch
    shape and refills them per batch (:class:`DeviceTableSlot`) runs every
    batch on the one kernel compiled against those buffers.
    """

    def __init__(self, tau_min, inv_dtau, values, rescaled_seconds):
        from numba_cuda_mlir import cuda

        def on_device(table, dtype):
            if cuda.is_cuda_array(table):
                return table
            return cuda.to_device(np.ascontiguousarray(table, dtype=dtype))

        n_grid = values.shape[-1]
        tau_min_dev = on_device(tau_min, np.float64)
        inv_dtau_dev = on_device(inv_dtau, np.float64)
        values_dev = on_device(values, THERMO_TABLE_DTYPE)
        rescaled_seconds_dev = on_device(rescaled_seconds, THERMO_TABLE_DTYPE)

        @cuda.jit(device=True)
        def uniform_spline_eval(x, cosmology_idx, channel):
            x_min = tau_min_dev[cosmology_idx]
            inv_dx = inv_dtau_dev[cosmology_idx]

            # 1. Non-branching index calculation
            i = (x - x_min) * inv_dx
            idx = max(0, min(int(i), n_grid - 2))

            # 2. Normalized local coordinate t in [0, 1]
            t = i - idx

            # 3. Load coefficients
            y0 = values_dev[cosmology_idx, channel, idx]
            y1 = values_dev[cosmology_idx, channel, idx + 1]
            k0 = rescaled_seconds_dev[cosmology_idx, channel, idx]
            k1 = rescaled_seconds_dev[cosmology_idx, channel, idx + 1]

            # 4. Transform to Horner evaluation scheme
            c0 = y0
            c1 = (y1 - y0) - (2.0 * k0 + k1)
            c2 = 3.0 * k0
            c3 = k1 - k0

            return c0 + t * (c1 + t * (c2 + t * c3))

        # The providers close over the spline, which closes over the device
        # arrays: a device function cannot take ``self``.
        def a(log_tau, cosmology_idx):
            return uniform_spline_eval(log_tau, cosmology_idx, 0)

        def opacity(log_tau, cosmology_idx):
            return max(uniform_spline_eval(log_tau, cosmology_idx, 1), 1e-30)

        def cs2(log_tau, cosmology_idx):
            return max(uniform_spline_eval(log_tau, cosmology_idx, 2), 0.0)

        def rhonu(log_tau, cosmology_idx):
            return uniform_spline_eval(log_tau, cosmology_idx, 3)

        self.spline = uniform_spline_eval
        self.a, self.opacity, self.cs2, self.rhonu = a, opacity, cs2, rhonu

    @staticmethod
    def log_tau(self, tau):
        return math.log(tau)

    @staticmethod
    def cosmology_idx(self, cosmology_tag):
        return int(cosmology_tag)


def build_numba_rhs(layout, tau_min, inv_dtau, values, rescaled_seconds):
    """Return the numba-CUDA right-hand side ``f(y, tau, p) -> tuple`` for a layout.

    The right-hand side is :func:`discoeb.eb_model.perturbation_model` built
    by :meth:`discoeb.component.Model.build`: every species' equations and the
    interactions between them, their dependencies resolved by name and emitted
    as one straight-line function. That is the form Enzyme can differentiate
    for the Jacobian and ``df/dt``: fixed-size tuples of scalars, ``y`` and
    ``p`` indexed only at build-time constants (a run-time index into a tuple
    makes numba emit a bounds check Enzyme refuses), no device-function call
    but the thermodynamics spline. The same function evaluated over array rows
    serves the kernel's primal stage evaluations.

    The multipole hierarchies are not looped over: each multipole above the
    quadrupole is its own line, with its ``l`` and neighbours written in, so
    the truncations are read from ``layout`` and the state-vector layout, the
    initial conditions and the sparsity pattern change together. The
    dynamical-dark-energy fluid and the massive-neutrino momentum bins are
    components registered only when the layout enables them (the
    flat-LambdaCDM kernel is unchanged). The thermodynamics tables are read by
    a :class:`ThermalHistory`; see there for their layout and for how one
    kernel serves every batch of a shape.

    The generated text is ``rhs._generated_source``.
    """

    thermal_history = ThermalHistory(tau_min, inv_dtau, values, rescaled_seconds)
    model = perturbation_model(layout, thermal_history)
    expected = layout_state_names(layout)
    if tuple(model.state) != expected or model.nvar != layout.nvar:
        raise AssertionError(
            "the component model's state order does not reproduce the layout's indices"
        )
    return model.build(PARAM_COLUMNS)


class DeviceTableSlot:
    """Device buffers of a fixed shape that one compiled kernel reads its tables from.

    A numba-CUDA kernel that closes over a device array bakes in the array's
    address, so the right-hand side (and the CMB source hook) built against a
    batch's thermodynamics tables is a different kernel for every batch, and
    every batch of an inference loop would compile its own, ~45 s each. A slot
    instead owns device buffers sized for a batch *shape* -- the number of
    cosmologies and the table length -- and the device function compiled
    against them; :meth:`load` copies a batch's tables into the buffers, so the
    same kernel serves every batch of that shape.

    The copy is ordered against the launches by a device-wide synchronisation
    on either side of it: every earlier launch (on XLA's streams too, which do
    not synchronise with numba's) has finished reading the old tables before
    they are overwritten, and the new ones have landed before anything is
    launched against them. That wait is only taken when the slot changes hands,
    never when a batch is re-solved. A kernel reads the tables loaded at its
    launch, so one batch at a time per slot: loading the next batch waits for
    the previous one's launches.
    """

    def __init__(self, shapes_dtypes, build):
        from numba_cuda_mlir import cuda

        self._cuda = cuda
        self.buffers = tuple(cuda.device_array(shape, dtype) for shape, dtype in shapes_dtypes)
        self.device_function = build(*self.buffers)
        self.loaded = None

    def load(self, key, *tables) -> None:
        """Make ``tables`` (host arrays, one per buffer) the slot's contents, under ``key``."""

        if self.loaded is not None and self.loaded == key:
            return
        self.loaded = None
        self._cuda.synchronize()
        for buffer, table in zip(self.buffers, tables, strict=True):
            buffer.copy_to_device(np.ascontiguousarray(table, dtype=buffer.dtype))
        self._cuda.synchronize()
        self.loaded = key


class _PreparedSolve(NamedTuple):
    """Compiled artifacts shared by every cosmology of a (possibly batched) solve."""

    cosmologies: tuple
    layout: object
    tables: tuple  # per-cosmology (tau_grid, values, seconds, tau0)
    tau0: np.ndarray  # (n_cosmologies,)
    ode_fn: object
    sparsity: object
    slot: DeviceTableSlot
    stacked: tuple  # host (tau_min, inv_dtau, values, rescaled_seconds) the slot is loaded with

    def bind(self) -> None:
        """Load this batch's thermodynamics tables into the kernel's buffers.

        Called right before every launch of ``ode_fn``; a no-op when the slot
        already holds them (a repeated solve of the same batch).
        """

        self.slot.load(self.cosmologies, *self.stacked)


_PREPARED_SOLVE_CACHE: BoundedCache = BoundedCache(BATCH_CACHE_SIZE)
_DENSITY_CACHE = BoundedCache(THERMO_TABLE_CACHE_SIZE)
_PACKED_BATCH_CACHE: BoundedCache = BoundedCache(BATCH_CACHE_SIZE)
_TABLE_SLOTS: dict = {}


def _stack_thermo_tables(tables):
    """Stack per-cosmology thermo tables for the multi-cosmology device evaluator.

    Returns ``(tau_min, inv_dtau, values, rescaled_seconds, tau0)`` where
    ``values`` and ``rescaled_seconds`` (second derivatives times ``h^2 / 6``,
    the form :func:`build_numba_rhs`'s Horner evaluator reads) are
    ``(n_cosmologies, n_channels, n_grid)`` and the rest are ``(n_cosmologies,)``.
    The device spline evaluator selects a cosmology's row by the per-trajectory
    ``IX_COSMOLOGY`` tag.
    """

    tau_min = np.array([np.log(t[0][0]) for t in tables], dtype=np.float64)
    inv_dtau = np.array(
        [(t[0].shape[0] - 1) / (np.log(t[0][-1]) - np.log(t[0][0])) for t in tables],
        dtype=np.float64,
    )
    values = np.stack([t[1] for t in tables])
    rescaled_seconds = np.stack([t[2]*(np.log(t[0][1]) - np.log(t[0][0]))**2/6.0 for t in tables])
    tau0 = np.array([t[3] for t in tables], dtype=np.float64)
    return tau_min, inv_dtau, values, rescaled_seconds, tau0


def _layout_for(cosmology) -> PerturbationLayout:
    """The perturbation layout the module presets imply for a cosmology."""

    return PerturbationLayout.from_cosmology(
        cosmology,
        lmaxg=Photons.LMAX_G,
        lmaxpol=Photons.LMAX_POL,
        lmaxr=MasslessNeutrinos.LMAX_NR,
        lmaxnu=MassiveNeutrinos.LMAX_NU,
        nqmax=MassiveNeutrinos.NQMAX,
    )


def _prepare_solve(cosmologies) -> _PreparedSolve:
    """Return (and cache) the compiled solve artifacts for a batch of cosmologies.

    The thermodynamics tables, the numba-CUDA right-hand side device function and
    the Jacobian sparsity pattern depend only on the cosmologies (and the
    module-level grid and quadrature presets), not on the wave modes, the save
    times, or the solver tolerances. The right-hand side and the sparsity
    pattern are also the identity keys of modax's compiled-kernel cache, and
    the kernel compilation is the *slow* step (tens of seconds). So the
    right-hand side is not built per batch: it is compiled once per batch
    *shape* against a :class:`DeviceTableSlot` of device buffers, and each
    batch's tables are copied into those buffers before its launches
    (:meth:`_PreparedSolve.bind`). A sampler proposing a fresh batch of the same
    size every iteration thus pays the thermodynamics solve and the copy, never
    a recompile.

    All cosmologies in a batch must share a perturbation layout (same dark-energy
    and massive-neutrino settings); one compiled kernel then solves them all,
    selecting each cosmology's thermodynamics table by the ``IX_COSMOLOGY`` tag.

    The prepared solves of the last :data:`BATCH_CACHE_SIZE` batches are kept
    (host tables only); the slots hold the device memory, one set of buffers
    per batch shape. :func:`clear_prepared_solve_cache` releases both.
    """

    cosmologies = tuple(_as_cosmology(cosmology) for cosmology in cosmologies)
    # Include the presets that change the compiled artifacts, so a caller that
    # rebinds them (e.g. NQMAX) does not get a stale kernel.
    key = (
        cosmologies,
        MassiveNeutrinos.NQMAX,
        Photons.LMAX_G,
        Photons.LMAX_POL,
        MasslessNeutrinos.LMAX_NR,
        MassiveNeutrinos.LMAX_NU,
        N_THERMO_GRID,
        TAU_START,
        TRAJECTORIES_PER_BLOCK,
    )
    cached = _PREPARED_SOLVE_CACHE.get(key)
    if cached is not None:
        return cached

    layout = _layout_for(cosmologies[0])
    for cosmology in cosmologies[1:]:
        if _layout_for(cosmology) != layout:
            raise ValueError(
                "all cosmologies in a batch must share a perturbation layout "
                "(same dark-energy / massive-neutrino settings)"
            )

    tables = build_thermo_tables_batch(cosmologies)
    tau_min, inv_dtau, values, rescaled_seconds, tau0 = _stack_thermo_tables(tables)
    n_cosmo, n_channels, n_grid = values.shape
    slot_key = key[1:] + (layout, n_cosmo, n_channels, n_grid)
    slot = _TABLE_SLOTS.get(slot_key)
    if slot is None:
        slot = DeviceTableSlot(
            (
                ((n_cosmo,), np.float64),
                ((n_cosmo,), np.float64),
                ((n_cosmo, n_channels, n_grid), THERMO_TABLE_DTYPE),
                ((n_cosmo, n_channels, n_grid), THERMO_TABLE_DTYPE),
            ),
            lambda *buffers: build_numba_rhs(layout, *buffers),
        )
        _TABLE_SLOTS[slot_key] = slot
    # The pattern is all modax needs: it colours it for the Enzyme sweeps, and
    # orders it with AMD, factorises it symbolically and compiles a sparse LU
    # and sparse triangular solves for that exact structure.
    prepared = _PreparedSolve(
        cosmologies,
        layout,
        tables,
        tau0,
        slot.device_function,
        perturbation_sparsity(layout),
        slot,
        (tau_min, inv_dtau, values, rescaled_seconds),
    )
    _PREPARED_SOLVE_CACHE[key] = prepared
    return prepared


def _pack_batch(prepared, k_values, tau_save=None):
    """Return (and cache) the packed ``(params, y0, t_span, order, k_sorted)``.

    Building the packed inputs means one :func:`make_initial_states` per cosmology
    -- i.e. ``n_cosmologies * n_k`` scalar :func:`adiabatic_initial_conditions`
    calls, tens of milliseconds for a large batch. That work depends only on the
    cosmologies and the wave modes (not the solver tolerances), so it is cached
    on ``(cosmologies, k, tau_save)``; a repeated solve of the same batch then
    skips straight to the kernel launch.

    ``tau_save`` is the shared grid of conformal times at which every trajectory
    is recorded (the CMB line-of-sight integration needs a dense one). It
    defaults to just the two endpoints, i.e. the final state only.
    """

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    tau_key = None if tau_save is None else np.asarray(tau_save, np.float64).tobytes()
    key = (prepared.cosmologies, k_arr.shape, k_arr.tobytes(), tau_key)
    cached = _PACKED_BATCH_CACHE.get(key)
    if cached is not None:
        return cached

    layout = prepared.layout
    n_cosmo = len(prepared.cosmologies)
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]
    n_k = len(k_sorted)

    # Pack trajectories as row = k_idx * n_cosmo + cosmology_idx, so a block of
    # TRAJECTORIES_PER_BLOCK consecutive rows shares one wave mode when
    # n_cosmo >= TRAJECTORIES_PER_BLOCK: neighbouring threads then take near-identical
    # adaptive steps, minimizing warp divergence.
    n_traj = n_k * n_cosmo
    params = np.empty((n_traj, N_PARAM), dtype=np.float64)
    y0 = np.empty((n_traj, layout.nvar), dtype=np.float64)
    for ci, cosmology in enumerate(prepared.cosmologies):
        base = make_params(cosmology, k_sorted, float(prepared.tau0[ci]))
        base[:, IX_COSMOLOGY] = float(ci)
        params[ci::n_cosmo] = base
        y0[ci::n_cosmo] = make_initial_states(cosmology, k_sorted, layout)
    params = np.ascontiguousarray(params)
    y0 = np.ascontiguousarray(y0)

    # Each trajectory stops at its own tau0 (IX_TAU_END); the shared save grid
    # runs to the largest tau0 and the trailing save slot holds each trajectory's
    # final state.
    if tau_save is None:
        t_span = np.asarray((TAU_START, float(np.max(prepared.tau0))), dtype=np.float64)
    else:
        t_span = np.ascontiguousarray(np.asarray(tau_save, dtype=np.float64))

    packed = (params, y0, t_span, order, k_sorted)
    _PACKED_BATCH_CACHE[key] = packed
    return packed


def clear_prepared_solve_cache() -> None:
    """Drop all cached prepared and packed solves and the device table slots.

    Releases the slots' device buffers; the next solve of each batch shape
    compiles its kernel again.
    """

    _PREPARED_SOLVE_CACHE.clear()
    _PACKED_BATCH_CACHE.clear()
    _THERMO_TABLE_CACHE.clear()
    _TABLE_SLOTS.clear()


def _launch(prepared, params, y0, t_span, *, rtol, atol, first_step, max_steps):
    """One modax Rodas5P launch of the prepared solve over the packed trajectories."""

    from modax.rodas5P import solve as rodas5P_solve

    prepared.bind()
    return rodas5P_solve(
        prepared.ode_fn,
        y0,
        t_span,
        params,
        sparsity=prepared.sparsity,
        lu_precision="fp32",
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
    )


def solve_perturbation_history(
    k_values,
    cosmology,
    *,
    tau_save=None,
    as_jax: bool = False,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve the perturbation hierarchy of one cosmology and return the saved history.

    The one-cosmology case of :func:`solve_perturbation_history_batch`, with the
    wave modes put back in the caller's order. ``tau_save`` are strictly
    increasing conformal times at which to record the state, from
    ``TAU_START`` to ``tau0`` (default: just the two endpoints, i.e. the final
    state). ``as_jax`` keeps the history on the device.

    Returns ``(history, layout, tables)``: ``history`` has shape
    ``(n_k, len(tau_save), nvar)`` and ``tables`` is the ``build_thermo_tables``
    tuple of the cosmology.
    """

    hist, k_sorted, prepared = solve_perturbation_history_batch(
        k_values, (cosmology,), tau_save=tau_save, rtol=rtol, atol=atol,
        first_step=first_step, max_steps=max_steps,
    )
    inv_order = np.argsort(np.argsort(np.asarray(k_values, dtype=np.float64), kind="stable"))
    history = hist[:, 0][jnp.asarray(inv_order)]
    if not as_jax:
        history = np.asarray(history)
    return history, prepared.layout, prepared.tables[0]


def solve_perturbation_history_batch(
    k_values,
    cosmologies,
    *,
    tau_save=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve a batch of cosmologies on a shared ``tau_save`` grid in one launch.

    All ``n_cosmologies * n_k`` trajectories are integrated by a single
    modax Rodas5P launch. Each trajectory still stops at its *own* ``tau0``
    (via ``IX_TAU_END``), so save times beyond a cosmology's ``tau0`` simply hold
    its frozen final state -- consumers must mask them (as
    :mod:`discoeb.cmb` does through the opacity and the source channels).

    Returns
    -------
    (history, k_sorted, prepared)
        ``history`` is a device-resident JAX array of shape
        ``(n_k, n_cosmologies, len(tau_save), nvar)``, with the wave modes in
        **ascending** order (``k_sorted``).

        The history is deliberately left in this layout: it is the solver's own
        packing, so the reshape is a free view. Transposing to a cosmology-major
        layout, or gathering the wave modes back into the caller's order, would
        each need a full second copy of the array -- 6.5 GB for a 128-cosmology,
        1000-save-point batch, enough to exhaust a 12 GB GPU on its own. Ascending
        ``k`` is what the downstream spline and the ``dln k`` quadrature want
        anyway.
    """

    prepared = _prepare_solve(tuple(cosmologies))
    layout = prepared.layout
    n_cosmo = len(prepared.cosmologies)

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_save)
    n_k = len(k_sorted)
    sol = _launch(prepared, params, y0, t_span, rtol=rtol, atol=atol, first_step=first_step, max_steps=max_steps)
    # Rows are packed k-major (row = k_idx * n_cosmo + cosmology_idx), so this
    # reshape is a free view -- no copy of the (potentially multi-GB) history.
    hist = jnp.asarray(sol).reshape(n_k, n_cosmo, len(t_span), layout.nvar)
    return hist, k_sorted, prepared


def solve_perturbations(k_values, cosmology, **solve_kwargs):
    """Solve the perturbation hierarchy for each ``k`` and return final states.

    Returns
    -------
    np.ndarray
        Final perturbation state for each wave mode, shape ``(n_k, nvar)``,
        evaluated at ``tau0`` (today).
    """

    solve_kwargs.pop("tau_save", None)
    hist, _, _ = solve_perturbation_history(k_values, cosmology, **solve_kwargs)
    return hist[:, -1, :]


def solve_matter_power_spectrum(k_values, cosmology, **solve_kwargs) -> np.ndarray:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3 for one cosmology."""

    return solve_matter_power_spectrum_batch(k_values, (cosmology,), **solve_kwargs)[0]


PHYSICAL_DENSITY_COLUMNS = (0, 1, 2, 3, 4)
"""Packed-parameter columns carrying the background densities.

``(grhog, grhornomass, grhoc, grhob, grhov)`` -- the entries of a parameter row
that are physical quantities one would differentiate. The rest of the row is
the wave number, two integration bounds, two integer selectors and the
dark-energy and massive-neutrino settings, none of which a gradient with
respect to "the cosmology" wants, and two of which are not differentiable at
all.
"""


def matter_power_spectrum_jax(
    k_values,
    cosmology,
    densities=None,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    one_column_per_trajectory: bool = True,
):
    """``P(k)`` as a JAX array, differentiable in the background densities.

    ``densities`` is the five-vector of :data:`PHYSICAL_DENSITY_COLUMNS`,
    defaulting to the cosmology's own. Differentiating with respect to it runs
    modax's continuous forward-sensitivity system alongside the state. The
    densities enter twice -- through the hierarchy, and through the
    matter-weighting of ``delta_m`` -- and both paths are differentiated, so
    this is the total derivative and not just the ODE's part of it. What is
    held fixed is the thermodynamics: the recombination history is tabulated on
    the host and uploaded, so this is a derivative at fixed ionisation history.

    The same sensitivity system can be laid out over the threads two ways.

    ``one_column_per_trajectory=True`` (the default)
        Every wave mode is replicated once per density, and replica ``j``
        carries one sensitivity column, along the unit direction ``e_j``,
        through the ``IX_EPS`` / ``IX_DIR`` columns of its parameter row. A
        thread then integrates ``2 * nvar`` components instead of ``6 * nvar``,
        and the five columns run side by side on threads the launch would
        otherwise leave idle -- 128 modes are four blocks of 32 on a GPU with
        dozens of SMs. The state and the Jacobian are recomputed by each
        replica, which costs nothing while the card has SMs to spare. Measured
        on an RTX 4070 SUPER for the 128 test modes (``jax.value_and_grad``,
        ``test_matter_power_spectrum_derivatives``): 0.28 s for the plain
        solve, 1.92 s for five columns in one thread, 0.59 s for one column
        per thread. A backward adjoint solve would need at least the forward
        solve plus a second pass of comparable cost, so this is already at the
        adjoint's floor with no trajectory storage.

    ``one_column_per_trajectory=False``
        One trajectory per mode carrying all five columns
        (``sens_param_columns=PHYSICAL_DENSITY_COLUMNS``). Five times less
        thread-time in total, since the state and the Jacobian are formed once
        per mode rather than once per column, so it is the arrangement to
        prefer once a batch is large enough to saturate the GPU by itself.

    Either way the sensitivities are kept out of step-size control
    (``sens_error_control=False``): they run to ~1e17 while the state is O(1),
    so letting them into the weighted error norm hands them the step size and
    costs 20x the solve for no accuracy the tolerances asked for. The LU is
    fp64, unlike the plain solve: the Rosenbrock-W property lets an fp32
    factorisation carry the *state* at full order, but the sensitivity rows are
    driven by that same matrix and measure ~25% error from it, against ~5e-4
    in fp64.
    """
    import jax.numpy as jnp

    cosmology = _as_cosmology(cosmology)
    prepared = _prepare_solve((cosmology,))
    cosmology = prepared.cosmologies[0]

    history, densities, k_sorted, order = _history_jax(
        prepared,
        k_values,
        densities,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        one_column_per_trajectory=one_column_per_trajectory,
    )
    y_final = history[:, -1, :]

    grhoc_val = densities[2]
    grhob_val = densities[3]
    delta_m = (
        grhoc_val * y_final[:, IX_CLXC] + grhob_val * y_final[:, IX_CLXB]
    ) / (grhoc_val + grhob_val)
    ks = jnp.asarray(k_sorted)
    pk_sorted = (
        2.0
        * jnp.pi**2
        / ks**3
        * cosmology.A_s
        * (ks / cosmology.k_pivot) ** (cosmology.n_s - 1.0)
        * delta_m**2
    )
    return pk_sorted[jnp.asarray(np.argsort(order))]


def _history_jax(
    prepared,
    k_values,
    densities=None,
    tau_save=None,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    one_column_per_trajectory: bool = True,
    sens_error_control: bool = False,
    error_weights=None,
):
    """The saved history of one cosmology's modes, differentiable in the densities.

    The solve behind :func:`matter_power_spectrum_jax` and
    :func:`discoeb.cmb.cl_power_spectrum_jax`, which documents the two thread
    layouts (``one_column_per_trajectory``) and the solver settings. ``prepared``
    is the :func:`_prepare_solve` of the single cosmology; ``tau_save`` is the
    save grid (default: the two endpoints, i.e. the final state only);
    ``error_weights`` are optional per-mode weights ``(n_k, nvar)`` in
    ascending ``k`` for the solver's error norm (modax's ``error_weights``),
    replicated for the sensitivity layout.

    Returns ``(history, densities, k_sorted, order)``: ``history`` is
    ``(n_k, n_save, nvar)`` in ascending ``k`` as a JAX array whose derivative
    with respect to ``densities`` is modax's forward sensitivity; ``densities``
    is the resolved five-vector; ``k_sorted`` and ``order`` are the sorted wave
    numbers and the sort of the caller's ``k_values``.
    """
    import jax
    import jax.numpy as jnp

    from modax.rodas5P import solve as rodas5P_solve

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_save)
    prepared.bind()
    columns = list(PHYSICAL_DENSITY_COLUMNS)
    n_dir = len(columns)
    n_k = params.shape[0]
    if densities is None:
        densities = params[0, columns]
    densities = jnp.asarray(densities)

    settings = dict(
        sparsity=prepared.sparsity,
        lu_precision="fp64",
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
        sens_error_control=sens_error_control,
        error_weights=error_weights,
    )
    settings_rep = dict(
        settings,
        error_weights=None
        if error_weights is None
        else np.repeat(np.asarray(error_weights), n_dir, axis=0),
    )
    t_span_j = jnp.asarray(t_span)

    def with_densities(base, d):
        return jnp.asarray(base).at[:, jnp.asarray(columns)].set(
            jnp.broadcast_to(d, (base.shape[0], n_dir))
        )

    if not one_column_per_trajectory:
        history = rodas5P_solve(
            prepared.ode_fn,
            jnp.asarray(y0),
            t_span_j,
            with_densities(params, densities),
            sens_param_columns=PHYSICAL_DENSITY_COLUMNS,
            **settings,
        )
    else:
        # Replica j of mode i is row i * n_dir + j, so a block of consecutive
        # rows still spans neighbouring wave modes and their similar step
        # sequences; each replica's direction is the unit vector e_j.
        params_rep = np.repeat(params, n_dir, axis=0)
        params_rep[:, IX_DIR : IX_DIR + n_dir] = np.tile(np.eye(n_dir), (n_k, 1))
        y0_rep = jnp.asarray(np.repeat(y0, n_dir, axis=0))
        y0_j = jnp.asarray(y0)

        @jax.custom_jvp
        def saved_history(d):
            return rodas5P_solve(
                prepared.ode_fn, y0_j, t_span_j, with_densities(params, d), **settings
            )

        @saved_history.defjvp
        def saved_history_jvp(primals, tangents):
            (d,), (dd,) = primals, tangents
            p_rep = with_densities(params_rep, d)

            def along_directions(eps):
                return rodas5P_solve(
                    prepared.ode_fn,
                    y0_rep,
                    t_span_j,
                    p_rep.at[:, IX_EPS].set(eps),
                    sens_param_columns=(IX_EPS,),
                    **settings_rep,
                )

            # d/d(eps) at eps = 0 along every replica at once: one joint launch
            # of n_k * n_dir trajectories, each carrying its own single column.
            n_rep = p_rep.shape[0]
            sol, dsol = jax.jvp(
                along_directions, (jnp.zeros(n_rep),), (jnp.ones(n_rep),)
            )
            # Every replica of a mode integrates the same state; keep the first.
            history = sol[::n_dir]
            # sens[i, j] = d y_i(tau) / d density_j at every save. The
            # contraction with the input tangent is linear, so JAX transposes
            # it and jax.grad falls out of the same rule with no adjoint solve.
            sens = dsol.reshape(n_k, n_dir, *dsol.shape[1:])
            return history, jnp.einsum("ijtv,j->itv", sens, dd)

        history = saved_history(densities)

    return history, densities, k_sorted, order



def solve_matter_power_spectrum_batch(
    k_values,
    cosmologies,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
) -> np.ndarray:
    """Return ``P(k)`` for a batch of cosmologies in one GPU solve.

    All cosmologies must share a perturbation layout. The ``n_cosmologies * n_k``
    trajectories are integrated by a single modax Rodas5P launch, so the whole
    batch amortizes one kernel compilation and one set of device buffers.

    Returns
    -------
    np.ndarray
        ``P(k)`` in Mpc^3, shape ``(n_cosmologies, n_k)``, in the caller's
        original cosmology and ``k`` order.
    """

    prepared = _prepare_solve(tuple(cosmologies))
    cosmologies = prepared.cosmologies
    layout = prepared.layout
    n_cosmo = len(cosmologies)

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values)
    n_k = len(k_sorted)
    sol = _launch(prepared, params, y0, t_span, rtol=rtol, atol=atol, first_step=first_step, max_steps=max_steps)
    y_final = np.asarray(sol)[:, -1, :].reshape(n_k, n_cosmo, layout.nvar)

    pk_sorted = np.empty((n_cosmo, n_k), dtype=np.float64)
    for ci, cosmology in enumerate(cosmologies):
        densities = density_coefficients(cosmology)
        grhoc_val, grhob_val = densities[2], densities[3]
        for ki in range(n_k):
            yv = y_final[ki, ci]
            delta_m = total_matter_density_contrast(
                grhoc_val, yv[IX_CLXC], grhob_val, yv[IX_CLXB]
            )
            pk_sorted[ci, ki] = matter_power_spectrum(
                float(k_sorted[ki]), delta_m, cosmology.A_s, cosmology.n_s, cosmology.k_pivot
            )

    pk = np.empty_like(pk_sorted)
    pk[:, order] = pk_sorted
    return pk
