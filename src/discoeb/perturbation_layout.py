"""Compile-time state-vector layout for the Einstein-Boltzmann perturbations.

The perturbation state vector is assembled from a set of hierarchies whose
presence and length depend on the cosmology. To keep the integrated system (and
the matching block-LU solver) as small as possible, the layout is *trimmed at
compile time* to only the components a given cosmology actually needs:

    * always: metric ``etak``, CDM ``clxc``, baryon ``clxb``/``vb``, the photon
      temperature hierarchy ``Theta_0..Theta_lmaxg``, the E-mode polarization
      hierarchy ``E_2..E_lmaxpol``, and the massless-neutrino hierarchy
      ``N_0..N_lmaxr``;
    * dynamical dark energy (``w_0 != -1`` or ``w_a != 0``): the DE fluid
      perturbations ``clxq``/``thetaq``;
    * massive neutrinos (``num_massive_neutrinos > 0``): ``nqmax`` momentum bins,
      each carrying a multipole hierarchy ``psi_0..psi_lmaxnu``.

Spatial curvature (``Omega_k != 0``) does **not** add state variables -- it only
modifies coefficients in the metric and free-streaming equations -- so it is
carried as the curvature scale ``K`` rather than a layout flag.

:class:`PerturbationLayout` is a frozen, hashable description of the enabled
components and their state-vector indices. It is a plain-Python, compile-time
object: the numba-CUDA right-hand side and the Schur-EB block-LU are specialized
on the layout the same way they are on the ``lmax`` truncations.

The flat-``LambdaCDM`` + massless-neutrino default reproduces the fixed layout of
:mod:`discoeb.perturbations` (``NVAR == 50``).
"""

from dataclasses import dataclass


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
        contiguous and can be treated as one tridiagonal block by the Schur-EB
        block-LU solver.
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
