"""Cosmological parameter containers for DISCO-EB."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cosmology:
    """Parameters for the general homogeneous cosmology in :mod:`discoeb.background`.

    Densities named ``omega_*_h2`` are physical densities,
    ``Omega_i h^2``. The compatibility helper :meth:`to_background_params`
    converts these into the historical DISCO-EB ``param`` dictionary where
    density fractions are stored as ``Omega_i``.
    """

    omega_b_h2: float
    """Physical baryon density, ``Omega_b h^2``."""

    omega_c_h2: float
    """Physical cold-dark-matter density, ``Omega_c h^2``."""

    h: float
    """Dimensionless Hubble parameter, ``H0 / (100 km/s/Mpc)``."""

    tau_reion: float
    """Thomson optical depth to reionization, ``tau_reio``."""

    A_s: float
    """Scalar primordial curvature amplitude at ``k_pivot``."""

    n_s: float
    """Scalar primordial spectral index, ``n_s``."""

    T_cmb: float
    """Present-day CMB temperature in Kelvin, ``T_cmb``."""

    Y_He: float
    """Primordial helium mass fraction, ``Y_He``."""

    N_eff: float
    """Total effective number of neutrino species, ``N_eff``."""

    k_pivot: float
    """Primordial power-spectrum pivot scale in ``Mpc^-1``."""

    Omegak: float
    """Present-day spatial-curvature density fraction, ``Omega_k``."""

    w_DE_0: float
    """Dark-energy equation of state today in the CPL form, ``w_0``."""

    w_DE_a: float
    """CPL dark-energy evolution parameter, ``w_a``, for ``w(a) = w_0 + w_a (1 - a)``."""

    cs2_DE: float
    """Dark-energy rest-frame sound speed squared, ``c_s^2``."""

    num_massive_neutrinos: float
    """Number of massive neutrino species represented by the degenerate-mass approximation."""

    mnu: float
    """Mass in eV of each massive neutrino species in the degenerate-mass approximation."""

    def __post_init__(self) -> None:
        """Validate parameter combinations that cannot be represented."""

        if self.num_massive_neutrinos < 0.0:
            raise ValueError("num_massive_neutrinos must be non-negative")
        if self.num_massive_neutrinos > self.N_eff:
            raise ValueError("num_massive_neutrinos cannot exceed N_eff")
        if self.mnu < 0.0:
            raise ValueError("mnu must be non-negative")

    @property
    def H0(self) -> float:
        """Hubble constant in km/s/Mpc."""

        return 100.0 * self.h

    @property
    def Omegab(self) -> float:
        """Present-day baryon density fraction, ``Omega_b``."""

        return self.omega_b_h2 / self.h**2

    @property
    def Omegac(self) -> float:
        """Present-day cold-dark-matter density fraction, ``Omega_c``."""

        return self.omega_c_h2 / self.h**2

    @property
    def Omegam(self) -> float:
        """Present-day baryon plus cold-dark-matter density fraction."""

        return self.Omegab + self.Omegac

    @property
    def Neff_massless(self) -> float:
        """Effective number of massless neutrino species used by DISCO-EB."""

        return self.N_eff - self.num_massive_neutrinos

    @property
    def summed_neutrino_mass(self) -> float:
        """Total neutrino mass in eV in the degenerate-mass approximation."""

        return self.num_massive_neutrinos * self.mnu

    def to_background_params(self) -> dict[str, float]:
        """Return a legacy ``param`` dictionary accepted by ``evolve_background``."""

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
        }
