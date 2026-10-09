from .component import Species, Interaction

# --- example components -------------------------------------------------------------
# Toy equations with the shape of the real ones, in conformal time with
# velocity divergences: pressureless CDM, baryons with a sound speed, a photon
# fluid truncated at the dipole, and Thomson scattering between the last two.


class ColdDarkMatter(Species):
    params = ("grhoc",)
    state = ("clxc",)

    @staticmethod
    def grho(y, p):
        return p.grhoc / y.a

    @staticmethod
    def rhs(y, t, p, H):
        return Y(clxc=y.clxc / t)


class Baryons(Species):
    params = ("grhob", "cs2b", "k")
    state = ("clxb", "vb")

    @staticmethod
    def grho(y, p):
        return p.grhob / y.a

    @staticmethod
    def rhs(y, t, p, H):
        return Y(
            clxb=-y.vb,
            vb=-y.a * H * y.vb + p.cs2b * p.k * p.k * y.clxb,
        )


class Photons(Species):
    params = ("grhog", "k")
    state = ("clxg", "qg")

    @staticmethod
    def grho(y, p):
        return p.grhog / (y.a * y.a)

    @staticmethod
    def rhs(y, t, p, H):
        return Y(
            clxg=-4.0 / 3.0 * y.qg,
            qg=0.25 * p.k * p.k * y.clxg,
        )


class ThomsonScattering(Interaction):
    """Momentum exchange between the baryons and the photons."""

    params = ("opacity", "grhog", "grhob")
    writes = ("vb", "qg")

    @staticmethod
    def rhs(y, t, p, H):
        R = 4.0 * p.grhog / (3.0 * p.grhob * y.a)  # photon-to-baryon momentum density
        slip = y.qg - y.vb
        return Y(
            vb=R * p.opacity * slip,
            qg=-p.opacity * slip,
        )


