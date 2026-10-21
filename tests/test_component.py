"""The component framework: dependency injection by name, states, settings, errors.

All host-side: the providers are left as plain Python and the device names
``cuda`` and ``float64`` are NumPy stand-ins. The GPU tests compile the same
components with numba.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from discoeb.component import (
    Component,
    Interaction,
    Model,
    Setting,
    Species,
    State,
    cuda,
    float64,
    load_setting,
)

HOST = dict(
    jit=lambda f: f,
    cuda=SimpleNamespace(local=SimpleNamespace(array=lambda shape, dtype: np.zeros(shape))),
    float64=np.float64,
)
FLOOR = 1e-30  # a module constant a provider reads


class Background(Component):
    def inv_a(a):
        return 1.0 / a

    def adotoa(grho):
        return math.sqrt(grho / 3.0)


class ColdDarkMatter(Species):
    clxc = State(initial=lambda clxg: 0.75 * clxg)

    def grho(grhoc, inv_a):
        return grhoc * inv_a

    def d_clxc(clxc, tau):
        return clxc / tau


class Baryons(Species):
    clxb = State(initial=lambda clxg: 0.75 * clxg)
    vb = State()

    def grho(grhob, inv_a):
        return grhob * inv_a

    def d_clxb(vb):
        return -vb

    def d_vb(a, adotoa, vb, cs2b, k, clxb):
        return -a * adotoa * vb + cs2b * k * k * clxb


class Photons(Species):
    clxg = State(initial=lambda k, tau: (k * tau) ** 2 / 3.0)
    qg = State()

    def grho(grhog, inv_a):
        return grhog * inv_a * inv_a

    def d_clxg(qg):
        return -4.0 / 3.0 * qg

    def d_qg(k, clxg):
        return 0.25 * k * k * clxg


class ThomsonScattering(Interaction):
    def R(grhog, grhob, a):
        return 4.0 * grhog / (3.0 * grhob * a)

    def kappa(opacity):
        return max(opacity, FLOOR)

    def d_vb(R, kappa, qg, vb):
        return R * kappa * (qg - vb)

    def d_qg(kappa, qg, vb):
        return -kappa * (qg - vb)


PARAMS = {"grhoc": 0, "grhob": 1, "grhog": 2, "cs2b": 3, "k": 4, "opacity": 5, "a": 6}
Y0 = (0.3, 0.2, 0.1, 0.4, 0.05)
TAU = 2.0
P = (1.0, 0.5, 2.0, 0.01, 0.3, 7.0, 0.1)


def reference(y, tau, p):
    clxc, clxb, vb, clxg, qg = y
    grhoc, grhob, grhog, cs2b, k, opacity, a = p
    H = math.sqrt((grhoc / a + grhob / a + grhog / a**2) / 3.0)
    R = 4.0 * grhog / (3.0 * grhob * a)
    return (
        clxc / tau,
        -vb,
        -a * H * vb + cs2b * k * k * clxb + R * opacity * (qg - vb),
        -4.0 / 3.0 * qg,
        0.25 * k * k * clxg - opacity * (qg - vb),
    )


def toy_model(*extra):
    return Model(
        [Background(), ColdDarkMatter(), Baryons(), Photons(), ThomsonScattering(), *extra],
        sums=("grho",),
    )


def test_the_built_rhs_evaluates_the_equations():
    model = toy_model()
    rhs = model.build(PARAMS, **HOST)
    assert model.state == ["clxc", "clxb", "vb", "clxg", "qg"]
    np.testing.assert_allclose(rhs(Y0, TAU, P), reference(Y0, TAU, P), rtol=1e-15, atol=0)
    assert rhs._generated_source == model.source


def test_the_driver_injects_by_name_in_dependency_order():
    """Baryons and Thomson scattering both write ``vb``; Photons and Thomson ``qg``."""
    model = toy_model()
    model.build(PARAMS, **HOST)
    lines = [l.strip() for l in model.source.splitlines()]
    assert "d_vb = Baryons__d_vb(a, adotoa, vb, cs2b, k, clxb) + ThomsonScattering__d_vb(R, kappa, qg, vb)" in lines
    assert "d_qg = Photons__d_qg(k, clxg) + ThomsonScattering__d_qg(kappa, qg, vb)" in lines
    total = "grho = ColdDarkMatter__grho(grhoc, inv_a) + Baryons__grho(grhob, inv_a) + Photons__grho(grhog, inv_a)"
    assert total in lines
    assert lines.index("inv_a = Background__inv_a(a)") < lines.index(total) < lines.index("adotoa = Background__adotoa(grho)")
    # Every y and p access is a literal index.
    for line in lines:
        if "y[" in line or "p[" in line:
            assert line.split("[")[1].split("]")[0].replace(" + i", "").isdigit()


def test_only_what_the_derivatives_need_is_compiled():
    class Unused(Component):
        def never_read(a):
            return a * a

    compiled = []
    model = toy_model(Unused())
    model.build(PARAMS, **{**HOST, "jit": lambda f: compiled.append(f.__name__) or f})
    assert "never_read" not in compiled and "never_read" not in model.source
    assert "inv_a" in compiled


def test_initial_conditions_are_declared_on_the_states():
    """``clxc`` and ``clxb`` start at 3/4 of the photon density, which starts at (k tau)^2 / 3."""
    model = toy_model()
    y0 = model.initial_state([0.1, 0.2], tau=2.0)
    clxg = (np.array([0.1, 0.2]) * 2.0) ** 2 / 3.0
    np.testing.assert_allclose(y0[:, model.index["clxg"]], clxg)
    np.testing.assert_allclose(y0[:, model.index["clxc"]], 0.75 * clxg)
    np.testing.assert_allclose(y0[:, model.index["clxb"]], 0.75 * clxg)
    assert (y0[:, [model.index["vb"], model.index["qg"]]] == 0.0).all()  # no initial condition: zero


def stream(kv, x, inv_tau, output):
    """A helper shared between hierarchies, compiled for the device by the framework."""
    lmax = len(x) - 1
    for l in range(1, lmax):
        output[l] = kv * (l * x[l - 1] - (l + 1) * x[l + 1]) * (1.0 / (2 * l + 1))
    output[lmax] = kv * x[lmax - 1] - (lmax + 1) * x[lmax] * inv_tau


class Hierarchy(Species):
    LMAX = Setting(3)

    def _x_initial(k, output):
        output[0] = k
        output[1] = -k

    x = State(lambda self: self.LMAX + 1, tail_from=1, initial=_x_initial)

    def inv_tau(tau):
        return 1.0 / tau

    def d_x(k, x, inv_tau, output):
        output[0] = -k * x[1]
        stream(k, x, inv_tau, output)


def test_vector_states_are_arrays_and_their_output_buffers_are_provided():
    compiled = []
    model = Model([Hierarchy()])
    rhs = model.build({"k": 0}, **{**HOST, "jit": lambda f: compiled.append(f.__name__) or f})
    assert model.state == ["x_0", "x_1", "x_2", "x_3"]
    k, tau, x = 0.7, 3.0, (1.0, 2.0, 4.0, 8.0)
    np.testing.assert_allclose(
        rhs(x, tau, (k,)),
        (-k * 2.0, k * (1.0 - 8.0) / 3.0, k * (4.0 - 24.0) / 5.0, k * 4.0 - 32.0 / tau),
    )
    # The helper the provider calls is compiled too, before the provider.
    assert compiled.index("stream") < compiled.index("d_x")
    lines = [l.strip() for l in model.source.splitlines()]
    assert "c_Hierarchy__d_x = cuda.local.array(4, float64)" in lines
    assert "Hierarchy__d_x(k, x, inv_tau, c_Hierarchy__d_x)" in lines
    assert "d_x[i] += c_Hierarchy__d_x[i]" in lines
    assert "d_x[i] = -0.0" in lines
    # The vector's initial condition fills an output too.
    np.testing.assert_array_equal(model.initial_state([2.0]), [[2.0, -2.0, 0.0, 0.0]])


def test_sparsity_roles_give_the_jacobian_pattern():
    """x_0 is core; x_1..x_3 a tridiagonal tail hanging off x_0."""
    model = Model([Hierarchy()])
    entries = {tuple(r) for r in model.sparsity()}
    assert entries == {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)}

    class Core(Species):
        v = State(2)  # no tail: densely coupled to everything else in the core

        def d_v(k, v, output):
            output[0] = k * v[1]

    entries = {tuple(r) for r in Model([Core(), Hierarchy()]).sparsity()}
    core = {0, 1, 2}  # v_0, v_1, x_0
    assert all((r, c) in entries for r in core for c in core)
    assert (0, 4) not in entries and (3, 4) in entries and (2, 3) in entries


def test_settings_are_class_attributes_loaded_from_the_environment(monkeypatch):
    monkeypatch.setenv("DISCO_LMAX_T", "7")
    monkeypatch.setenv("DISCO_SCALE", "2.5")
    monkeypatch.setenv("DISCO_FLAG", "yes")

    class Tuned(Component):
        LMAX_T = Setting(3)
        SCALE = Setting(1.0)
        FLAG = Setting(False)
        UNSET = Setting(4)

    assert (Tuned.LMAX_T, Tuned.SCALE, Tuned.FLAG, Tuned.UNSET) == (7, 2.5, True, 4)
    assert Tuned.setting_names() == ("LMAX_T", "SCALE", "FLAG", "UNSET")
    instance = Tuned(LMAX_T=11)
    assert (instance.LMAX_T, instance.SCALE, Tuned.LMAX_T) == (11, 2.5, 7)
    assert instance.constants() == {"LMAX_T": 11, "SCALE": 2.5, "FLAG": True, "UNSET": 4}
    with pytest.raises(TypeError, match="has no settings \\['NOPE'\\]"):
        Tuned(NOPE=1)

    class Derived(Tuned):
        EXTRA = Setting("x")

    assert Derived.setting_names() == ("LMAX_T", "SCALE", "FLAG", "UNSET", "EXTRA")
    assert Model([Hierarchy()]).nvar == 4 and Model([Hierarchy(LMAX=6)]).nvar == 7

    monkeypatch.setenv("DISCO_LMAX_T", "seven")
    with pytest.raises(ValueError, match="not a int"):
        load_setting("LMAX_T", 3)
    with pytest.raises(ValueError, match="UPPER_CASE"):

        class Lower(Component):
            lmax = Setting(3)

    with pytest.raises(TypeError, match="bool, int, float or str"):
        Setting([1, 2])


def test_device_names_are_replaced_when_a_provider_is_compiled():
    """``cuda`` and ``float64`` are placeholders until then; constants appear by name."""
    with pytest.raises(RuntimeError, match="only available inside a provider"):
        cuda.local.array(3, float64)

    class Scratch(Species):
        N = Setting(3)
        v = State(lambda self: self.N)

        def d_v(k, v, output):
            tmp = cuda.local.array(N, float64)  # noqa: F821  (N is the component's setting)
            for i in range(N):  # noqa: F821
                tmp[i] = k * v[i]
            for i in range(N):  # noqa: F821
                output[i] = -tmp[i]

    seen = {}

    def jit(f):
        seen[f.__name__] = f.__globals__
        return f

    rhs = Model([Scratch()]).build({"k": 0}, **{**HOST, "jit": jit})
    np.testing.assert_allclose(rhs((1.0, 2.0, 3.0), 0.0, (2.0,)), (-2.0, -4.0, -6.0))
    assert seen["d_v"]["cuda"] is HOST["cuda"] and seen["d_v"]["N"] == 3
    # The module's own globals are untouched.
    assert Scratch.d_v.__globals__["cuda"] is cuda


def test_partial_writes_of_a_vector_accumulate():
    """Each writer gets a zeroed buffer; entries it leaves alone contribute nothing."""

    class Damping(Interaction):
        def d_x(opacity, x, output):
            for l in range(2, len(x)):
                output[l] = -opacity * x[l]

    model = Model([Hierarchy(), Damping()])
    rhs = model.build({"k": 0, "opacity": 1}, **HOST)
    k, tau, x, op = 0.7, 3.0, (1.0, 2.0, 4.0, 8.0), 5.0
    undamped = Model([Hierarchy()]).build({"k": 0}, **HOST)(x, tau, (k,))
    np.testing.assert_allclose(rhs(x, tau, (k, op)), np.add(undamped, (0.0, 0.0, -op * 4.0, -op * 8.0)))
    lines = [l.strip() for l in model.source.splitlines()]
    assert lines.index("Hierarchy__d_x(k, x, inv_tau, c_Hierarchy__d_x)") < lines.index(
        "Damping__d_x(opacity, x, c_Damping__d_x)"
    )


def test_matrix_states_and_tuple_constants():
    """A component written once for all bins; its weights are a tuple constant."""

    class Bins(Species):
        NQ = Setting(2)

        def _psi_initial(k, output):
            for i in range(NQ):  # noqa: F821
                output[i, 0] = k * weight[i]  # noqa: F821

        psi = State(lambda self: (self.NQ, 2), tail_from=1, initial=_psi_initial)

        def __init__(self, weights, **settings):
            super().__init__(**settings)
            self.weight = tuple(weights)

        def grho(a, psi):
            total = 0.0
            for i in range(NQ):  # noqa: F821
                total += weight[i] * psi[i, 0] * (a * weight[i])  # noqa: F821
            return total

        def d_psi(k, a, psi, output):
            for i in range(NQ):  # noqa: F821
                kv = k * a * weight[i]  # noqa: F821
                output[i, 0] = -kv * psi[i, 1]
                output[i, 1] = kv * psi[i, 0]

    class Reads(Species):
        w = State()

        def d_w(grho):
            return grho

    model = Model([Bins([0.5, 0.25]), Reads()], sums=("grho",))
    rhs = model.build({"k": 0, "a": 1}, **HOST)
    assert model.state == ["psi_0_0", "psi_0_1", "psi_1_0", "psi_1_1", "w"]
    k, a, y = 3.0, 2.0, (1.0, 2.0, 4.0, 8.0, 0.0)
    v0, v1 = a * 0.5, a * 0.25
    np.testing.assert_allclose(
        rhs(y, 1.0, (k, a)),
        (-k * v0 * 2.0, k * v0 * 1.0, -k * v1 * 8.0, k * v1 * 4.0, 0.5 * 1.0 * v0 + 0.25 * 4.0 * v1),
    )
    assert "psi = cuda.local.array((2, 2), float64)" in model.source
    assert "d_psi[i, j] += c_Bins__d_psi[i, j]" in model.source
    # Matrix initial conditions and sparsity work row by row.
    np.testing.assert_array_equal(model.initial_state([2.0]), [[1.0, 0.0, 0.5, 0.0, 0.0]])
    entries = {tuple(r) for r in model.sparsity()}
    assert (0, 1) in entries and (0, 3) not in entries and (1, 2) not in entries and (2, 3) in entries


def test_a_device_function_a_component_holds_is_called_through_a_closure():
    class Tables(Component):
        def __init__(self, table):
            def a(i):
                return max(table(i), 1e-30)

            self.a = a

        def root(a):
            return math.sqrt(a**2 + 1.0)

    class Reads(Species):
        w = State()

        def d_w(root):
            return root

    model = Model([Tables({0: 2.0, 1: -1.0}.get), Reads()])
    rhs = model.build({"i": 0}, **HOST)
    assert rhs((0.0,), 1.0, (0,)) == (math.sqrt(5.0),)
    assert rhs((0.0,), 1.0, (1,)) == (math.sqrt(1e-60 + 1.0),)


def test_every_state_needs_a_derivative():
    class Silent(Species):
        q = State()

    with pytest.raises(ValueError, match="derivatives of \\['q'\\]"):
        toy_model(Silent()).build(PARAMS, **HOST)


def test_an_unprovided_dependency_names_the_provider():
    with pytest.raises(NameError, match="Baryons.d_vb needs 'cs2b'"):
        toy_model().build({k: v for k, v in PARAMS.items() if k != "cs2b"}, **HOST)
    with pytest.raises(NameError, match="'tau' is neither given nor provided"):
        toy_model().initial_state([0.1])


def test_two_providers_of_one_quantity_are_rejected():
    class Other(Component):
        def inv_a(a):
            return 1.0 / a

    with pytest.raises(ValueError, match="both provide 'inv_a'"):
        toy_model(Other()).build(PARAMS, **HOST)


def test_names_that_clash_with_state_elements_are_rejected():
    class Alias(Component):
        def x_2(x):
            return x[0]

    with pytest.raises(ValueError, match="provides 'x_2', which is a state"):
        Model([Hierarchy(), Alias()]).build({"k": 0}, **HOST)

    class Scalar(Species):
        x_1 = State()

        def d_x_1(x_1):
            return x_1

    with pytest.raises(ValueError, match="clashes"):
        Model([Hierarchy(), Scalar()])


def test_output_is_only_for_vector_derivatives_and_initial_conditions():
    class Stray(Interaction):
        def d_nothing(k):
            return k

    with pytest.raises(ValueError, match="writes 'nothing'"):
        toy_model(Stray()).build(PARAMS, **HOST)

    class Returns(Interaction):
        def d_x(k, x):
            return k

    with pytest.raises(TypeError, match="take 'output' and fill it"):
        Model([Hierarchy(), Returns()]).build({"k": 0}, **HOST)

    class TakesOutput(Interaction):
        def d_vb(k, output):
            return k

    with pytest.raises(TypeError, match="return the contribution"):
        toy_model(TakesOutput()).build(PARAMS, **HOST)

    class Quantity(Interaction):
        def inv_tau(tau, output):
            return 1.0 / tau

    with pytest.raises(TypeError, match="only a vector state's derivative fills"):
        Model([Hierarchy(), Quantity()]).build({"k": 0}, **HOST)

    class BadInitial(Species):
        v = State(2, initial=lambda k: k)

        def d_v(k, v, output):
            output[0] = k

    with pytest.raises(TypeError, match="takes 'output' and fills it"):
        Model([BadInitial()]).initial_state([1.0])


def test_cycles_are_reported():
    class Loop(Component):
        def u(v):
            return v

        def v(u):
            return u

    class Reads(Species):
        w = State()

        def d_w(u):
            return u

    with pytest.raises(ValueError, match="circular dependency"):
        Model([Loop(), Reads()]).build({}, **HOST)

    class Chicken(Species):
        c = State(initial=lambda c: c)

        def d_c(c):
            return c

    with pytest.raises(ValueError, match="circular dependency on the host"):
        Model([Chicken()]).initial_state([1.0])


def test_methods_are_providers_without_staticmethod_and_take_no_self():
    assert Baryons.d_clxb(2.0) == -2.0  # a plain function on the class, no self

    class Bad(Species):
        x = State()

        def d_x(self, x):
            return x

    with pytest.raises(TypeError, match="a provider has no self"):
        Model([Bad()]).build({}, **HOST)


def test_interactions_own_no_state():
    with pytest.raises(TypeError, match="cannot own state"):

        class Bad(Interaction):
            x = State()


def test_providers_are_plain_positional_functions():
    class Defaulted(Species):
        x = State()

        def d_x(x, k=1.0):
            return k * x

    with pytest.raises(TypeError, match="plain positional parameters"):
        Model([Defaulted()]).build({"k": 0}, **HOST)
