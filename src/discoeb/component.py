"""A right-hand side assembled from components that declare what they read and write.

A model is built from *components*. Each declares the constant parameters it
reads (``params``) and the derivatives it contributes to (``writes``), and
provides a numba-CUDA ``rhs(y, t, p, H)``. Inside it, ``y`` and ``p`` are
namedtuples over every registered name (``y.vb``, ``p.grhob``), and the
function returns ``Y(name=value, ...)`` over exactly the names in ``writes``.

There are two kinds of component:

* a :class:`Species` owns state variables and contributes a density ``grho`` to
  the Friedmann equation -- cold dark matter, baryons, photons;
* an :class:`Interaction` owns no state and writes the derivatives of species'
  state -- Thomson scattering between the baryons and the photons.

:meth:`Model.build` returns the flat ``rhs(y, t, p) -> tuple`` an ODE kernel
integrates: every component's contribution, added by field name into the full
derivative. It generates no source. Name matching happens when numba types the
call, in the :func:`accumulate` intrinsic.

Why a component returns only the fields it writes, instead of a full-length
``Y``: something would have to zero the fields it does not name, and numba-cuda
(numba 0.66) offers no way to. It drops namedtuple ``defaults`` from the
compiled type (``Y(a=x)`` with a defaulted ``delta`` compiles to
``Y(float64 x 1)``), and device functions take neither keyword arguments nor
omitted defaults. Keyword construction of a namedtuple *class* is typed directly
and does work, so each component's ``Y`` is a namedtuple of its ``writes``.

Run this file to check the example model on the GPU against the same equations
evaluated in NumPy.
"""

import collections
import math
import types as pytypes

import numpy as np
from llvmlite import ir
from numba import cuda
from numba.core import types
from numba.core.errors import TypingError
from numba.extending import intrinsic


class Y:
    """Placeholder for a component's derivative type ``Y(name=value, ...)``.

    Components refer to it inside ``rhs``; :meth:`Model.build` rebinds the name
    to a namedtuple over that component's ``writes``.
    """


def device(f):
    return cuda.jit(device=True, inline=True)(f)


def rebind(f, **names):
    """A copy of ``f`` whose globals also hold ``names`` (numba reads ``__globals__``)."""
    return pytypes.FunctionType(
        f.__code__,
        {**f.__globals__, **names},
        f.__name__,
        f.__defaults__,
        f.__closure__,
    )


# --- intrinsics -----------------------------------------------------------------
# Namedtuples of float64 are flat LLVM aggregates, so these only extract, insert
# and add values; nothing touches memory.

# Generate a list of zeros
@intrinsic
def zeros_like(typingctx, y):
    """A namedtuple of ``y``'s type filled with ``-0.0``.

    ``-0.0`` rather than ``0.0`` because ``-0.0 + x == x`` exactly, so the first
    :func:`accumulate` into it folds away; ``0.0 + x`` does not for ``x = -0.0``. it gives ``+0.0``.
    """

    def codegen(context, builder, sig, args):
        value = ir.Constant(context.get_value_type(sig.return_type), ir.Undefined)
        for i, ty in enumerate(sig.return_type):
            value = builder.insert_value(value, context.get_constant(ty, -0.0), i)
        return value

    return y(y), codegen

# Accumulate multiple items from namedtuple to a given true tuple
@intrinsic
def accumulate(typingctx, total, part):
    """``total`` with each field of the namedtuple ``part`` added to the same-named field."""
    if not (
        isinstance(total, types.BaseNamedTuple)
        and isinstance(part, types.BaseNamedTuple)
    ):
        raise TypingError("accumulate takes two namedtuples")
    unknown = [name for name in part.fields if name not in total.fields]
    if unknown:
        raise TypingError(
            f"{part.instance_class.__name__} writes unknown state {unknown}"
        )
    slots = [(total.fields.index(name), j) for j, name in enumerate(part.fields)]

    def codegen(context, builder, sig, args):
        acc, contribution = args
        for i, j in slots:
            term = builder.extract_value(contribution, j)
            term = context.cast(builder, term, part[j], total[i])
            acc = builder.insert_value(
                acc, builder.fadd(builder.extract_value(acc, i), term), i
            )
        return acc

    return total(total, part), codegen

# Convert namedtuple into true tuple
@intrinsic
def as_tuple(typingctx, y):
    """The fields of a homogeneous namedtuple as a plain tuple -- the same bits."""
    return types.UniTuple(y.dtype, y.count)(
        y
    ), lambda context, builder, sig, args: args[0]


# --- components -----------------------------------------------------------------


class Component:
    """Anything that reads ``params`` and contributes to the derivatives in ``writes``."""

    params = ()
    writes = ()

    def __init__(self, model):
        model.register(self)

    @property
    def name(self):
        return type(self).__name__

    @staticmethod
    def rhs(y, t, p, H):
        """``Y(name=value, ...)`` over exactly the names in ``writes``."""
        raise NotImplementedError


class Species(Component):
    """A component that owns state variables and has a density.

    ``writes`` defaults to the species' own ``state``. A species that also
    writes another species' derivatives lists both; its own state must be among
    them.
    """

    state = ()
    writes = None

    def __init__(self, model):
        if not self.state:
            raise TypeError(
                f"{self.name} is a Species but owns no state; make it an Interaction"
            )
        if self.writes is None:
            self.writes = self.state
        missing = [name for name in self.state if name not in self.writes]
        if missing:
            raise TypeError(f"{self.name} owns {missing} but does not write them")
        super().__init__(model)

    @staticmethod
    def grho(y, p):
        """This species' ``8 pi G rho a^2``, summed into the Friedmann equation."""
        return 0.0


class Interaction(Component):
    """A component that owns no state and writes the derivatives of species' state."""

    def __init__(self, model):
        if getattr(self, "state", ()):
            raise TypeError(f"{self.name} is an Interaction and cannot own state")
        if not self.writes:
            raise TypeError(f"{self.name} writes nothing")
        super().__init__(model)




# --- model ----------------------------------------------------------------------

ScaleFactorRate = collections.namedtuple("ScaleFactorRate", "a")


class Model:
    """Registers components, then builds the flat right-hand side from them.

    The model itself owns the scale factor ``a`` and evolves it by the Friedmann
    equation, with the density summed over the species.
    """

    own_state = ("a",)

    def __init__(self):
        self.param_names = []
        self.state_names = list(self.own_state)
        self.owner = {name: self for name in self.own_state}
        self.components = []

    def register(self, component):
        for name in component.params:
            if name not in self.param_names:  # components may share a parameter
                self.param_names.append(name)
        for name in getattr(component, "state", ()):
            if name in self.owner:  # but each state variable has one owner
                raise ValueError(
                    f"{component.name} registers {name!r}, already owned by "
                    f"{getattr(self.owner[name], 'name', 'the model')}"
                )
            self.owner[name] = component
            self.state_names.append(name)
        self.components.append(component)

    @property
    def species(self):
        return [c for c in self.components if isinstance(c, Species)]

    def check(self):
        """Fail before compiling if a component writes state that no species owns."""
        if not self.species:
            raise ValueError("a model needs at least one species")
        for c in self.components:
            for name in c.writes:
                if name not in self.owner:
                    raise ValueError(f"{c.name} writes {name!r}, which no species owns")
                if self.owner[name] is self:
                    raise ValueError(
                        f"{c.name} writes {name!r}, which the model evolves itself"
                    )

    def build(self):
        """Return the flat ``rhs(y, t, p) -> tuple`` an ODE kernel integrates."""
        self.check()
        Yfull = self.Y = collections.namedtuple("Y", self.state_names)
        P = self.P = collections.namedtuple("P", self.param_names)

        grho = fold_sum([device(s.grho) for s in self.species])
        contributions = fold_accumulate(
            [
                device(rebind(c.rhs, Y=collections.namedtuple(f"d{c.name}", c.writes)))
                for c in self.components
            ]
        )

        @device
        def rhs(y_flat, t, p_flat):
            y, p = Yfull(*y_flat), P(*p_flat)
            H = math.sqrt(grho(y, p) / 3.0) / y.a  # a'/a^2 in conformal time
            dy = contributions(y, t, p, H)
            return as_tuple(accumulate(dy, ScaleFactorRate(a=y.a * y.a * H)))

        return rhs

    def pack_params(self, **values):
        return self._pack(self.param_names, values, "parameter")

    def pack_state(self, **values):
        return self._pack(self.state_names, values, "state variable")

    @staticmethod
    def _pack(names, values, kind):
        missing = [n for n in names if n not in values]
        unknown = [n for n in values if n not in names]
        if missing or unknown:
            raise KeyError(f"{kind}s missing: {missing}, {kind}s unknown: {unknown}")
        return np.array([values[n] for n in names], dtype=np.float64)


def fold_sum(fns):
    """One device function returning the sum of ``fns(y, p)``, unrolled at build time."""
    if len(fns) == 1:
        return fns[0]
    rest, last = fold_sum(fns[:-1]), fns[-1]

    @device
    def total(y, p):
        return rest(y, p) + last(y, p)

    return total


def fold_accumulate(fns):
    """One device function adding every ``fns(y, t, p, H)`` into a full-length ``Y``.

    A Python loop over device functions of different types does not compile, so
    the loop over components is unrolled into nested closures here.
    """
    if len(fns) == 1:
        (only,) = fns

        @device
        def first(y, t, p, H):
            return accumulate(zeros_like(y), only(y, t, p, H))

        return first
    rest, last = fold_accumulate(fns[:-1]), fns[-1]

    @device
    def total(y, t, p, H):
        return accumulate(rest(y, t, p, H), last(y, t, p, H))

    return total

