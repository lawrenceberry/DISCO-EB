"""Turn an out-array right-hand side into the tuple form Enzyme can read.

modax derives the Jacobian by forward-differentiating ``ode_fn`` with Enzyme,
which requires the callback to take and return fixed-size tuples of scalars.
The perturbation right-hand side is not written that way and cannot simply be
rewritten: its multipole hierarchies index ``y`` with a loop variable, and a
runtime index into a tuple makes numba emit a bounds-check ``cmpxchg`` that
Enzyme refuses ("cannot handle unknown instruction"). Wrapping the loops around
a thread-local array instead gets past typing but produces IR that NVVM will not
verify.

What is left is to remove the loops, and the safe way to do that is mechanically
rather than by hand: :func:`unroll_to_tuple` parses the *existing* right-hand
side, unrolls every loop whose bounds are compile-time constants, folds the
resulting index expressions to integer literals, and returns the components as a
tuple. The physics is never restated, so the generated callback cannot drift
from the one it came from -- and ``tests`` check the two agree numerically.

The transformation is deliberately narrow. It constant-folds only what it can
evaluate from the function's own globals and closure; anything touching ``y``,
``p`` or ``tau`` fails to evaluate and is left exactly as written.
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap


def _static_namespace(fn) -> dict:
    """Everything the function can see that is fixed at build time."""
    namespace = dict(fn.__globals__)
    if fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                namespace[name] = cell.cell_contents
            except ValueError:  # an unset cell, e.g. recursion
                pass
    return namespace


class _Folder(ast.NodeTransformer):
    """Substitute known names and evaluate what that makes constant."""

    def __init__(self, env: dict):
        self.env = env

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.env:
            return ast.copy_location(ast.Constant(self.env[node.id]), node)
        return node


class _Unroller(ast.NodeTransformer):
    """Unroll constant loops, prune constant branches, fold indices."""

    def __init__(
        self,
        namespace: dict,
        out_name: str,
        state_name: str,
        n_vars: int,
        foldable: frozenset[str],
    ):
        self.namespace = namespace
        self.out_name = out_name
        self.state_name = state_name
        self.n_vars = n_vars
        # Names assigned exactly once may become build-time constants. One
        # assigned twice may not: `opacity = spline_eval(...)` followed by a
        # clamp `opacity = 1e-30` is a run-time value with a floor, and folding
        # the floor would pin it to that floor for good.
        self.foldable = foldable
        self.env: dict = {}

    # --- evaluation ----------------------------------------------------------
    def _eval(self, node: ast.expr):
        """Value of ``node`` if it is fixed at build time, else ``_UNKNOWN``."""
        folded = _Folder(self.env).visit(ast.fix_missing_locations(_clone(node)))
        try:
            return eval(  # noqa: S307 - the source is this package's own
                compile(ast.Expression(folded), "<unroll>", "eval"),
                self.namespace,
                {},
            )
        except Exception:
            return _UNKNOWN

    def _index_of(self, node: ast.expr) -> int:
        value = self._eval(node)
        if value is _UNKNOWN or int(value) != value:
            raise ValueError(
                f"index {ast.unparse(node)!r} is not a build-time constant; the "
                "right-hand side cannot be unrolled into tuple form"
            )
        return int(value)

    # --- statements ----------------------------------------------------------
    def visit_For(self, node: ast.For):
        call = node.iter
        if not (isinstance(call, ast.Call) and getattr(call.func, "id", "") == "range"):
            raise ValueError(f"cannot unroll a loop over {ast.unparse(node.iter)!r}")
        bounds = [self._eval(a) for a in call.args]
        if any(b is _UNKNOWN for b in bounds):
            raise ValueError(f"loop bounds {ast.unparse(node.iter)!r} are not constant")
        if not isinstance(node.target, ast.Name):
            raise ValueError("cannot unroll a loop with a tuple target")

        name = node.target.id
        outer = self.env.get(name, _UNKNOWN)
        body: list[ast.stmt] = []
        for value in range(*(int(b) for b in bounds)):
            self.env[name] = value
            for stmt in node.body:
                body.extend(_as_list(self.visit(_clone(stmt))))
        if outer is _UNKNOWN:
            self.env.pop(name, None)
        else:
            self.env[name] = outer
        return body

    def visit_If(self, node: ast.If):
        test = self._eval(node.test)
        if test is _UNKNOWN:  # a genuine run-time branch, e.g. a floor on opacity
            node = self.generic_visit(node)
            node.body = node.body or [ast.Pass()]
            return ast.fix_missing_locations(node)
        branch = node.body if test else node.orelse
        body: list[ast.stmt] = []
        for stmt in branch:
            body.extend(_as_list(self.visit(_clone(stmt))))
        return body

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1:
            target = node.targets[0]
            # out[expr] = value  ->  _o<i> = value
            if _is_subscript_of(target, self.out_name):
                index = self._index_of(target.slice)
                node.value = self.visit(node.value)
                node.targets = [ast.Name(id=_slot(index), ctx=ast.Store())]
                return ast.fix_missing_locations(node)
            # A scalar local that is itself constant becomes part of the env and
            # its statement disappears; anything depending on y/p/tau does not
            # evaluate and survives untouched.
            if isinstance(target, ast.Name) and target.id in self.foldable:
                value = self._eval(node.value)
                if value is not _UNKNOWN and isinstance(value, (int, float, bool)):
                    self.env[target.id] = value
                    return []
        node.value = self.visit(node.value)
        return node

    def visit_AugAssign(self, node: ast.AugAssign):
        if _is_subscript_of(node.target, self.out_name):
            index = self._index_of(node.target.slice)
            node.value = self.visit(node.value)
            node.target = ast.Name(id=_slot(index), ctx=ast.Store())
            return ast.fix_missing_locations(node)
        return self.generic_visit(node)

    # --- expressions ---------------------------------------------------------
    def visit_Subscript(self, node: ast.Subscript):
        node.value = self.visit(node.value)
        # y[...] must become a literal index: it is a tuple in the compiled form.
        if _is_name(node.value, self.state_name) and isinstance(node.ctx, ast.Load):
            node.slice = ast.Constant(self._index_of(node.slice))
            return ast.fix_missing_locations(node)
        node.slice = self.visit(node.slice)
        return node

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.env:
            return ast.copy_location(ast.Constant(self.env[node.id]), node)
        return node


class _Unknown:
    def __repr__(self):
        return "<unknown>"


_UNKNOWN = _Unknown()


def _slot(index: int) -> str:
    return f"_o{index}"


def _clone(node):
    """A deep copy, so each unrolled iteration rewrites its own nodes."""
    return copy.deepcopy(node)


def _as_list(node):
    if node is None:
        return []
    return node if isinstance(node, list) else [node]


def _is_name(node, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_subscript_of(node, name: str) -> bool:
    return isinstance(node, ast.Subscript) and _is_name(node.value, name)


def _single_assignment_names(func: ast.FunctionDef) -> frozenset[str]:
    """Plain locals written exactly once, and so safe to treat as constants."""
    counts: dict[str, int] = {}
    for node in ast.walk(func):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.For)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                counts[target.id] = counts.get(target.id, 0) + 1
    return frozenset(n for n, c in counts.items() if c == 1)


def unroll_to_tuple(fn, n_vars: int, *, name: str = "ode_tuple"):
    """Rewrite ``fn(y, t, p, out)`` as ``name(y, t, p)`` returning a tuple.

    Every loop is unrolled and every ``y``/``out`` index folded to a literal, so
    the result indexes only with constants -- which is what makes it safe to
    compile against a tuple signature and hand to Enzyme.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef), "expected a plain function"
    func.decorator_list = []
    args = [a.arg for a in func.args.args]
    if len(args) != 4:
        raise ValueError(f"expected fn(y, t, p, out), got {args}")
    state_name, out_name = args[0], args[3]

    namespace = _static_namespace(fn)
    unroller = _Unroller(
        namespace, out_name, state_name, n_vars, _single_assignment_names(func)
    )
    body: list[ast.stmt] = []
    for stmt in func.body:
        body.extend(_as_list(unroller.visit(_clone(stmt))))

    assigned = {
        t.id
        for stmt in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(stmt, (ast.Assign, ast.AugAssign))
        for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
        if isinstance(t, ast.Name)
    }
    missing = [i for i in range(n_vars) if _slot(i) not in assigned]
    if missing:
        raise ValueError(f"components {missing} are never assigned by {fn.__name__}")

    body.append(
        ast.Return(
            value=ast.Tuple(
                elts=[ast.Name(id=_slot(i), ctx=ast.Load()) for i in range(n_vars)],
                ctx=ast.Load(),
            )
        )
    )
    func.body = body
    func.name = name
    func.args.args = func.args.args[:3]
    ast.fix_missing_locations(func)

    source = ast.unparse(func)
    scope: dict = {}
    exec(compile(source, f"<unrolled {fn.__name__}>", "exec"), namespace, scope)  # noqa: S102
    generated = scope[name]
    generated._unrolled_source = source
    return generated
