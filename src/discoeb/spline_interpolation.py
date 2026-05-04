import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class spline_interpolation(object):
    def __init__(self, xin: jnp.ndarray, yin: jnp.ndarray, integrate_from_start: bool = True, uniform: bool = False):
        # def spline_interpolation(x: jnp.ndarray, y: jnp.ndarray, integrate_from_start: bool = True):
        """
        Constructs a natural cubic spline interpolator and an integrator from input data x and y.
        
        The spline is built by solving a tridiagonal system for the second derivatives
        (using the Thomas algorithm) under natural boundary conditions (S[0] = S[-1] = 0).
        
        In addition to returning an interpolator function that evaluates the spline at new x values,
        this function returns an integrator function that computes the definite integral of the spline.
        The integrator takes an extra flag:
        - from_start=True returns the integral from x[0] to x_new.
        - from_start=False returns the integral from x_new to x[-1].
        
        Args:
            x: 2D array of x-coordinates (assumed sorted in increasing order).
            y: 2D array of y-coordinates (1st same as x, 2nd=batch)
        """
        # If xin or yin is passed as 1d, promote to 2d with dummy index
        xin = xin.reshape(xin.shape[0], -1)
        yin = yin.reshape(xin.shape[0], -1)

        # filter out NaNs at end of arrays
        def _fill_forward( last_observed_yi, yi, fac ):
            yi = jnp.where(jnp.isnan(yi), last_observed_yi*fac, yi)
            return yi, yi
        
        # n = jnp.sum(jnp.isnan(yin) == False)
        n = yin.shape[0]
        _, y = jax.lax.scan(lambda c,v: _fill_forward(c,v,1.0), yin[0], yin)
        _, x = jax.lax.scan(lambda c,v: _fill_forward(c,v,1.01), xin[0], xin)

        
        if n < 2:
            raise ValueError("There must be at least two data points.")
        
        # Compute intervals between knots.
        h = x[1:] - x[:-1]  # shape: (n-1,)
        
        # Compute the second derivatives S at the knots using a Thomas algorithm.
        m = n - 2  # number of interior points
        #B = jnp.maximum(y.shape[1], x.shape[1]) # number of batch dimensions
        y = y + x[0:1, :] * 0.0 # Broadcast shapes without explicitly finding out shapes
        B = y.shape[1]
        Bx = x.shape[1]
        if m > 0:
            # Build tridiagonal system for S[1] ... S[n-2]
            a = h[:-1]              # lower diagonal (length m)
            b_diag = 2 * (h[:-1] + h[1:])  # main diagonal (length m)
            c = h[1:]               # upper diagonal (length m)
            d = 6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
            
            # Allocate arrays for modified coefficients.
            cp = jnp.zeros((m, Bx))
            cp = cp.at[0].set(c[0] / b_diag[0])
            def forward_step_cp(i, cp):
              denom = b_diag[i] - a[i] * cp[i - 1]
              return cp.at[i].set(c[i] / denom)
            cp = jax.lax.fori_loop(1, m - 1, forward_step_cp, cp)

            dp = jnp.zeros((m,B))
            dp = dp.at[0].set(d[0] / b_diag[0])
            def forward_step_dp(i, dp):
                denom = b_diag[i] - a[i] * cp[i - 1]
                return dp.at[i].set((d[i] - a[i] * dp[i - 1]) / denom)
            dp = jax.lax.fori_loop(1, m, forward_step_dp, dp)

            # Backward substitution.
            S_interior = jnp.zeros((m,B))
            S_interior = S_interior.at[m - 1].set(dp[m - 1])
            
            def backward_step(i, S_int):
                idx = m - 2 - i
                S_int = S_int.at[idx].set(dp[idx] - cp[idx] * S_int[idx + 1])
                return S_int
            
            S_interior = jax.lax.fori_loop(0, m - 1, backward_step, S_interior)
            
            zeros = jnp.zeros((1,B))
            # Natural boundary conditions: S[0] = S[-1] = 0.
            S_full = jnp.concatenate([zeros, S_interior, zeros], axis=0)
        else:
            # Only two points – linear interpolation.
            S_full = jnp.zeros((2,B))

        # Precompute coefficients and integrals on each interval.
        # For interval i, the cubic polynomial is represented as:
        #   P_i(x) = a_i + b_i*(x - x[i]) + c_i*(x - x[i])^2 + d_i*(x - x[i])^3
        # where:
        #   a_i = y[i]
        #   b_i = (y[i+1]-y[i])/h[i] - (h[i]/6)*(S_full[i+1]+2*S_full[i])
        #   c_i = S_full[i] / 2
        #   d_i = (S_full[i+1]-S_full[i]) / (6*h[i])
        a_i = y[:-1]
        b_i = (y[1:] - y[:-1]) / h - h * (S_full[1:] + 2 * S_full[:-1]) / 6.0
        c_i = S_full[:-1] / 2.0
        d_i = (S_full[1:] - S_full[:-1]) / (6.0 * h)
        
        # The exact integral over the full interval [x[i], x[i+1]] is:
        #   I_full[i] = a_i*h[i] + b_i*h[i]^2/2 + c_i*h[i]^3/3 + d_i*h[i]^4/4
        I_full = a_i * h + b_i * (h ** 2) / 2 + c_i * (h ** 3) / 3 + d_i * (h ** 4) / 4
        zeros = jnp.zeros((1,B))
        # Cumulative integral from x[0] up to each knot.
        I_cum = jnp.concatenate([zeros, jnp.cumsum(I_full, axis=0)], axis=0)
        I_total = I_cum[-1, :]

        # Store the spline data.
        self._n_ = n
        self._x_, self._y_, self._integrate_from_start_ = x, y, integrate_from_start
        self._S_full_, self._I_total_, self._I_cum_ = S_full, I_total, I_cum
        self._uniform_ = uniform
        self._x0_ = x[0]
        self._dx_inv_ = 1.0 / (x[1] - x[0])

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = (self._x_, self._y_, self._integrate_from_start_, (self._S_full_, self._I_total_, self._I_cum_), self._x0_, self._dx_inv_)
        aux_data = {'n': self._n_, 'uniform': self._uniform_}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)  # Create instance without calling __init__
        obj._x_, obj._y_, obj._integrate_from_start_, (obj._S_full_, obj._I_total_, obj._I_cum_), obj._x0_, obj._dx_inv_ = children
        obj._n_ = aux_data['n']
        obj._uniform_ = aux_data.get('uniform', False)
        return obj

    def _find_index(self, x_new):
        """Find the interval index for spline evaluation.

        For uniform grids, uses O(1) direct index computation instead of searchsorted.
        """
        n = self._n_
        if self._uniform_:
            idx = jnp.clip(jnp.floor((x_new - self._x0_) * self._dx_inv_).astype(jnp.int32), 0, n - 2)
        else:
            #idx = jnp.clip(jnp.searchsorted(self._x_, x_new) - 1, 0, n - 2)
            x_new_ = jnp.atleast_2d(x_new)
            idx = jnp.clip(jnp.sum(self._x_[:, None, :] < x_new_[None, :, :], axis=0) - 1, 0, n - 2)
            idx = idx.reshape(jnp.shape(x_new))
        return idx

    def evaluate(self, x_new: jnp.ndarray):
        """
        Evaluates the natural cubic spline at new x positions.
        
        Args:
            x_new: scalar or 1D array of new x values.
            
        Returns:
            Interpolated y values.
        """
        x_new = jnp.atleast_1d(x_new)
        x_new = x_new.reshape(x_new.shape[0], -1) # Extend last dimension if not already done
        n = self._x_.shape[0]
        # Find the interval index i such that x[i] <= x_new < x[i+1]
        idx = self._find_index(x_new)
        xidx = jnp.take_along_axis(self._x_, idx, axis=0)
        xidxp1 = jnp.take_along_axis(self._x_, idx+1, axis=0)
        yidx = jnp.take_along_axis(self._y_, idx, axis=0)
        yidxp1 = jnp.take_along_axis(self._y_, idx+1, axis=0)
        Sidx = jnp.take_along_axis(self._S_full_, idx, axis=0)
        Sidxp1 = jnp.take_along_axis(self._S_full_, idx+1, axis=0)
   
        h_local = xidxp1 - xidx
        d_val = x_new - xidx  # local offset
        t = d_val / h_local     # normalized coordinate
        A = 1 - t
        B = t
        # Standard cubic spline evaluation.
        y_new = (A * yidx + B * yidxp1 +
                 ((A ** 3 - A) * Sidx + (B ** 3 - B) * Sidxp1) *
                 (h_local ** 2) / 6.0)
        return y_new #jnp.where(x_new.shape[0] == 1, y_new[0], y_new)
    
    def integral(self, x_new: jnp.ndarray):
        """
        Evaluates the definite integral of the spline.
        
        Args:
            x_new: scalar or 1D array of new x values.
            from_start: if True, returns the integral from x[0] to x_new;
                        if False, returns the integral from x_new to x[-1].
                        
        Returns:
            The definite integral values.
        """
        n = self._x_.shape[0]
        x_new = jnp.atleast_1d(x_new)
        x_new = x_new.reshape(x_new.shape[0], -1) # Extend last dimension if not already done
        # Locate the interval index for each x_new.
        idx = self._find_index(x_new)
        xidx = jnp.take_along_axis(self._x_, idx, axis=0)
        xidxp1 = jnp.take_along_axis(self._x_, idx+1, axis=0)
        yidx = jnp.take_along_axis(self._y_, idx, axis=0)
        yidxp1 = jnp.take_along_axis(self._y_, idx+1, axis=0)
        Sidx = jnp.take_along_axis(self._S_full_, idx, axis=0)
        Sidxp1 = jnp.take_along_axis(self._S_full_, idx+1, axis=0)
        Icumidx = jnp.take_along_axis(self._I_cum_, idx+1, axis=0)
        
        h_local = xidxp1 - xidx
        d_val = (x_new - xidx)
        # Compute the local coefficients for the interval.
        a_local = yidx
        b_local = (yidxp1 - yidx) / h_local - h_local * (yidxp1 + 2 * Sidx) / 6.0
        c_local = Sidx / 2.0
        d_local = (Sidxp1 - Sidx) / (6.0 * h_local)
        # Compute the partial integral over the interval from x[idx] to x_new:
        I_partial = (a_local * d_val +
                     b_local * d_val ** 2 / 2 +
                     c_local * d_val ** 3 / 3 +
                     d_local * d_val ** 4 / 4)
        I_forward = Icumidx + I_partial
        # If integrating from the start, return I_forward; otherwise, subtract from total.
        if self._integrate_from_start_:
            result = jnp.where(x_new.shape[0] == 1, I_forward[0], I_forward)
        else:
            result = jnp.where(x_new.shape[0] == 1, self._I_total_ - I_forward[0], self._I_total_ - I_forward)
        return result
    
    def derivative(self, x_new: jnp.ndarray):
        """
        Computes the derivative of the spline at new x positions.
        
        Args:
            x_new: scalar or 1D array of new x values.
            
        Returns:
            The derivative (dy/dx) evaluated at x_new.
        """
        n = self._x_.shape[0]
        x_new = jnp.atleast_1d(x_new)
        x_new = x_new.reshape(x_new.shape[0], -1) # Extend last dimension if not already done
        idx = self._find_index(x_new)
        xidx = jnp.take_along_axis(self._x_, idx, axis=0)
        xidxp1 = jnp.take_along_axis(self._x_, idx+1, axis=0)
        yidx = jnp.take_along_axis(self._y_, idx, axis=0)
        yidxp1 = jnp.take_along_axis(self._y_, idx+1, axis=0)
        Sidx = jnp.take_along_axis(self._S_full_, idx, axis=0)
        Sidxp1 = jnp.take_along_axis(self._S_full_, idx+1, axis=0)
        Icumidx = jnp.take_along_axis(self._I_cum_, idx+1, axis=0)

        h_local = xidxp1 - xidx
        d_val = x_new - xidx
        # Local coefficients (as defined in the cubic polynomial):
        b_local = (yidxp1 - yidx) / h_local - h_local * (Sidxp1 + 2 * Sidx) / 6.0
        c_local = Sidx / 2.0
        d_local = (Sidxp1 - Sidx) / (6.0 * h_local)
        # Derivative of P_i(x) = b_i + 2*c_i*(x-x_i) + 3*d_i*(x-x_i)^2
        dydx = b_local + 2 * c_local * d_val + 3 * d_local * d_val**2
        return jnp.where(x_new.shape[0] == 1, dydx[0], dydx)

    def derivative2(self, x_new: jnp.ndarray):
        """
        Computes the second derivative of the spline at new x positions.

        Args:
            x_new: scalar or 1D array of new x values.

        Returns:
            The derivative (d^2y/dx^2) evaluated at x_new.
        """
        n = self._x_.shape[0]
        x_new = jnp.atleast_1d(x_new)
        idx = self._find_index(x_new)
        h_local = self._x_[idx + 1] - self._x_[idx]
        d_val = x_new - self._x_[idx]
        # Local coefficients (as defined in the cubic polynomial):
        b_local = (self._y_[idx + 1] - self._y_[idx]) / h_local - h_local * (self._S_full_[idx + 1] + 2 * self._S_full_[idx]) / 6.0
        c_local = self._S_full_[idx] / 2.0
        d_local = (self._S_full_[idx + 1] - self._S_full_[idx]) / (6.0 * h_local)
        # 2nd derivative of P_i(x) = 2*c_i + 6*d_i*(x-x_i)
        d2ydx2 = 2 * c_local + 6 * d_local * d_val
        return jnp.where(x_new.shape[0] == 1, d2ydx2[0], d2ydx2)
    
    def derivative12(self, x_new: jnp.ndarray):
        """
        Computes both the first and the second derivative of the spline at new x positions.

        Args:
            x_new: scalar or 1D array of new x values.
            
        Returns:
            The derivatives (dy/dx),(d^2y/dx^2) evaluated at x_new.
        """
        n = self._x_.shape[0]
        x_new = jnp.atleast_1d(x_new)
        x_new = x_new.reshape(x_new.shape[0], -1) # Extend last dimension if not already done
        idx = self._find_index(x_new)
        xidx = jnp.take_along_axis(self._x_, idx, axis=0)
        xidxp1 = jnp.take_along_axis(self._x_, idx+1, axis=0)
        yidx = jnp.take_along_axis(self._y_, idx, axis=0)
        yidxp1 = jnp.take_along_axis(self._y_, idx+1, axis=0)
        Sidx = jnp.take_along_axis(self._S_full_, idx, axis=0)
        Sidxp1 = jnp.take_along_axis(self._S_full_, idx+1, axis=0)
        Icumidx = jnp.take_along_axis(self._I_cum_, idx+1, axis=0)
        h_local = xidxp1 - xidx
        d_val = x_new - xidx
        # Local coefficients (as defined in the cubic polynomial):
        b_local = (yidxp1 - yidx) / h_local - h_local * (Sidxp1 + 2 * Sidx) / 6.0
        c_local = Sidx / 2.0
        d_local = (Sidxp1 - Sidx) / (6.0 * h_local)
        # Derivative of P_i(x) = b_i + 2*c_i*(x-x_i) + 3*d_i*(x-x_i)^2
        dydx = b_local + 2 * c_local * d_val + 3 * d_local * d_val**2
        # 2nd derivative of P_i(x) = 2*c_i + 6*d_i*(x-x_i)
        d2ydx2 = 2 * c_local + 6 * d_local * d_val
        return jnp.where(x_new.shape[0] == 1, dydx[0], dydx), jnp.where(x_new.shape[0] == 1, d2ydx2[0], d2ydx2)


