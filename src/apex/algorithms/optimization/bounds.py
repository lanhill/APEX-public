"""Bounds of the controls: projection, the projected gradient and the unit-cube transform.

Ported from PET
---------------
PET module : src/popt/optimization_methods/optimizer_base.py (BoundTransformHandler)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Nocedal, J. & Wright, S. J. (2006). Numerical Optimization (2nd ed.), ch. 16.7
             (projected gradient). Springer.
Differences: none. A one-sided bound is accepted without ``transform`` (PET requires both
             sides finite whenever bounds are given); with ``transform`` both must be finite.
"""

import numpy as np

from apex.algorithms.optimization.protocols import FloatArray


class BoundTransform:
    """Bounds ``[lower, upper]`` and, with ``transform``, the map to the unit cube.

    With ``transform`` the optimizer works on ``u = (x - lower) / (upper - lower)``; gradients
    scale by ``upper - lower`` and Hessians by its outer product.

    Parameters
    ----------
    lower, upper
        Bounds ``(n_x,)``, or ``None`` for unbounded.
    transform
        Optimize in the unit cube.
    """

    def __init__(
        self, lower: FloatArray | None, upper: FloatArray | None, *, transform: bool = False
    ) -> None:
        self.bounded = lower is not None or upper is not None
        self.transform = transform and self.bounded
        if not self.bounded:
            self.lower = self.upper = self.width = np.empty(0)
            return
        size = (lower if lower is not None else upper).size  # type: ignore[union-attr]
        self.lower = np.full(size, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        self.upper = np.full(size, np.inf) if upper is None else np.asarray(upper, dtype=float)
        if np.any(self.upper <= self.lower):
            raise ValueError("every upper bound must be strictly greater than its lower bound")
        if transform and not (np.all(np.isfinite(self.lower)) and np.all(np.isfinite(self.upper))):
            raise ValueError("transform needs finite lower and upper bounds for every control")
        self.width = self.upper - self.lower

    def to_internal(self, x: FloatArray) -> FloatArray:
        """Map physical controls to the optimizer's coordinates (``(n_x,)`` or ``(n_x, n)``)."""
        if not self.transform:
            return x
        if np.any(x < self.lower) or np.any(x > self.upper):
            raise ValueError("the controls are outside their bounds")
        return np.asarray((x - self._column(self.lower, x)) / self._column(self.width, x))

    def to_physical(self, u: FloatArray) -> FloatArray:
        """Map the optimizer's coordinates to physical controls (``(n_x,)`` or ``(n_x, n)``)."""
        if not self.transform:
            return u
        return np.asarray(self._column(self.lower, u) + u * self._column(self.width, u))

    def project(self, x: FloatArray) -> FloatArray:
        """Clip ``x`` (in the optimizer's coordinates) to the feasible box."""
        x = np.asarray(x, dtype=float)
        if not self.bounded:
            return x
        if self.transform:
            return np.clip(x, 0.0, 1.0)
        return np.clip(x, self._column(self.lower, x), self._column(self.upper, x))

    def project_gradient(self, x: FloatArray, g: FloatArray, tol: float = 1e-8) -> FloatArray:
        """Zero the gradient's components that would leave an active bound.

        At a lower bound only a negative component (moving up) is kept; at an upper bound only
        a positive one.
        """
        if not self.bounded:
            return g
        lower, upper = (tol, 1.0 - tol) if self.transform else (self.lower + tol, self.upper - tol)
        projected = np.array(g, dtype=float, copy=True)
        at_lower = x <= lower
        at_upper = x >= upper
        projected[at_lower] = np.minimum(projected[at_lower], 0.0)
        projected[at_upper] = np.maximum(projected[at_upper], 0.0)
        return projected

    def gradient_to_internal(self, g: FloatArray) -> FloatArray:
        """Chain rule for a physical gradient: ``dF/du = dF/dx * width``."""
        return g * self.width if self.transform else g

    def gradient_to_physical(self, g: FloatArray) -> FloatArray:
        """Inverse of :meth:`gradient_to_internal`."""
        return g / self.width if self.transform else g

    def hessian_to_internal(self, h: FloatArray) -> FloatArray:
        """Chain rule for a physical Hessian: ``H_u = H_x * outer(width, width)``."""
        return h * np.outer(self.width, self.width) if self.transform else h

    def hessian_to_physical(self, h: FloatArray) -> FloatArray:
        """Inverse of :meth:`hessian_to_internal`."""
        return h / np.outer(self.width, self.width) if self.transform else h

    @staticmethod
    def _column(bound: FloatArray, like: FloatArray) -> FloatArray:
        # Bounds are per control; a set of points (n_x, n) needs them as a column.
        return bound[:, None] if like.ndim == 2 else bound
