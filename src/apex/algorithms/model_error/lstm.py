"""The physics-guided LSTM well residual model of model error characterization.

Ported from PET-fork
-----------------------
PET module : Examples/.../Brugge/ml_model_error.py (``LSTMWellModel.forward``,
             ``get_nn_params``, ``assign_nn_params``) and simulator/eclipse.py
             (``add_me_residual``), the checkout's working copy
PET commit : 1348861b08314b8b3f3e1ccf886c7c095273e202 (not public)
Reference  : Luo, X. Model error characterization for ensemble-based history matching
             through a physics guided long short-term memory network (manuscript,
             ``lstm_for_mec_CG.tex``); Hochreiter, S. & Schmidhuber, J. (1997). Long
             short-term memory. Neural Computation, 9(8), 1735-1780.
Differences: NumPy instead of torch, forward only (history matching updates the
             parameters through the smoother, never by training); the same parameter
             layout and float32 arithmetic (PET-fork forms u (2 (s - 0.5)) from 0-d
             tensors before rounding to float32, a difference of at most ~1 ulp).

For one well with ``I`` input features, ``O`` outputs and ``L`` report times, the network is
one LSTM layer of hidden size ``H`` with ``c_0 = h_0 = 0`` (manuscript Eqs. forget gate to
hidden state)::

    i_t = sigma(W_i x_t + b_ii + U_i h_{t-1} + b_hi)      (input gate)
    f_t = sigma(W_f x_t + b_if + U_f h_{t-1} + b_hf)      (forget gate)
    g_t = tanh (W_g x_t + b_ig + U_g h_{t-1} + b_hg)      (internal cell)
    o_t = sigma(W_o x_t + b_io + U_o h_{t-1} + b_ho)      (output gate)
    c_t = f_t * c_{t-1} + i_t * g_t,   h_t = o_t * tanh(c_t)

followed, at each report time, by a fully connected layer of its own and the "Scale"
function: ``r_t = u * (2 sigma(F_t h_t + e_t) - 1)``, zero where the well is shut.

The parameter vector ``theta`` is torch's ``named_parameters`` order: ``weight_ih`` (4H, I)
with the gates stacked (i, f, g, o), ``weight_hh`` (4H, H), ``bias_ih`` (4H), ``bias_hh``
(4H), then ``fc_t.weight`` (O, H) and ``fc_t.bias`` (O) for t = 0..L-1, each flattened in
row-major (C) order.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["WellNetwork", "well_residuals"]


@dataclass(frozen=True, slots=True)
class WellNetwork:
    """The shape of one well's PG-LSTM network.

    Attributes
    ----------
    input_size
        Features per report time, ``I``.
    hidden_size
        LSTM hidden size, ``H``.
    output_size
        Output data types, ``O``.
    n_steps
        Report times, ``L``.
    """

    input_size: int
    hidden_size: int
    output_size: int
    n_steps: int

    def __post_init__(self) -> None:
        """Validate the sizes."""
        for name in ("input_size", "hidden_size", "output_size", "n_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @property
    def n_params(self) -> int:
        """Number of parameters: ``4H(I + H) + 8H + L(OH + O)``."""
        h, i, o = self.hidden_size, self.input_size, self.output_size
        return 4 * h * (i + h) + 8 * h + self.n_steps * (o * h + o)

    def unpack(self, theta: NDArray[np.floating]) -> dict[str, NDArray[np.float32]]:
        """Split ``theta`` ``(n_params,)`` into the named float32 arrays (torch's layout)."""
        theta = np.asarray(theta)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must be ({self.n_params},), got {theta.shape}")
        # torch copies the float64 vector into float32 parameters (load_state_dict).
        flat = theta.astype(np.float32)
        h, i, o = self.hidden_size, self.input_size, self.output_size
        shapes = [
            ("weight_ih", (4 * h, i)),
            ("weight_hh", (4 * h, h)),
            ("bias_ih", (4 * h,)),
            ("bias_hh", (4 * h,)),
        ]
        for t in range(self.n_steps):
            shapes += [(f"fc_{t}.weight", (o, h)), (f"fc_{t}.bias", (o,))]
        arrays: dict[str, NDArray[np.float32]] = {}
        start = 0
        for name, shape in shapes:
            size = int(np.prod(shape))
            arrays[name] = flat[start : start + size].reshape(shape)
            start += size
        return arrays


def _sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
    # Split by sign so that exp never overflows (float32 overflows past ~88).
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    z = np.exp(x[~positive])
    out[~positive] = z / (1 + z)
    return out


def well_residuals(
    network: WellNetwork,
    features: NDArray[np.floating],
    theta: NDArray[np.floating],
    scale: NDArray[np.floating],
    open_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return one well's residuals for an ensemble of feature sequences and parameters.

    Parameters
    ----------
    network
        The network's shape.
    features
        Input features, ``(L, I, n)``: report time, feature, member.
    theta
        Parameters, ``(n_params, n)``.
    scale
        ``u`` per output, ``(O,)``: the residual lies in ``(-u, u)``.
    open_mask
        Whether each output counts at each report time, ``(L, O)``; ``False`` gives 0.

    Returns
    -------
    NDArray
        The residuals ``r``, ``(L, O, n)``, float64 values of float32 arithmetic.
    """
    n_steps, n_in = network.n_steps, network.input_size
    features = np.asarray(features)
    theta = np.asarray(theta)
    if features.ndim != 3 or features.shape[:2] != (n_steps, n_in):
        raise ValueError(f"features must be ({n_steps}, {n_in}, n), got {features.shape}")
    n = features.shape[-1]
    if theta.shape != (network.n_params, n):
        raise ValueError(f"theta must be ({network.n_params}, {n}), got {theta.shape}")
    scale = np.asarray(scale, dtype=np.float32)
    if scale.shape != (network.output_size,):
        raise ValueError(f"scale must be ({network.output_size},), got {scale.shape}")
    open_mask = np.asarray(open_mask, dtype=bool)
    if open_mask.shape != (n_steps, network.output_size):
        raise ValueError(
            f"open_mask must be ({n_steps}, {network.output_size}), got {open_mask.shape}"
        )
    x = features.astype(np.float32)
    h_size = network.hidden_size
    result = np.empty((n_steps, network.output_size, n))
    for j in range(n):  # each member has its own weights
        p = network.unpack(theta[:, j])
        h = np.zeros(h_size, dtype=np.float32)
        c = np.zeros(h_size, dtype=np.float32)
        # The input projections of every step at once, as torch does.
        projected = x[:, :, j] @ p["weight_ih"].T + p["bias_ih"]  # (L, 4H)
        for t in range(n_steps):
            gates = projected[t] + (p["weight_hh"] @ h + p["bias_hh"])
            i_gate = _sigmoid(gates[:h_size])
            f_gate = _sigmoid(gates[h_size : 2 * h_size])
            g_gate = np.tanh(gates[2 * h_size : 3 * h_size])
            o_gate = _sigmoid(gates[3 * h_size :])
            c = f_gate * c + i_gate * g_gate
            h = o_gate * np.tanh(c)
            s = _sigmoid(p[f"fc_{t}.weight"] @ h + p[f"fc_{t}.bias"])
            # The "Scale" function: u (2 s - 1), in (-u, u); torch computes 2 (s - 0.5).
            r = scale * (np.float32(2) * (s - np.float32(0.5)))
            result[t, :, j] = np.where(open_mask[t], r, np.float32(0))
    return result
