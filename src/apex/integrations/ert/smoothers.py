"""ERT's smoothers as APEX update strategies.

Adapted from ERT
----------------
Library   : iterative_ensemble_smoother 1.3.0 (github.com/equinor/iterative_ensemble_smoother,
            tag v1.3.0; GPL-3.0), the analysis of ERT 25.0.4 and 26.0.1 (``ESMDA``,
            ``AdaptiveESMDA``, ``LocalizedESMDA``); called, not copied. Pinned in the ``ert`` extra.
Reference : Emerick, A. A. & Reynolds, A. C. (2013). Ensemble smoother with multiple data
            assimilation. Computers & Geosciences, 55, 3-15 (ES-MDA). Vossepoel, F. C.,
            Evensen, G. & van Leeuwen, P. J. (2025), Monthly Weather Review, eq. 6 (the
            adaptive ``3/sqrt(N)`` correlation threshold, as the library cites).
            Emerick, A. A. (2016). Analysis of the performance of ensemble-based
            assimilation of production and seismic data. Journal of Petroleum Science and
            Engineering, 139, 219-239 (Kalman-gain localization).
Status    : validated (an ERT method PET lacks; independent references in
            tests/unit/integrations/test_ert_smoothers.py, and on 5Spots, 2026-10-04:
            ert_esmda equals APEX's approx analysis and ert_localized APEX's distance
            localization with the same taper, ert_adaptive improves on no localization).

Each plugin is the analysis of APEX's ES and ES-MDA (and GN-EnRML, which runs its analysis
at damping 0): the step gets APEX's perturbed observations ``D`` and the already inflated
``C_d``, and calls the library with ``alpha = 1``, ``observations = 0`` and
``observation_perturbations = D`` (so the library's ``D_obs`` is APEX's ``D``). APEX's random
streams, perturbations and checkpoints therefore apply unchanged; the library draws nothing.
GIES and LM-EnRML damp their updates and refuse these strategies (``supports_damping``).

The state is assimilated in row batches within ``max_batch_mb`` (the library's results do
not depend on the batching: each row's update is its own), so the adaptive correlations and
the localized gain, ``(rows, n_data)`` each, stay bounded. Rows without spread are not
updated. Registered through the entry point ``apex.update_strategies``; the library is
imported only when a step runs.
"""

import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.core.errors import ConfigError
from apex.core.options import InputFile

logger = logging.getLogger(__name__)

#: The library's default fraction of singular values kept (Emerick's recommendation).
DEFAULT_TRUNCATION = 0.99
#: Memory budget of one batch's (rows, n_data) arrays.
DEFAULT_BATCH_MB = 256.0


def _library() -> Any:  # noqa: ANN401 (the module)
    try:
        import iterative_ensemble_smoother
    except ImportError as error:
        raise ConfigError(
            "ERT's smoothers need the `ert` extra (iterative_ensemble_smoother): "
            "pip install apex[ert]"
        ) from error
    return iterative_ensemble_smoother


class _ErtSmoother:
    """The step every ERT smoother shares (see the module docstring)."""

    #: Validated against independent references and on 5Spots.
    status = "validated"
    #: Its library, of the ``ert`` extra (checked by ``apex plugins``, not imported).
    requires: ClassVar[Mapping[str, str]] = MappingProxyType({"iterative_ensemble_smoother": "ert"})
    requires_reference_prediction = False
    relative_damping = False
    #: No Levenberg-Marquardt damping: GIES and LM-EnRML refuse these strategies.
    supports_damping = False
    #: The library class.
    kind = "ESMDA"

    def __init__(
        self, truncation: float = DEFAULT_TRUNCATION, max_batch_mb: float = DEFAULT_BATCH_MB
    ) -> None:
        if not 0 < truncation <= 1:
            raise ValueError(f"truncation must be in (0, 1], got {truncation}")
        if max_batch_mb <= 0:
            raise ValueError(f"max_batch_mb must be positive, got {max_batch_mb}")
        self.truncation = truncation
        self.max_batch_mb = max_batch_mb

    def step(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        *,
        damping: float = 0.0,
        reference_prediction: NDArray[np.float64] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Return the increment ``(n_state, n_ensemble)`` of ERT's analysis.

        Raises
        ------
        ValueError
            With damping or a reference prediction: ERT's smoothers are the analysis of ES
            and ES-MDA, not of the damped iterative methods.
        ConfigError
            Without the ``ert`` extra.
        """
        if damping != 0.0:
            raise ValueError("ERT's smoothers take no damping: use them with es or esmda")
        if reference_prediction is not None:
            raise ValueError("ERT's smoothers centre on the ensemble mean, not a reference")
        n_data, n = predicted.shape
        if perturbed_observations.shape != predicted.shape or state.shape[1] != n:
            raise ValueError(
                f"shapes differ: state {state.shape}, predicted {predicted.shape}, "
                f"perturbed observations {perturbed_observations.shape}"
            )
        self._check(state, predicted)
        library = _library()
        smoother = getattr(library, self.kind)(
            covariance=data_error.values,
            observations=np.zeros(n_data),
            alpha=np.array([1.0]),
            seed=0,  # unused: the perturbations are APEX's
        )
        smoother.prepare_assimilation(
            Y=np.array(predicted, dtype=np.float64),
            truncation=self.truncation,
            observation_perturbations=np.array(perturbed_observations, dtype=np.float64),
        )
        increment = np.zeros_like(state, dtype=np.float64)
        # Rows with no spread have no correlation and no update (and would only give NaN
        # warnings in the adaptive correlations).
        rows = np.flatnonzero(np.std(state, axis=1) > 0)
        batch = max(1, int(self.max_batch_mb * 2**20 / (8 * max(n_data, n))))
        for start in range(0, rows.size, batch):
            chosen = rows[start : start + batch]
            prior = np.array(state[chosen], dtype=np.float64)
            posterior = self._assimilate(smoother, prior, chosen, n)
            increment[chosen] = posterior - prior
        return np.ascontiguousarray(increment)

    def _check(self, state: NDArray[np.float64], predicted: NDArray[np.float64]) -> None:
        """Check what the strategy needs of the shapes (nothing by default)."""

    def _assimilate(
        self,
        smoother: Any,  # noqa: ANN401 (the library's object)
        state: NDArray[np.float64],
        rows: NDArray[np.int64],
        n_members: int,
    ) -> NDArray[np.float64]:
        return np.asarray(smoother.assimilate_batch(X=state), dtype=np.float64)


class ErtEsmda(_ErtSmoother):
    """ERT's ES-MDA analysis (``iterative_ensemble_smoother.ESMDA``).

    Parameters
    ----------
    truncation
        Fraction of the singular values kept in the inversion, in ``(0, 1]``.
    max_batch_mb
        Memory budget of one batch of state rows.
    """

    kind = "ESMDA"


class ErtAdaptive(_ErtSmoother):
    """ERT's adaptive localization (``AdaptiveESMDA``).

    For each state row, the data it correlates with above ``threshold`` (``|corr| >
    threshold``) are kept, and its update is recomputed with the inversion over those data
    only; a row correlating with none is not updated.

    Parameters
    ----------
    truncation
        Fraction of the singular values kept in the inversion, in ``(0, 1]``.
    threshold
        Correlation threshold; default ERT's ``3 / sqrt(N)``, which keeps nothing below ten
        members (refused there).
    n_jobs
        Worker processes of the library's per-row-pattern inversions (default 1: the
        analysis stays within the run's own core).
    max_batch_mb
        Memory budget of one batch of state rows.
    """

    kind = "AdaptiveESMDA"

    def __init__(
        self,
        truncation: float = DEFAULT_TRUNCATION,
        threshold: float | None = None,
        n_jobs: int = 1,
        max_batch_mb: float = DEFAULT_BATCH_MB,
    ) -> None:
        super().__init__(truncation, max_batch_mb)
        if threshold is not None and not 0 <= threshold < 1:
            raise ValueError(f"threshold must be in [0, 1), got {threshold}")
        if n_jobs == 0:
            raise ValueError("n_jobs must be positive, or negative for all cores less some")
        self.threshold = threshold
        self.n_jobs = n_jobs

    def check_members(self, n: int) -> None:
        """Refuse an ensemble of ``n`` members ERT's default threshold keeps no data with.

        Raises
        ------
        ValueError
            If the default threshold ``3/sqrt(n)`` is 1 or more.
        """
        self._threshold(n)

    def _threshold(self, n_members: int) -> float:
        threshold = self.threshold
        if threshold is None:
            threshold = 3 / np.sqrt(n_members)
            if threshold >= 1:
                raise ValueError(
                    f"ERT's default threshold 3/sqrt(N) = {threshold:.3g} keeps no data with "
                    f"{n_members} members: give a threshold below 1"
                )
        return float(threshold)

    def _assimilate(
        self,
        smoother: Any,  # noqa: ANN401
        state: NDArray[np.float64],
        rows: NDArray[np.int64],
        n_members: int,
    ) -> NDArray[np.float64]:
        threshold = self._threshold(n_members)
        kept = [0]

        def keep(correlation: NDArray[np.float64], members: Any) -> NDArray[np.bool_]:  # noqa: ANN401
            mask = np.abs(correlation) > threshold
            kept[0] += int(mask.sum())
            return np.asarray(mask)

        with warnings.catch_warnings():
            # Data with no spread give NaN correlations, which are not kept (no update).
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*not in \\[-1, 1\\].*")
            posterior = smoother.assimilate_batch(
                X=state, correlation_callback=keep, n_jobs=self.n_jobs
            )
        if kept[0] == 0:
            logger.warning(
                "ert_adaptive kept no state-datum pairs: these rows are not updated",
                extra={"event": "adaptive_kept_nothing", "rows": int(rows.size),
                       "threshold": float(threshold)},
            )  # fmt: skip
        return np.asarray(posterior, dtype=np.float64)


class ErtLocalized(_ErtSmoother):
    """ERT's localized ES-MDA (``LocalizedESMDA``): the Kalman gain times a given taper.

    Parameters
    ----------
    taper_file
        ``.npy``, or ``.npz`` with ``taper``: the taper ``(n_state, n_data)`` in ``[0, 1]``,
        state rows in layout order and data columns in the analysis data's order (only the
        shape can be checked: build it for the configuration it is used with).
    truncation
        Fraction of the singular values kept in the inversion, in ``(0, 1]``.
    max_batch_mb
        Memory budget of one batch of state rows.
    """

    kind = "LocalizedESMDA"

    def __init__(
        self,
        taper_file: InputFile,
        truncation: float = DEFAULT_TRUNCATION,
        max_batch_mb: float = DEFAULT_BATCH_MB,
    ) -> None:
        super().__init__(truncation, max_batch_mb)
        path = Path(taper_file)
        if path.suffix == ".npz":
            with np.load(path) as stored:
                taper = np.asarray(stored["taper"], dtype=np.float64)
        else:
            taper = np.asarray(np.load(path), dtype=np.float64)
        if taper.ndim != 2 or not np.all(np.isfinite(taper)):
            raise ValueError(f"{path}: the taper must be a finite (n_state, n_data) array")
        if np.any(taper < 0) or np.any(taper > 1):
            raise ValueError(f"{path}: taper values must be in [0, 1]")
        self.taper = taper

    def _check(self, state: NDArray[np.float64], predicted: NDArray[np.float64]) -> None:
        if self.taper.shape != (state.shape[0], predicted.shape[0]):
            raise ValueError(
                f"the taper is {self.taper.shape}; this analysis needs "
                f"{(state.shape[0], predicted.shape[0])}"
            )

    def _assimilate(
        self,
        smoother: Any,  # noqa: ANN401
        state: NDArray[np.float64],
        rows: NDArray[np.int64],
        n_members: int,
    ) -> NDArray[np.float64]:
        taper = self.taper[rows]

        def localize(gain: NDArray[np.float64]) -> NDArray[np.float64]:
            return gain * taper

        return np.asarray(
            smoother.assimilate_batch(X=state, localization_callback=localize), dtype=np.float64
        )
