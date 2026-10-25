"""Net present value of a simulation's field totals.

Ported from PET
---------------
PET module : popt/cost_functions/npv.py (``npv``)
PET commit : fd17979dde98928e9a7191d7a251be4ca59ed50a (the NPV objectives were removed
             upstream before e65fb10)
Reference  : Brouwer, D. R. & Jansen, J. D. (2004). Dynamic optimization of waterflooding
             with smart wells using optimal control theory. SPE Journal, 9(4), 391-402
             (the discounted NPV of production and injection).
Differences: the report times are dates of the simulator's summary; the start is the
             simulation's start, where the cumulative totals are zero (PET reads them at its
             first report point, which its tutorial puts at the start). Simulators do not
             report at the start, so the summary's TIME (days since the start) at the first
             report is checked against ``start``: a ``start`` that is not the deck's START
             would silently count earlier production. The discount uses the days from the
             start to each report, over 365, as PET's.

For report times ``t_1 < ... < t_K`` after the start ``t_0``:

    NPV = sum_k (ΔQ_op,k r_o + ΔQ_gp,k r_g - ΔQ_wp,k r_w - ΔQ_wi,k r_wi) / (1 + d)^(Δt_k / 365)

with ``ΔQ_k`` the increase of the field totals (FOPT, FGPT, FWPT, FWIT) over ``(t_{k-1},
t_k]``, ``r`` the prices and costs, ``d`` the yearly discount rate and ``Δt_k`` the days
from ``t_0`` to ``t_k``. The objective is ``NPV / scaling``: PET's tutorial scales by
``-1e6``, so that minimising it maximises NPV in millions.
"""

import datetime as dt
from collections.abc import Sequence
from typing import Annotated

import numpy as np

from apex.algorithms.optimization.protocols import FloatArray
from apex.algorithms.optimization.registries import response_objectives
from apex.core.options import TEXT, TextList
from apex.domain.observations import DataKey

#: The field totals NPV reads, and PET's price keyword of each.
TOTALS = {"FOPT": "wop", "FGPT": "wgp", "FWPT": "wwp", "FWIT": "wwi"}
#: The sign of each total in NPV: revenue, or cost.
SIGNS = {"FOPT": 1.0, "FGPT": 1.0, "FWPT": -1.0, "FWIT": -1.0}
#: The summary vector of days since the simulation's start, which checks ``start``.
TIME = "TIME"
#: How far TIME may be from the days ``start`` implies (summaries store it in single
#: precision; half a day separates whole dates).
TIME_TOLERANCE_DAYS = 0.5


@response_objectives.register("npv")
class NetPresentValue:
    """NPV of the field's oil and gas production, water production and water injection.

    Parameters
    ----------
    start
        The simulation's start date (ISO), where the totals are zero.
    times
        Report dates (ISO), increasing and after ``start``; the simulator must report at
        each.
    wop, wgp
        Oil and gas prices, per unit of FOPT and FGPT.
    wwp, wwi
        Costs of produced and injected water, per unit of FWPT and FWIT.
    disc
        Yearly discount rate.
    obj_scaling
        Divides NPV; negative to maximise it (PET's tutorial: ``-1e6``).
    from_time
        One of ``times`` from which NPV counts: only the steps after it, with
        the discount from it. For a forecast after a history, whose production is not the
        decision's. Default: from ``start``, as PET.
    """

    status = "parity"

    def __init__(
        self,
        start: Annotated[str, TEXT],
        times: TextList,
        wop: float,
        wgp: float = 0.0,
        wwp: float = 0.0,
        wwi: float = 0.0,
        disc: float = 0.0,
        obj_scaling: float = 1.0,
        from_time: Annotated[str | None, TEXT] = None,
    ) -> None:
        dates = [dt.date.fromisoformat(t) for t in times]
        first = dt.date.fromisoformat(start)
        if not dates:
            raise ValueError("npv needs at least one report time")
        if any(b <= a for a, b in zip([first, *dates], dates, strict=False)):
            raise ValueError("npv's times must increase, after the start")
        values = {"wop": wop, "wgp": wgp, "wwp": wwp, "wwi": wwi, "disc": disc,
                  "obj_scaling": obj_scaling}  # fmt: skip
        bad = [k for k, v in values.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"npv's {', '.join(bad)} must be finite")
        if obj_scaling == 0:
            raise ValueError("obj_scaling must not be zero")
        if disc <= -1:
            raise ValueError(f"disc must be above -1, got {disc}")
        if wop == wgp == wwp == wwi == 0:
            raise ValueError("npv needs a nonzero price or cost (wop, wgp, wwp or wwi)")
        self.times = tuple(times)
        self.prices = {"FOPT": wop, "FGPT": wgp, "FWPT": wwp, "FWIT": wwi}
        self.obj_scaling = obj_scaling
        days = np.array([(d - first).days for d in dates], dtype=float)
        self.first_days = float(days[0])
        # Steps counted: all (PET), or those after from_time.
        self.counted = np.ones(len(dates), dtype=bool)
        origin = 0.0
        if from_time is not None:
            if from_time not in self.times:
                raise ValueError(f"from_time {from_time} must be one of the report times")
            index = self.times.index(from_time)
            if index == len(dates) - 1:
                raise ValueError("from_time must be before the last report time")
            self.counted[: index + 1] = False
            origin = float(days[index])
        self.from_time = from_time
        # PET: (1 + disc)^(days since the start / 365) for the step ending at each report;
        # with from_time, the days since it.
        self.discount = (1.0 + disc) ** ((days - origin) / 365.0)
        # Only totals with a price are simulated; a zero price needs no response.
        self.vectors = tuple(v for v in TOTALS if self.prices[v] != 0.0)

    def response_keys(self) -> Sequence[DataKey]:
        """Return each priced total at each report time, vector by vector, then TIME."""
        totals = [DataKey(t, v) for v in self.vectors for t in self.times]
        return [*totals, DataKey(self.times[0], TIME)]

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return ``NPV / obj_scaling`` of each simulation ``(n,)``.

        Parameters
        ----------
        responses
            The totals and TIME at :meth:`response_keys` ``(n_vectors * n_times + 1, n)``.

        Raises
        ------
        ValueError
            If the shape is wrong, or TIME shows that ``start`` is not the simulation's start.
        """
        n_times = len(self.times)
        n_totals = len(self.vectors) * n_times
        if responses.ndim != 2 or responses.shape[0] != n_totals + 1:
            raise ValueError(f"responses must be ({n_totals + 1}, n), got {responses.shape}")
        elapsed = responses[n_totals]
        if np.any(np.abs(elapsed - self.first_days) > TIME_TOLERANCE_DAYS):
            raise ValueError(
                f"npv's start is not the simulation's start: {self.times[0]} is "
                f"{self.first_days:g} days after start, the simulation says "
                f"{sorted(set(elapsed.tolist()))} (set start to the deck's START)"
            )
        totals = responses[:n_totals].reshape(len(self.vectors), n_times, -1)
        # The increase over each report step; the totals are zero at the start.
        steps = np.diff(totals, axis=1, prepend=0.0)
        rates = np.array([SIGNS[v] * self.prices[v] for v in self.vectors])
        cash = np.einsum("v,vtn->tn", rates, steps)
        npv = np.sum((cash / self.discount[:, None])[self.counted], axis=0)
        return np.asarray(npv / self.obj_scaling, dtype=np.float64)


@response_objectives.register("ren_npv")
class RenewableNetPresentValue:
    """NPV with water injection priced per well: by grid or by renewable power.

    Ported from PET
    ---------------
    PET module : popt/cost_functions/ren_npv.py (``ren_npv``)
    PET commit : fd17979dde98928e9a7191d7a251be4ca59ed50a (removed before e65fb10)
    Reference  : as ``npv``; injection powered by renewable energy is priced separately.
    Differences: PET splits injectors by ``'ren'`` in their ``wwit`` keys; here they are
                 listed (``injectors`` at ``wwi``, ``renewable_injectors`` at ``wrenwi``).
                 Only priced totals are simulated, and the start is checked by TIME, as
                 ``npv``.

    ``NPV = sum_k (ΔFOPT r_o + ΔFGPT r_g - ΔFWPT r_w - ΣΔWWIT_inj r_wi - ΣΔWWIT_ren r_ren)
    / (1 + d)^(Δt_k / 365)``, the totals' increases over each report step, ``Δt_k`` the days
    from the start to the step's end (PET's).

    Parameters
    ----------
    start, times, wop, wgp, wwp, disc, obj_scaling, from_time
        As ``npv``.
    injectors
        Wells whose injected water (WWIT) costs ``wwi``.
    renewable_injectors
        Wells whose injected water costs ``wrenwi`` (powered by renewable energy).
    wwi
        Cost per unit of water injected by ``injectors``.
    wrenwi
        Cost per unit of water injected by ``renewable_injectors`` (powered by renewable
        energy).
    """

    status = "parity"

    def __init__(
        self,
        start: Annotated[str, TEXT],
        times: TextList,
        wop: float,
        wgp: float = 0.0,
        wwp: float = 0.0,
        wwi: float = 0.0,
        wrenwi: float = 0.0,
        injectors: TextList = (),
        renewable_injectors: TextList = (),
        disc: float = 0.0,
        obj_scaling: float = 1.0,
        from_time: Annotated[str | None, TEXT] = None,
    ) -> None:
        overlap = set(injectors) & set(renewable_injectors)
        if overlap:
            raise ValueError(f"wells both in injectors and renewable_injectors: {sorted(overlap)}")
        if not (np.isfinite(wwi) and np.isfinite(wrenwi)):
            raise ValueError("ren_npv's wwi and wrenwi must be finite")
        if (wwi and not injectors) or (wrenwi and not renewable_injectors):
            raise ValueError("ren_npv prices injection without wells: list the injectors")
        if not (wop or wgp or wwp):
            raise ValueError("ren_npv needs a nonzero wop, wgp or wwp")
        # The field part: oil, gas and produced water as npv (FWIT is not used: injection
        # is priced well by well).
        self.field = NetPresentValue(
            start, times, wop, wgp, wwp, 0.0, disc, obj_scaling, from_time=from_time
        )
        self.times = self.field.times
        self.wells = (*injectors, *renewable_injectors)
        self.costs = np.array([wwi] * len(injectors) + [wrenwi] * len(renewable_injectors))

    def response_keys(self) -> Sequence[DataKey]:
        """Return npv's keys (totals, then TIME), then each injector's WWIT at each time."""
        wells = [DataKey(t, f"WWIT:{w}") for w in self.wells for t in self.times]
        return [*self.field.response_keys(), *wells]

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return ``NPV / obj_scaling`` of each simulation ``(n,)``.

        Raises
        ------
        ValueError
            If the shape is wrong, or TIME shows that ``start`` is not the simulation's start.
        """
        n_field = len(self.field.response_keys())
        n_times = len(self.times)
        expected = n_field + len(self.wells) * n_times
        if responses.ndim != 2 or responses.shape[0] != expected:
            raise ValueError(f"responses must be ({expected}, n), got {responses.shape}")
        value = self.field.evaluate(responses[:n_field]) * self.field.obj_scaling
        if self.wells:
            totals = responses[n_field:].reshape(len(self.wells), n_times, -1)
            steps = np.diff(totals, axis=1, prepend=0.0)  # zero at the start
            cost = np.einsum("w,wtn->tn", self.costs, steps)
            counted = self.field.counted
            value = value - np.sum((cost / self.field.discount[:, None])[counted], axis=0)
        return np.asarray(value / self.field.obj_scaling, dtype=np.float64)


@response_objectives.register("field_total")
class FieldTotal:
    """A field total at one time, scaled: e.g. the water injected.

    A second objective next to an economic one, such as NPV against the water injected
    (``vector = "FWIT"``); the value is ``scale * vector(time)``, minimised like every
    objective (a negative ``scale`` maximises it).

    Parameters
    ----------
    vector
        The summary vector, a cumulative total such as ``FWIT``, ``FOPT`` or ``FGPT``.
    time
        The report time (ISO date).
    scale
        Multiplies the total (e.g. ``1e-6`` for millions).
    """

    status = "validated"

    def __init__(
        self,
        vector: Annotated[str, TEXT],
        time: Annotated[str, TEXT],
        scale: float = 1.0,
    ) -> None:
        dt.date.fromisoformat(time)  # an ISO date, or a clear error
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"scale must be finite and nonzero, got {scale}")
        self.vector = vector
        self.time = time
        self.scale = scale

    def response_keys(self) -> Sequence[DataKey]:
        """Return the one key: the vector at the time."""
        return [DataKey(self.time, self.vector)]

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return ``scale * total`` of each simulation ``(n,)`` from ``(1, n)`` responses."""
        if responses.shape[0] != 1:
            raise ValueError(f"field_total takes one response, got {responses.shape[0]}")
        return np.asarray(self.scale * responses[0], dtype=float)
