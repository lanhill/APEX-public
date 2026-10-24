from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError, SimulationError
from apex.core.plugins import executors, simulators
from apex.core.simulation import MemberFailure, MemberInput
from apex.domain import DataKey
from apex.execution.local import LocalProcessExecutor, SerialExecutor
from apex.simulation.analytic import LinearModel

KEYS = [DataKey("t0", "y"), DataKey("t1", "y")]


def _model(**kwargs: object) -> LinearModel:
    return LinearModel(
        outputs=[["t0", "y"], ["t1", "y"]],
        variables=["x"],
        matrix=[[1.0, 0.0], [0.0, 2.0]],
        **kwargs,  # type: ignore[arg-type]
    )


def _members(n: int) -> list[MemberInput]:
    return [MemberInput(i, {"x": np.array([float(i), 1.0])}) for i in range(n)]


class _Buggy:
    """A simulator with a programming error, which must not be swallowed."""

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        raise KeyError("bug")


def test_linear_model_evaluates_and_selects_keys() -> None:
    model = _model(nonlinearity="tanh")

    y = model.run(_members(1)[0], [KEYS[1]])

    np.testing.assert_allclose(y, [np.tanh(2.0)])
    with pytest.raises(SimulationError, match="no output"):
        model.run(_members(1)[0], [DataKey("t9", "y")])


@pytest.mark.parametrize("executor", [SerialExecutor(), LocalProcessExecutor(workers=2)])
def test_executors_keep_order_and_capture_simulation_failures(
    executor: SerialExecutor | LocalProcessExecutor,
) -> None:
    outputs = executor.run_all(_model(fail_members=[2]), _members(4), KEYS)

    assert isinstance(outputs[2], MemberFailure)
    assert outputs[2].member_id == 2
    for i in (0, 1, 3):
        np.testing.assert_array_equal(outputs[i], [float(i), 2.0])


@pytest.mark.parametrize("executor", [SerialExecutor(), LocalProcessExecutor(workers=3)])
def test_executors_report_each_member_as_it_finishes(
    executor: SerialExecutor | LocalProcessExecutor,
) -> None:
    finished: list[tuple[int, bool]] = []

    outputs = executor.run_all(
        _model(fail_members=[2]),
        _members(5),
        KEYS,
        on_result=lambda i, ok: finished.append((i, ok)),
    )

    # Once per member, failures included; the results keep the input order regardless.
    assert sorted(finished) == [(0, True), (1, True), (2, False), (3, True), (4, True)]
    assert isinstance(outputs[2], MemberFailure)
    np.testing.assert_array_equal(outputs[4], [4.0, 2.0])


def test_executor_propagates_bugs_and_creates_work_dirs(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="bug"):
        SerialExecutor().run_all(_Buggy(), _members(1), KEYS)

    member = MemberInput(0, {"x": np.zeros(2)}, work_dir=tmp_path / "m0")
    SerialExecutor().run_all(_model(), [member], KEYS)
    assert (tmp_path / "m0").is_dir()


class _BuggyMembers:
    """A simulator with a bug in some members only."""

    def __init__(self, buggy: set[int]) -> None:
        self.buggy = buggy

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        if member.member_id in self.buggy:
            raise KeyError(f"bug in member {member.member_id}")
        return np.zeros(len(keys))


@pytest.mark.parametrize("executor", [SerialExecutor(), LocalProcessExecutor(workers=3)])
def test_a_bug_propagates_with_a_callback_too(
    executor: SerialExecutor | LocalProcessExecutor,
) -> None:
    finished: list[int] = []

    # The first buggy member in input order raises, whichever finishes first.
    with pytest.raises(KeyError, match="member 1"):
        executor.run_all(
            _BuggyMembers({1, 3}), _members(5), KEYS, on_result=lambda i, ok: finished.append(i)
        )
    assert 1 not in finished


def test_plugins_are_registered() -> None:
    assert {"serial", "process"} <= set(executors.names())
    assert "linear" in simulators.names()


class _Timed:
    """Records when each run starts and ends, to measure how many run at once."""

    needs_work_dir = True

    def __init__(self, max_parallel_runs: int | None) -> None:
        self.max_parallel_runs = max_parallel_runs

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        import time

        assert member.work_dir is not None
        start = time.monotonic()
        time.sleep(0.3)
        (member.work_dir / "span").write_text(f"{start} {time.monotonic()}")
        return np.zeros(len(keys))


def _most_at_once(members: Sequence[MemberInput]) -> int:
    spans = [
        tuple(map(float, (m.work_dir / "span").read_text().split()))  # type: ignore[operator]
        for m in members
    ]
    return max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)


@pytest.mark.parametrize("cap", [1, 3])
def test_a_simulator_licence_cap_limits_the_parallel_runs(tmp_path: Path, cap: int) -> None:
    # E.g. Eclipse with licenses = 3: never more runs at once, whatever the workers.
    members = [MemberInput(i, {}, tmp_path / f"member_{i}") for i in range(8)]
    results = LocalProcessExecutor(workers=8).run_all(_Timed(cap), members, [])
    assert all(not isinstance(r, MemberFailure) for r in results)
    assert _most_at_once(members) <= cap


@pytest.mark.parametrize(
    "executor_type",
    [SerialExecutor, lambda **k: LocalProcessExecutor(workers=2, **k)],
    ids=["serial", "process"],
)
def test_transient_failures_are_retried_and_others_are_not(executor_type: object) -> None:
    """Regression for the architecture review's item E3: a licence hiccup dropped a member
    for the rest of the run. Member 1 fails transiently twice then works; member 3 fails
    transiently more often than the retries allow; member 2 fails for good.
    """
    executor = executor_type(retries=2, retry_delay_s=0.0)  # type: ignore[operator]
    model = _model(flaky_members={"1": 2, "3": 5}, fail_members=[2])

    outputs = executor.run_all(model, _members(4), KEYS)

    np.testing.assert_array_equal(outputs[1], [1.0, 2.0])  # succeeded on the third run
    permanent, exhausted = outputs[2], outputs[3]
    assert isinstance(permanent, MemberFailure)
    assert (permanent.transient, permanent.attempts) == (False, 1)  # never retried
    assert isinstance(exhausted, MemberFailure)
    assert (exhausted.transient, exhausted.attempts) == (True, 3)
    assert "licence not available" in exhausted.message


def test_retries_wait_longer_each_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    waits: list[float] = []
    monkeypatch.setattr(time, "sleep", waits.append)  # the executor calls time.sleep

    SerialExecutor(retries=3, retry_delay_s=10.0).run_all(
        _model(flaky_members={0: 3}), _members(1), KEYS
    )

    assert waits == [10.0, 20.0, 40.0]


def test_without_retries_a_transient_failure_fails_the_member() -> None:
    outputs = SerialExecutor(retries=0).run_all(_model(flaky_members={0: 1}), _members(1), KEYS)

    assert isinstance(outputs[0], MemberFailure)
    assert (outputs[0].transient, outputs[0].attempts) == (True, 1)


def test_workers_do_not_fork_a_multi_threaded_process() -> None:
    """Workers start from a forkserver, not from the run's process (which may have threads).

    Libraries start threads in the run's own process (OpenBLAS; pyarrow, which pandas loads
    when installed); forking such a process may deadlock the child, and Python warns.
    """
    import threading
    import warnings

    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)
    thread.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outputs = LocalProcessExecutor(workers=2).run_all(_model(), _members(3), KEYS)
    finally:
        stop.set()
        thread.join()

    np.testing.assert_array_equal(outputs[1], [1.0, 2.0])
    assert not [w for w in caught if "fork()" in str(w.message)]


class _MisconfiguredMembers:
    """A simulator whose members find a configuration error in their own output."""

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        if member.member_id == 2:
            raise ConfigError("the deck writes no restart at a vintage")
        return np.zeros(len(keys))


@pytest.mark.parametrize("executor", [SerialExecutor(), LocalProcessExecutor(workers=3)])
def test_a_members_configuration_error_stops_the_run(
    executor: SerialExecutor | LocalProcessExecutor,
) -> None:
    """A member's ``ConfigError`` (e.g. a geophysics read) is not a member
    failure: it stops the run with its message, under the process executor too."""
    with pytest.raises(ConfigError, match="no restart at a vintage"):
        executor.run_all(_MisconfiguredMembers(), _members(4), KEYS)
