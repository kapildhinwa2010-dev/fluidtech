import threading

import pytest

from deployment.session_registry import (
    SessionBusyError,
    SessionCapacityError,
    SessionRegistry,
)


class DummyBranch:
    def __init__(self):
        self.values = []
        self.reset_count = 0

    def reset(self):
        self.values.clear()
        self.reset_count += 1


def test_sessions_are_isolated_by_tenant_and_session():
    registry = SessionRegistry(DummyBranch)

    with registry.lease("tenant-a", "conversation") as first:
        first.values.append("private")
    with registry.lease("tenant-b", "conversation") as second:
        assert second.values == []
        second.values.append("other")
    with registry.lease("tenant-a", "conversation") as first_again:
        assert first_again is first
        assert first_again.values == ["private"]


def test_capacity_is_a_hard_bound_and_expired_sessions_are_reclaimed():
    now = [0.0]
    registry = SessionRegistry(
        DummyBranch, max_sessions=1, ttl_seconds=5, clock=lambda: now[0]
    )
    with registry.lease("tenant", "one"):
        pass

    with pytest.raises(SessionCapacityError):
        with registry.lease("tenant", "two"):
            pass

    now[0] = 5.0
    with registry.lease("tenant", "two"):
        assert registry.session_count == 1


def test_active_leases_are_not_expired():
    now = [0.0]
    registry = SessionRegistry(DummyBranch, ttl_seconds=2, clock=lambda: now[0])

    with registry.lease("tenant", "one"):
        now[0] = 100.0
        assert registry.session_count == 1


def test_reset_and_close_require_an_idle_session():
    registry = SessionRegistry(DummyBranch)
    with registry.lease("tenant", "one") as branch:
        branch.values.append("value")
        with pytest.raises(SessionBusyError):
            registry.reset_session("tenant", "one")
        with pytest.raises(SessionBusyError):
            registry.close_session("tenant", "one")

    assert registry.reset_session("tenant", "one")
    with registry.lease("tenant", "one") as reset_branch:
        assert reset_branch.values == []
        assert reset_branch.reset_count == 1
    assert registry.close_session("tenant", "one")
    assert not registry.close_session("tenant", "one")


def test_requests_for_one_session_are_serialized():
    registry = SessionRegistry(DummyBranch)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def hold_first_lease():
        with registry.lease("tenant", "same"):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def acquire_second_lease():
        with registry.lease("tenant", "same"):
            second_entered.set()

    first = threading.Thread(target=hold_first_lease)
    second = threading.Thread(target=acquire_second_lease)
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_sessions": 0}, ValueError),
        ({"max_sessions": True}, TypeError),
        ({"ttl_seconds": 0}, ValueError),
        ({"ttl_seconds": float("inf")}, ValueError),
    ],
)
def test_configuration_is_validated(kwargs, error):
    with pytest.raises(error):
        SessionRegistry(DummyBranch, **kwargs)


def test_blank_session_identifiers_are_rejected():
    registry = SessionRegistry(DummyBranch)
    with pytest.raises(ValueError):
        with registry.lease("tenant", "  "):
            pass
