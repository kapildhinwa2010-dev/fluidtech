"""Bounded, process-local isolation for stateful per-session memory branches.

This helper is an application integration aid, not an authentication system
or a cross-process session store. Applications must derive identifiers from
their authenticated identity layer and must not expose branch objects beyond
the lease that owns them.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Generic, Protocol, TypeVar


class Resettable(Protocol):
    """The minimum interface required from a stateful branch."""

    def reset(self) -> None:
        """Clear all state associated with the branch."""


BranchT = TypeVar("BranchT", bound=Resettable)


class SessionRegistryError(RuntimeError):
    """Base exception for session-registry lifecycle errors."""


class SessionCapacityError(SessionRegistryError):
    """Raised when the registry is full of unexpired sessions."""


class SessionBusyError(SessionRegistryError):
    """Raised when a caller tries to reset or close an active session."""


@dataclass
class _Session(Generic[BranchT]):
    branch: BranchT
    last_used: float
    active_leases: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionRegistry(Generic[BranchT]):
    """Manage separate stateful branches for `(tenant_id, session_id)` keys.

    Each lease holds a per-session lock for its whole duration. This prevents
    concurrent requests from interleaving writes to one branch while allowing
    independent sessions to proceed concurrently. Expired idle sessions are
    removed when the registry is next accessed. Capacity is a hard bound:
    active or unexpired sessions are never silently evicted.

    The registry is process-local and volatile. For multiple workers, use
    worker affinity or an application-owned session service; do not serialize
    live branch objects and assume they are safe to restore.
    """

    def __init__(
        self,
        branch_factory: Callable[[], BranchT],
        *,
        max_sessions: int = 1024,
        ttl_seconds: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(branch_factory):
            raise TypeError("branch_factory must be callable")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
            raise TypeError("max_sessions must be an integer")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be greater than zero")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise TypeError("ttl_seconds must be a finite positive number")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a finite positive number")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._branch_factory = branch_factory
        self._max_sessions = max_sessions
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._sessions: dict[tuple[str, str], _Session[BranchT]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_id(value: str, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
        return value

    def _expire_idle_sessions(self, now: float) -> None:
        expired = [
            key
            for key, session in self._sessions.items()
            if session.active_leases == 0
            and now - session.last_used >= self._ttl_seconds
        ]
        for key in expired:
            del self._sessions[key]

    @contextmanager
    def lease(self, tenant_id: str, session_id: str) -> Iterator[BranchT]:
        """Yield the branch for one identity pair, serialized per session."""
        tenant = self._validate_id(tenant_id, "tenant_id")
        session_name = self._validate_id(session_id, "session_id")
        key = (tenant, session_name)
        now = self._clock()

        with self._lock:
            self._expire_idle_sessions(now)
            session = self._sessions.get(key)
            if session is None:
                if len(self._sessions) >= self._max_sessions:
                    raise SessionCapacityError(
                        f"session registry is at capacity "
                        f"({self._max_sessions}); retry after sessions expire "
                        "or close an idle session"
                    )
                session = _Session(self._branch_factory(), last_used=now)
                self._sessions[key] = session
            session.active_leases += 1
            session.last_used = now

        try:
            with session.lock:
                yield session.branch
        finally:
            with self._lock:
                session.active_leases -= 1
                session.last_used = self._clock()

    def reset_session(self, tenant_id: str, session_id: str) -> bool:
        """Reset an idle session; return False if it does not exist."""
        key = (
            self._validate_id(tenant_id, "tenant_id"),
            self._validate_id(session_id, "session_id"),
        )
        with self._lock:
            self._expire_idle_sessions(self._clock())
            session = self._sessions.get(key)
            if session is None:
                return False
            if session.active_leases:
                raise SessionBusyError("cannot reset a session with an active lease")
            session.branch.reset()
            session.last_used = self._clock()
            return True

    def close_session(self, tenant_id: str, session_id: str) -> bool:
        """Remove an idle session; return False if it does not exist."""
        key = (
            self._validate_id(tenant_id, "tenant_id"),
            self._validate_id(session_id, "session_id"),
        )
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return False
            if session.active_leases:
                raise SessionBusyError("cannot close a session with an active lease")
            del self._sessions[key]
            return True

    @property
    def session_count(self) -> int:
        """Return the number of registered sessions after expiring idle entries."""
        with self._lock:
            self._expire_idle_sessions(self._clock())
            return len(self._sessions)
