from __future__ import annotations

__all__ = [
    "CrashFrontierError",
    "InfeasibleCrossSectionError",
    "MissingForwardError",
    "NoEligibleExpiryError",
    "NoSuchDateError",
    "SchemaError",
    "SolverFailure",
]


class CrashFrontierError(Exception):
    """Base exception for crashfrontier."""


class SchemaError(CrashFrontierError):
    """Input columns or values do not match the expected schema."""


class NoSuchDateError(CrashFrontierError):
    """Requested observation date is not present in the option source."""


class NoEligibleExpiryError(CrashFrontierError):
    """No expiry satisfies the selection rules for the requested date."""


class MissingForwardError(CrashFrontierError):
    """No source in the forward chain returned a value."""


class InfeasibleCrossSectionError(CrashFrontierError):
    """Quotes admit no feasible measure under the chosen model."""

    def __init__(
        self,
        message: str = "",
        *,
        date=None,
        expiry=None,
        quote_set: str | None = None,
        gamma: int | None = None,
        support_cap: float | None = None,
        status: str | None = None,
    ) -> None:
        self.date = date
        self.expiry = expiry
        self.quote_set = quote_set
        self.gamma = None if gamma is None else int(gamma)
        self.support_cap = support_cap
        self.status = status
        super().__init__(message)


class SolverFailure(CrashFrontierError):
    """An LP or conic solver failed to return a usable solution."""
