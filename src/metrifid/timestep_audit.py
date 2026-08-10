"""Model-specific MuJoCo timestep fidelity audit over the accepted comparator.

The audit evaluates every declared candidate timestep against one compiled reference
model, one existing canonical workload, and predeclared joint tolerances. It never
assumes monotonic behaviour and never performs a binary search: every candidate is
attempted independently and the recommendation is the largest candidate supported by an
unbroken ascending prefix of within-tolerance completed results.

Both roles compile the exact admitted user model. A candidate timestep is applied in
memory to the candidate compiled model's `mjModel.opt.timestep`; the user's MJCF is never
serialized, rewritten, or copied. `mjOption` is runtime simulation state and does not
affect compilation, so the two roles differ only by the timestep under audit.

The reference timestep is the comparison reference only. It is not asserted to be
physically correct or a ground truth.
"""

from __future__ import annotations

from ._audit_config import (
    OPERATION,
    AuditAbort,
    AuditConfig,
)

# Internal entry points that existing tests import from this module. The redundant aliases mark
# them as deliberate re-exports so the split changed no import path.
from ._audit_config import _parse_config as _parse_config
from ._audit_config import _steps_per_control as _steps_per_control
from ._audit_config import _tree_digest as _tree_digest
from ._audit_execution import AuditRunResult, audit_configuration_file
from ._audit_reporting import _STATUS_CLASSIFICATION as _STATUS_CLASSIFICATION
from ._audit_reporting import (
    AUDIT_SCHEMA,
    CLAIM_BOUNDARY,
    RECOMMENDATION_POLICY,
    candidate_token,
)
from ._audit_reporting import INCONCLUSIVE as INCONCLUSIVE
from ._audit_reporting import OUTSIDE as OUTSIDE
from ._audit_reporting import REFUSED as REFUSED
from ._audit_reporting import WITHIN as WITHIN
from ._audit_reporting import _candidate_row as _candidate_row
from ._audit_reporting import _recommendation as _recommendation
from ._audit_reporting import _render_markdown as _render_markdown

__all__ = [
    "AUDIT_SCHEMA",
    "INCONCLUSIVE",
    "OUTSIDE",
    "REFUSED",
    "WITHIN",
    "AuditAbort",
    "AuditConfig",
    "AuditRunResult",
    "CLAIM_BOUNDARY",
    "OPERATION",
    "RECOMMENDATION_POLICY",
    "audit_configuration_file",
    "candidate_token",
]
