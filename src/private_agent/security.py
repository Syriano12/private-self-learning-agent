from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ApprovalRequest:
    action: str
    reason: str
    risk: str
    expected_result: str
    approved: bool = False


class ApprovalGate:
    SENSITIVE = {"send_message", "publish", "payment", "delete_data", "account_change", "external_irreversible"}

    def check(self, action: str, *, approved: bool = False) -> ApprovalRequest | None:
        if action not in self.SENSITIVE:
            return None
        return ApprovalRequest(action, "Sensitive external action", "high", "Action completes only after explicit approval", approved)

    def allow(self, request: ApprovalRequest) -> bool:
        return request.approved
