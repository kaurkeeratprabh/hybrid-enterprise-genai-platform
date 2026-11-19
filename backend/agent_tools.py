
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


# Strict schemas: reject unexpected input fields

class PolicyLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=3, max_length=30)


class AccessRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str = Field(min_length=2, max_length=100)
    business_purpose: str = Field(min_length=10, max_length=500)
    access_level: Literal["read", "write", "admin"]
    end_date: str = Field(min_length=10, max_length=10)
    approved: bool = False


# Demo policy data
POLICIES = {
    "IT-001": "Company devices must be used primarily for approved business activities.",
    "HR-005": "Employees should follow the official annual leave request and approval process.",
    "FIN-001": "Business expenses must follow the company's reimbursement policy."
}

ACCESS_REQUESTS = []


def policy_lookup(data: PolicyLookupInput, role: str):
    """Read-only tool."""
    if role not in ["employee", "manager", "admin"]:
        return {"error": "Unauthorized"}

    policy = POLICIES.get(data.policy_id)

    if not policy:
        return {"error": "Policy not found"}

    return {
        "tool": "policy_lookup",
        "read_only": True,
        "policy_id": data.policy_id,
        "content": policy
    }


def submit_access_request(data: AccessRequestInput, role: str):
    """State-changing tool requiring approval."""

    if role not in ["manager", "admin"]:
        return {"error": "Unauthorized"}

    if data.access_level == "admin" and role != "admin":
        return {"error": "Admin role required for privileged access"}

    if not data.approved:
        return {
            "status": "approval_required",
            "message": "Explicit approval is required before submitting this request.",
            "action": "submit_access_request"
        }

    try:
        datetime.strptime(data.end_date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid end_date. Use YYYY-MM-DD."}

    request = {
        "request_id": f"ACCESS-{len(ACCESS_REQUESTS) + 1:03d}",
        "system": data.system,
        "business_purpose": data.business_purpose,
        "access_level": data.access_level,
        "end_date": data.end_date,
        "submitted_by": role,
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }

    ACCESS_REQUESTS.append(request)

    return {
        "status": "submitted",
        "request": request
    }


# Registered tools
TOOLS = {
    "policy_lookup": {
        "input_schema": PolicyLookupInput,
        "handler": policy_lookup,
        "read_only": True
    },
    "submit_access_request": {
        "input_schema": AccessRequestInput,
        "handler": submit_access_request,
        "read_only": False
    }
}


def execute_tool(name: str, arguments: dict, role: str):
    tool = TOOLS.get(name)

    if tool is None:
        return {"error": "Unknown tool"}

    try:
        validated = tool["input_schema"].model_validate(arguments)
    except Exception as exc:
        return {"error": "Invalid tool arguments", "details": str(exc)}

    return tool["handler"](validated, role)
