
from backend.agent_tools import execute_tool, ACCESS_REQUESTS


def test_employee_can_read_policy():
    result = execute_tool(
        "policy_lookup",
        {"policy_id": "IT-001"},
        "employee"
    )
    assert "content" in result


def test_unknown_policy_rejected():
    result = execute_tool(
        "policy_lookup",
        {"policy_id": "UNKNOWN"},
        "employee"
    )
    assert "error" in result


def test_employee_cannot_submit_access():
    result = execute_tool(
        "submit_access_request",
        {
            "system": "Payroll",
            "business_purpose": "Monthly reporting",
            "access_level": "read",
            "end_date": "2026-12-31"
        },
        "employee"
    )
    assert result["error"] == "Unauthorized"


def test_manager_requires_approval():
    result = execute_tool(
        "submit_access_request",
        {
            "system": "Payroll",
            "business_purpose": "Monthly reporting",
            "access_level": "read",
            "end_date": "2026-12-31"
        },
        "manager"
    )
    assert result["status"] == "approval_required"


def test_manager_cannot_request_admin_access():
    result = execute_tool(
        "submit_access_request",
        {
            "system": "Payroll",
            "business_purpose": "Administrative access",
            "access_level": "admin",
            "end_date": "2026-12-31",
            "approved": True
        },
        "manager"
    )
    assert "error" in result


def test_unknown_tool_rejected():
    result = execute_tool("unknown_tool", {}, "admin")
    assert "error" in result


def test_extra_arguments_rejected():
    result = execute_tool(
        "policy_lookup",
        {"policy_id": "IT-001", "malicious": True},
        "employee"
    )
    assert "error" in result
