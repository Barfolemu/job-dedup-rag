from dataclasses import dataclass

import pytest

from job_dedup_rag.integrations.jobtracker import map_job_item_to_request

VALID_ITEM_KWARGS = {
    "job_id": "manual:abc123",
    "company_name": "Example Company",
    "role_title": "Engineering Manager",
    "found_by": "manual",
    "job_description": "Lead a platform engineering team.",
}


@dataclass
class FakeJobTrackerItem:
    """Stands in for jobtracker.models.JobItem without importing it.

    Carries extra fields (status, notes) beyond the five the protocol
    requires, demonstrating that map_job_item_to_request only depends on
    structural compatibility for those five — not on the real JobItem type.
    """

    job_id: str
    company_name: str
    role_title: str
    found_by: str
    job_description: str
    status: str = "new"
    notes: str = ""


def test_map_job_item_to_request_maps_the_five_required_fields() -> None:
    item = FakeJobTrackerItem(**VALID_ITEM_KWARGS)

    request = map_job_item_to_request(item)

    assert request.job_id == item.job_id
    assert request.company_name == item.company_name
    assert request.role_title == item.role_title
    assert request.found_by == item.found_by
    assert request.job_description == item.job_description


@pytest.mark.parametrize(
    "field",
    ["job_id", "company_name", "role_title", "found_by", "job_description"],
)
def test_map_job_item_to_request_rejects_blank_required_fields(field: str) -> None:
    kwargs = {**VALID_ITEM_KWARGS, field: "   "}
    item = FakeJobTrackerItem(**kwargs)

    with pytest.raises(ValueError):
        map_job_item_to_request(item)


class _MissingJobDescriptionItem:
    """An object structurally missing one of the five required attributes."""

    def __init__(self) -> None:
        self.job_id = VALID_ITEM_KWARGS["job_id"]
        self.company_name = VALID_ITEM_KWARGS["company_name"]
        self.role_title = VALID_ITEM_KWARGS["role_title"]
        self.found_by = VALID_ITEM_KWARGS["found_by"]
        # job_description intentionally omitted


def test_map_job_item_to_request_raises_clear_value_error_for_missing_attribute() -> (
    None
):
    item = _MissingJobDescriptionItem()

    with pytest.raises(ValueError, match="job_description"):
        map_job_item_to_request(item)
