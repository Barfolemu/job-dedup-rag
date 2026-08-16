from typing import Protocol

from job_dedup_rag.boundary import DeduplicationRequest

_REQUIRED_FIELDS = (
    "job_id",
    "company_name",
    "role_title",
    "found_by",
    "job_description",
)


class JobTrackerLikeItem(Protocol):
    """Structural shape of the five JobTracker JobItem fields this boundary needs.

    Limited to exactly what deduplication requires — not the full JobItem
    shape (which also has status, job_url, date_found, date_posted, notes).
    This module never imports jobtracker.models.JobItem; any object with
    these five string attributes satisfies the protocol.
    """

    job_id: str
    company_name: str
    role_title: str
    found_by: str
    job_description: str


def map_job_item_to_request(item: JobTrackerLikeItem) -> DeduplicationRequest:
    """Map a JobTracker-shaped item onto the RAG request contract.

    Raises ValueError for both missing and blank required fields: a missing
    attribute is caught here and re-raised naming the field, and a present
    but blank field is rejected by DeduplicationRequest's own validation
    (pydantic.ValidationError is a ValueError subclass).
    """
    mapped_fields: dict[str, str] = {}

    for field_name in _REQUIRED_FIELDS:
        try:
            mapped_fields[field_name] = getattr(item, field_name)
        except AttributeError as error:
            raise ValueError(
                f"JobTracker item is missing required field {field_name!r}"
            ) from error

    return DeduplicationRequest(**mapped_fields)
