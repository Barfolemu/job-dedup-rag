import pytest

from job_dedup_rag.nodes import check_existing_job_id, mark_already_exists
from job_dedup_rag.state import IngestionState

EXISTING_STATE: IngestionState = {
    "job": {
        "job_id": "linkedin:4413856088",
        "company_name": "Affirm",
        "role_title": "Manager, Software Engineering",
        "found_by": "linkedin",
        "job_description": "Test description",
    },
}

MISSING_STATE: IngestionState = {
    "job": {
        "job_id": "test:exact-id-that-should-not-exist",
        "company_name": "Example Company",
        "role_title": "Engineering Manager",
        "found_by": "test",
        "job_description": "Test description",
    },
}


def test_check_existing_job_id_distinguishes_existing_from_missing() -> None:
    """Requires the production namespace to already contain linkedin:4413856088."""
    existing_update = check_existing_job_id(EXISTING_STATE)
    missing_update = check_existing_job_id(MISSING_STATE)

    assert existing_update["exact_id_exists"] is True
    assert missing_update["exact_id_exists"] is False


def test_mark_already_exists_requires_a_confirmed_exact_id() -> None:
    existing_checked_state: IngestionState = {
        **EXISTING_STATE,
        **check_existing_job_id(EXISTING_STATE),
    }

    already_exists_update = mark_already_exists(existing_checked_state)

    assert already_exists_update == {
        "matched_job_id": "linkedin:4413856088",
        "result_status": "already_exists",
    }

    missing_checked_state: IngestionState = {
        **MISSING_STATE,
        **check_existing_job_id(MISSING_STATE),
    }

    with pytest.raises(ValueError):
        mark_already_exists(missing_checked_state)
