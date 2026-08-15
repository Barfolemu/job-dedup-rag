from job_dedup_rag.nodes import (
    check_existing_job_id,
    mark_already_exists,
)
from job_dedup_rag.state import IngestionState

existing_state: IngestionState = {
    "job": {
        "job_id": "linkedin:4413856088",
        "company_name": "Affirm",
        "role_title": "Manager, Software Engineering",
        "found_by": "linkedin",
        "job_description": "Test description",
    },
}

missing_state: IngestionState = {
    "job": {
        "job_id": "test:exact-id-that-should-not-exist",
        "company_name": "Example Company",
        "role_title": "Engineering Manager",
        "found_by": "test",
        "job_description": "Test description",
    },
}

existing_update = check_existing_job_id(existing_state)
missing_update = check_existing_job_id(missing_state)

assert existing_update["exact_id_exists"] is True
assert missing_update["exact_id_exists"] is False

print("Existing ID check:", existing_update)
print("Missing ID check:", missing_update)

existing_checked_state: IngestionState = {
    **existing_state,
    **existing_update,
}

already_exists_update = mark_already_exists(existing_checked_state)

assert already_exists_update == {
    "matched_job_id": "linkedin:4413856088",
    "result_status": "already_exists",
}

missing_checked_state: IngestionState = {
    **missing_state,
    **missing_update,
}

try:
    mark_already_exists(missing_checked_state)
except ValueError as error:
    print("Expected invariant error:", error)
else:
    raise AssertionError("Missing exact ID should not produce already_exists")

print("Already-exists result:", already_exists_update)
