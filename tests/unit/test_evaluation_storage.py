from job_dedup_rag.evaluation import skip_document_storage
from job_dedup_rag.state import IngestionState


def test_skip_document_storage_never_writes() -> None:
    state: IngestionState = {
        "job": {
            "job_id": "test:no-write",
            "company_name": "Example Company",
            "role_title": "Engineering Manager",
            "found_by": "manual",
            "job_description": "Example description",
        }
    }

    update = skip_document_storage(state)

    assert update == {
        "stored_ids": [],
        "result_status": "stored",
    }
