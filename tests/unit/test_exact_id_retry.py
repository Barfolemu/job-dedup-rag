from unittest.mock import patch

import pytest
from pinecone.exceptions import PineconeException

from job_dedup_rag.exceptions import ExternalServiceOperationError
from job_dedup_rag.graph import build_ingestion_graph
from job_dedup_rag.state import IngestionState

JOB_ID = "test:production-retry"


class FakePineconeError(PineconeException):
    def __init__(self, status: int) -> None:
        self.status = status


INITIAL_STATE: IngestionState = {
    "job": {
        "job_id": JOB_ID,
        "company_name": "Example Company",
        "role_title": "Engineering Manager",
        "found_by": "test",
        "job_description": "This must not reach extraction.",
    },
}


def test_exact_id_check_retries_transient_errors_then_succeeds() -> None:
    attempts = 0

    def flaky_job_id_exists(job_id: str) -> bool:
        nonlocal attempts

        assert job_id == JOB_ID
        attempts += 1

        if attempts < 3:
            raise FakePineconeError(503)

        return True

    with patch("job_dedup_rag.nodes.job_id_exists", new=flaky_job_id_exists):
        graph = build_ingestion_graph()
        result = graph.invoke(INITIAL_STATE)

    assert attempts == 3
    assert result["result_status"] == "already_exists"
    assert result["matched_job_id"] == JOB_ID
    assert "extracted_features" not in result


def test_exact_id_check_exhausts_retries_and_raises() -> None:
    failed_attempts = 0

    def always_failing_job_id_exists(job_id: str) -> bool:
        nonlocal failed_attempts

        assert job_id == JOB_ID
        failed_attempts += 1
        raise FakePineconeError(503)

    with patch("job_dedup_rag.nodes.job_id_exists", new=always_failing_job_id_exists):
        failing_graph = build_ingestion_graph()

        with pytest.raises(ExternalServiceOperationError) as excinfo:
            failing_graph.invoke(INITIAL_STATE)

    error = excinfo.value

    assert error.operation == "exact_id_check"
    assert error.job_id == JOB_ID
    assert isinstance(error.__cause__, FakePineconeError)
    assert failed_attempts == 3
