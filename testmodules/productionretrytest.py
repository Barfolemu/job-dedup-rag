from unittest.mock import patch

from pinecone.exceptions import PineconeException

from job_dedup_rag.exceptions import (
    ExternalServiceOperationError,
)
from job_dedup_rag.graph import build_ingestion_graph
from job_dedup_rag.state import (
    IngestionState,
)


class FakePineconeError(PineconeException):
    def __init__(self, status: int) -> None:
        self.status = status


attempts = 0


def flaky_job_id_exists(
    job_id: str,
) -> bool:
    global attempts

    assert job_id == "test:production-retry"

    attempts += 1

    if attempts < 3:
        raise FakePineconeError(503)

    return True


initial_state: IngestionState = {
    "job": {
        "job_id": "test:production-retry",
        "company_name": "Example Company",
        "role_title": "Engineering Manager",
        "found_by": "test",
        "job_description": "This must not reach extraction.",
    },
}

with patch(
    "job_dedup_rag.nodes.job_id_exists",
    new=flaky_job_id_exists,
):
    graph = build_ingestion_graph()
    result = graph.invoke(initial_state)

assert attempts == 3
assert result["result_status"] == "already_exists"
assert result["matched_job_id"] == "test:production-retry"
assert "extracted_features" not in result

print("Production exact-ID attempts:", attempts)
print("Final result status:", result["result_status"])
print("Extraction bypassed:", "extracted_features" not in result)
failed_attempts = 0


def always_failing_job_id_exists(
    job_id: str,
) -> bool:
    global failed_attempts

    assert job_id == "test:production-retry"

    failed_attempts += 1
    raise FakePineconeError(503)


with patch(
    "job_dedup_rag.nodes.job_id_exists",
    new=always_failing_job_id_exists,
):
    failing_graph = build_ingestion_graph()

    try:
        failing_graph.invoke(initial_state)
    except ExternalServiceOperationError as error:
        assert error.operation == "exact_id_check"
        assert error.job_id == "test:production-retry"
        assert isinstance(error.__cause__, FakePineconeError)

        print("Expected exhausted error:", error)
    else:
        raise AssertionError("Exhausted exact-ID retries should raise")

assert failed_attempts == 3

print("Exhausted attempts:", failed_attempts)
