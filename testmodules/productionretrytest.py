from unittest.mock import patch

from pinecone.exceptions import PineconeException

from job_dedup_rag.graph import build_ingestion_graph
from job_dedup_rag.state import (
    ExactIdCheckUpdate,
    IngestionState,
)


class FakePineconeError(PineconeException):
    def __init__(self, status: int) -> None:
        self.status = status


attempts = 0


def flaky_exact_id_check(
    state: IngestionState,
) -> ExactIdCheckUpdate:
    global attempts

    attempts += 1

    if attempts < 3:
        raise FakePineconeError(503)

    return {
        "exact_id_exists": True,
    }


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
    "job_dedup_rag.graph.check_existing_job_id",
    new=flaky_exact_id_check,
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
