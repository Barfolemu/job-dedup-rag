import pytest

from job_dedup_rag.graph import build_ingestion_graph
from job_dedup_rag.state import IngestionState
from job_dedup_rag.vector_store import build_vector_store

from .conftest import require_mutating_namespace, unique_test_job_id

pytestmark = pytest.mark.mutating


def test_full_graph_stores_and_cleans_up_a_new_job() -> None:
    """Writes to and deletes from a guarded evaluation-* namespace only."""
    require_mutating_namespace()

    job_id = unique_test_job_id("test:ingestion-graph")

    initial_state: IngestionState = {
        "job": {
            "job_id": job_id,
            "company_name": "Example Company",
            "role_title": "Software Engineering Manager",
            "found_by": "manual",
            "job_description": (
                "Lead a software engineering team building cloud-native "
                "applications on AWS."
            ),
        }
    }

    graph = build_ingestion_graph()

    try:
        result = graph.invoke(initial_state)

        assert result["result_status"] == "stored"
        assert result["stored_ids"] == [job_id]
        assert result["candidate_index"] == len(result["candidates"]) - 1
        assert not result["comparison"].is_duplicate
    finally:
        # Deleting a non-existent ID is a no-op, so this is safe even if
        # invoke() raised before the job was actually stored.
        build_vector_store().delete(ids=[job_id])
