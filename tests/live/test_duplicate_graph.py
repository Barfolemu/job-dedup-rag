from pathlib import Path

from job_dedup_rag.evaluation import skip_document_storage
from job_dedup_rag.graph import build_ingestion_graph

from .conftest import requires_private_query_cases

QUERY_PATH = Path("data/jobs/query_cases/affirm-indeed.txt")


@requires_private_query_cases
def test_full_graph_flags_cross_source_repost_as_possible_duplicate() -> None:
    """Requires the private Affirm/Indeed query case and matching production record.

    Expected result never reaches the store_document node, but the graph is
    still built with a no-write storage node: a failed assertion here must
    not be able to write to production data.
    """
    graph = build_ingestion_graph(storage_node=skip_document_storage)

    initial_state = {
        "job": {
            "job_id": "indeed:test-affirm",
            "company_name": "Affirm",
            "role_title": "Engineering Manager, Developer Environments",
            "found_by": "indeed",
            "job_description": QUERY_PATH.read_text(encoding="utf-8"),
        }
    }

    result = graph.invoke(initial_state)

    assert result["result_status"] == "possible_duplicate"
    assert result["matched_job_id"] == "linkedin:4413856088"
    assert result["candidate_index"] == 0
    assert result["comparison"].is_duplicate
    assert "stored_ids" not in result
