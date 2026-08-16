from job_dedup_rag.evaluation import skip_document_storage
from job_dedup_rag.graph import build_ingestion_graph
from job_dedup_rag.state import IngestionState


def test_exact_id_match_bypasses_extraction_and_retrieval() -> None:
    """Requires the production namespace to already contain linkedin:4413856088.

    Expected result never reaches the store_document node, but the graph is
    still built with a no-write storage node: a failed assertion here must
    not be able to write to production data.
    """
    initial_state: IngestionState = {
        "job": {
            "job_id": "linkedin:4413856088",
            "company_name": "Affirm",
            "role_title": "Manager, Software Engineering",
            "found_by": "linkedin",
            "job_description": "This description must never reach feature extraction.",
        },
    }

    graph = build_ingestion_graph(storage_node=skip_document_storage)
    result = graph.invoke(initial_state)

    assert result["result_status"] == "already_exists"
    assert result["matched_job_id"] == "linkedin:4413856088"
    assert result["exact_id_exists"] is True
    assert "extracted_features" not in result
    assert "document" not in result
    assert "candidates" not in result
    assert "stored_ids" not in result
