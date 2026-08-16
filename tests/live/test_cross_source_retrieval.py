from pathlib import Path

from job_dedup_rag.nodes import (
    create_document,
    extract_job_features,
    retrieve_candidates,
)
from job_dedup_rag.state import IngestionState

from .conftest import requires_private_query_cases

QUERY_PATH = Path("data/jobs/query_cases/affirm-indeed.txt")


@requires_private_query_cases
def test_cross_source_retrieval_ranks_matching_record_first() -> None:
    """Requires the private Affirm/Indeed query case and matching production record."""
    initial_state: IngestionState = {
        "job": {
            "job_id": "indeed:test-affirm",
            "company_name": "Affirm",
            "role_title": "Engineering Manager, Developer Environments",
            "found_by": "indeed",
            "job_description": QUERY_PATH.read_text(encoding="utf-8"),
        }
    }

    state_with_features: IngestionState = {
        **initial_state,
        **extract_job_features(initial_state),
    }
    state_with_document: IngestionState = {
        **state_with_features,
        **create_document(state_with_features),
    }

    candidate_update = retrieve_candidates(state_with_document)
    candidates = candidate_update["candidates"]

    assert candidates, (
        "No candidates returned. Ensure structured-v1 was populated by "
        "running: uv run main.py"
    )

    top_candidate = candidates[0]
    top_document = top_candidate["document"]

    scores = [candidate["similarity_score"] for candidate in candidates]
    assert scores == sorted(scores, reverse=True)

    assert top_document.metadata["job_id"] == "linkedin:4413856088"
    assert "job_description" in top_document.metadata
