from pathlib import Path

from job_dedup_rag.nodes import (
    compare_current_candidate,
    create_document,
    extract_job_features,
    retrieve_candidates,
)
from job_dedup_rag.state import IngestionState

from .conftest import requires_private_query_cases

QUERY_PATH = Path("data/jobs/query_cases/affirm-indeed.txt")


@requires_private_query_cases
def test_cross_source_repost_is_flagged_as_duplicate() -> None:
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
    state_with_candidates: IngestionState = {
        **state_with_document,
        **retrieve_candidates(state_with_document),
    }

    comparison_update = compare_current_candidate(state_with_candidates)
    comparison = comparison_update["comparison"]

    candidate = state_with_candidates["candidates"][0]
    candidate_document = candidate["document"]

    assert candidate_document.metadata["job_id"] == "linkedin:4413856088"
    assert comparison.is_duplicate is True
    assert len(comparison.matching_signals) > 0
