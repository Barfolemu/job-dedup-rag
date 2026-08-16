from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from job_dedup_rag.nodes import retrieve_candidates
from job_dedup_rag.state import IngestionState


def test_retrieve_candidates_raises_on_same_id_semantic_result() -> None:
    current_job_id = "linkedin:4413856088"

    state: IngestionState = {
        "job": {
            "job_id": current_job_id,
            "company_name": "Affirm",
            "role_title": "Manager, Software Engineering",
            "found_by": "linkedin",
            "job_description": "Test description",
        },
        "document": Document(page_content="Normalized test search text"),
    }

    same_id_candidate = Document(
        page_content="Stored normalized search text",
        metadata={"job_id": current_job_id},
    )

    class FakeVectorStore:
        def similarity_search_with_score(
            self, query: str, k: int
        ) -> list[tuple[Document, float]]:
            assert query == "Normalized test search text"
            assert k == 5

            return [(same_id_candidate, 0.99)]

    with (
        patch(
            "job_dedup_rag.nodes.build_vector_store",
            return_value=FakeVectorStore(),
        ),
        pytest.raises(RuntimeError, match=current_job_id),
    ):
        retrieve_candidates(state)
