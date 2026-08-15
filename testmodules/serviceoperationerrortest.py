import os
from collections.abc import Callable
from unittest.mock import patch

from langchain_core.documents import Document

from job_dedup_rag.exceptions import (
    ExternalServiceOperationError,
)
from job_dedup_rag.nodes import (
    compare_current_candidate,
    extract_job_features,
    retrieve_candidates,
    store_document,
)
from job_dedup_rag.state import IngestionState


class FakeServiceError(RuntimeError):
    pass


class FakeStructuredModel:
    def invoke(self, messages: object) -> object:
        raise FakeServiceError("Simulated model failure")


class FakeChatOpenAI:
    def __init__(self, **kwargs: object) -> None:
        pass

    def with_structured_output(
        self,
        *args: object,
        **kwargs: object,
    ) -> FakeStructuredModel:
        return FakeStructuredModel()


class FakeVectorStore:
    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        raise FakeServiceError("Simulated retrieval failure")

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str],
    ) -> list[str]:
        raise FakeServiceError("Simulated storage failure")


job_id = "test:service-operation"

base_state: IngestionState = {
    "job": {
        "job_id": job_id,
        "company_name": "Example Company",
        "role_title": "Engineering Manager",
        "found_by": "test",
        "job_description": "Test job description",
    },
}

document = Document(
    page_content="Normalized search text",
    metadata={
        "job_id": job_id,
    },
)

retrieval_state: IngestionState = {
    **base_state,
    "document": document,
}

comparison_state: IngestionState = {
    **base_state,
    "candidates": [
        {
            "document": Document(
                page_content="Candidate search text",
                metadata={
                    "job_id": "test:candidate",
                    "job_description": ("Candidate job description"),
                },
            ),
            "similarity_score": 0.95,
        },
    ],
    "candidate_index": 0,
}


def assert_operation_error(
    operation_call: Callable[[], object],
    expected_operation: str,
) -> None:
    try:
        operation_call()
    except ExternalServiceOperationError as error:
        assert error.operation == expected_operation
        assert error.job_id == job_id
        assert isinstance(error.__cause__, FakeServiceError)

        print("Expected operation error:", error)
    else:
        raise AssertionError(f"{expected_operation} should have failed")


with (
    patch.dict(
        os.environ,
        {"OPENAI_CHAT_MODEL": "test-model"},
    ),
    patch(
        "job_dedup_rag.nodes.ChatOpenAI",
        new=FakeChatOpenAI,
    ),
):
    assert_operation_error(
        lambda: extract_job_features(base_state),
        "feature_extraction",
    )

    assert_operation_error(
        lambda: compare_current_candidate(comparison_state),
        "candidate_comparison",
    )


with patch(
    "job_dedup_rag.nodes.build_vector_store",
    return_value=FakeVectorStore(),
):
    assert_operation_error(
        lambda: retrieve_candidates(retrieval_state),
        "candidate_retrieval",
    )

    assert_operation_error(
        lambda: store_document(retrieval_state),
        "document_storage",
    )


invalid_comparison_state: IngestionState = {
    **base_state,
    "candidates": [
        {
            "document": Document(
                page_content="Candidate search text",
                metadata={
                    "job_id": "test:candidate",
                },
            ),
            "similarity_score": 0.95,
        },
    ],
    "candidate_index": 0,
}

try:
    compare_current_candidate(invalid_comparison_state)
except TypeError as error:
    assert not isinstance(
        error,
        ExternalServiceOperationError,
    )

    print("Expected internal validation error:", error)
else:
    raise AssertionError("Invalid candidate metadata should fail")

print("Service operation error checks passed")
