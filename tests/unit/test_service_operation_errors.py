import os
from typing import ClassVar
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from job_dedup_rag.exceptions import ExternalServiceOperationError
from job_dedup_rag.nodes import (
    compare_current_candidate,
    extract_job_features,
    retrieve_candidates,
    store_document,
)
from job_dedup_rag.state import IngestionState

JOB_ID = "test:service-operation"


class FakeServiceError(RuntimeError):
    pass


class FakeStructuredModel:
    invocation_configs: ClassVar[list[dict[str, object]]] = []

    def invoke(
        self, messages: object, config: dict[str, object] | None = None
    ) -> object:
        assert config is not None
        self.invocation_configs.append(config)
        raise FakeServiceError("Simulated model failure")


class FakeChatOpenAI:
    created_models: ClassVar[list[str]] = []

    def __init__(self, **kwargs: object) -> None:
        model = kwargs.get("model")
        max_retries = kwargs.get("max_retries")

        assert isinstance(model, str)
        assert max_retries == 0

        self.created_models.append(model)

    def with_structured_output(
        self, *args: object, **kwargs: object
    ) -> FakeStructuredModel:
        return FakeStructuredModel()


class FakeVectorStore:
    def similarity_search_with_score(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        raise FakeServiceError("Simulated retrieval failure")

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        raise FakeServiceError("Simulated storage failure")


BASE_STATE: IngestionState = {
    "job": {
        "job_id": JOB_ID,
        "company_name": "Example Company",
        "role_title": "Engineering Manager",
        "found_by": "test",
        "job_description": "Test job description",
    },
}

DOCUMENT = Document(
    page_content="Normalized search text",
    metadata={"job_id": JOB_ID},
)

RETRIEVAL_STATE: IngestionState = {**BASE_STATE, "document": DOCUMENT}

COMPARISON_STATE: IngestionState = {
    **BASE_STATE,
    "candidates": [
        {
            "document": Document(
                page_content="Candidate search text",
                metadata={
                    "job_id": "test:candidate",
                    "job_description": "Candidate job description",
                },
            ),
            "similarity_score": 0.95,
        },
    ],
    "candidate_index": 0,
}


@pytest.fixture(autouse=True)
def _reset_fake_call_history() -> None:
    FakeChatOpenAI.created_models.clear()
    FakeStructuredModel.invocation_configs.clear()


def _assert_operation_error(operation_call, expected_operation: str) -> None:
    with pytest.raises(ExternalServiceOperationError) as excinfo:
        operation_call()

    error = excinfo.value
    assert error.operation == expected_operation
    assert error.job_id == JOB_ID
    assert isinstance(error.__cause__, FakeServiceError)


def test_model_and_vector_store_failures_wrap_as_operation_errors() -> None:
    with (
        patch.dict(
            os.environ,
            {
                "OPENAI_EXTRACTION_MODEL": "test-extraction-model",
                "OPENAI_COMPARISON_MODEL": "test-comparison-model",
            },
        ),
        patch("job_dedup_rag.nodes.load_dotenv", return_value=None),
        patch("job_dedup_rag.nodes.ChatOpenAI", new=FakeChatOpenAI),
    ):
        _assert_operation_error(
            lambda: extract_job_features(BASE_STATE),
            "feature_extraction",
        )
        _assert_operation_error(
            lambda: compare_current_candidate(COMPARISON_STATE),
            "candidate_comparison",
        )

    with patch(
        "job_dedup_rag.nodes.build_vector_store", return_value=FakeVectorStore()
    ):
        _assert_operation_error(
            lambda: retrieve_candidates(RETRIEVAL_STATE),
            "candidate_retrieval",
        )
        _assert_operation_error(
            lambda: store_document(RETRIEVAL_STATE),
            "document_storage",
        )


def test_invalid_candidate_metadata_raises_internal_type_error_not_wrapped() -> None:
    invalid_comparison_state: IngestionState = {
        **BASE_STATE,
        "candidates": [
            {
                "document": Document(
                    page_content="Candidate search text",
                    metadata={"job_id": "test:candidate"},
                ),
                "similarity_score": 0.95,
            },
        ],
        "candidate_index": 0,
    }

    with pytest.raises(TypeError) as excinfo:
        compare_current_candidate(invalid_comparison_state)

    assert not isinstance(excinfo.value, ExternalServiceOperationError)


def test_model_invocations_use_configured_models_and_trace_metadata() -> None:
    with (
        patch.dict(
            os.environ,
            {
                "OPENAI_EXTRACTION_MODEL": "test-extraction-model",
                "OPENAI_COMPARISON_MODEL": "test-comparison-model",
            },
        ),
        patch("job_dedup_rag.nodes.load_dotenv", return_value=None),
        patch("job_dedup_rag.nodes.ChatOpenAI", new=FakeChatOpenAI),
    ):
        with pytest.raises(ExternalServiceOperationError):
            extract_job_features(BASE_STATE)
        with pytest.raises(ExternalServiceOperationError):
            compare_current_candidate(COMPARISON_STATE)

    assert FakeChatOpenAI.created_models == [
        "test-extraction-model",
        "test-comparison-model",
    ]

    assert [
        config["run_name"] for config in FakeStructuredModel.invocation_configs
    ] == [
        "feature_extraction_model",
        "duplicate_comparison_model",
    ]

    extraction_metadata = FakeStructuredModel.invocation_configs[0]["metadata"]
    comparison_metadata = FakeStructuredModel.invocation_configs[1]["metadata"]

    assert isinstance(extraction_metadata, dict)
    assert isinstance(comparison_metadata, dict)
    assert extraction_metadata["model_name"] == "test-extraction-model"
    assert comparison_metadata["model_name"] == "test-comparison-model"
    assert comparison_metadata["candidate_index"] == 0
