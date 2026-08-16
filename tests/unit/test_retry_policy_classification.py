from openai import APIConnectionError, APIStatusError
from pinecone.exceptions import PineconeException

from job_dedup_rag.exceptions import ExternalServiceOperationError
from job_dedup_rag.retry_policy import is_transient_external_error


class FakeOpenAIConnectionError(APIConnectionError):
    def __init__(self) -> None:
        pass


class FakeOpenAIStatusError(APIStatusError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakePineconeError(PineconeException):
    def __init__(self, status: int | None) -> None:
        self.status = status


def test_transient_direct_errors_are_classified_correctly() -> None:
    assert is_transient_external_error(FakeOpenAIConnectionError())

    assert is_transient_external_error(FakeOpenAIStatusError(429))
    assert is_transient_external_error(FakeOpenAIStatusError(503))

    assert not is_transient_external_error(FakeOpenAIStatusError(400))
    assert not is_transient_external_error(FakeOpenAIStatusError(401))

    assert is_transient_external_error(FakePineconeError(429))
    assert is_transient_external_error(FakePineconeError(503))

    assert not is_transient_external_error(FakePineconeError(400))
    assert not is_transient_external_error(FakePineconeError(401))

    assert not is_transient_external_error(ValueError("Invalid state"))
    assert not is_transient_external_error(TypeError("Invalid contract"))
    assert not is_transient_external_error(RuntimeError("Invariant violated"))

    assert is_transient_external_error(ConnectionError("Temporary connection failure"))
    assert is_transient_external_error(TimeoutError("Temporary timeout"))


def test_wrapped_error_classification_follows_cause_chain() -> None:
    try:
        raise FakePineconeError(503)
    except FakePineconeError as cause:
        try:
            raise ExternalServiceOperationError(
                operation="candidate_retrieval",
                job_id="test:retryable",
            ) from cause
        except ExternalServiceOperationError as error:
            wrapped_transient_error = error

    assert is_transient_external_error(wrapped_transient_error)
    assert "candidate_retrieval" in str(wrapped_transient_error)
    assert "test:retryable" in str(wrapped_transient_error)

    try:
        raise ValueError("Invalid candidate metadata")
    except ValueError as cause:
        try:
            raise ExternalServiceOperationError(
                operation="candidate_retrieval",
                job_id="test:not-retryable",
            ) from cause
        except ExternalServiceOperationError as error:
            wrapped_validation_error = error

    assert not is_transient_external_error(wrapped_validation_error)
