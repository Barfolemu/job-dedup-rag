from openai import APIConnectionError, APIStatusError
from pinecone.exceptions import PineconeException

from job_dedup_rag.retry_policy import (
    is_transient_external_error,
)


class FakeOpenAIConnectionError(APIConnectionError):
    def __init__(self) -> None:
        pass


class FakeOpenAIStatusError(APIStatusError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakePineconeError(PineconeException):
    def __init__(self, status: int | None) -> None:
        self.status = status


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

print("Transient retry classification checks passed")
