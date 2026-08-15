from langgraph.types import RetryPolicy
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from pinecone.exceptions import PineconeException

OPENAI_RETRYABLE_STATUS_CODES = {
    408,
    409,
    429,
}

PINECONE_RETRYABLE_STATUS_CODES = {
    429,
}


def is_transient_external_error(
    error: Exception,
) -> bool:

    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    if isinstance(
        error,
        (APIConnectionError, APITimeoutError),
    ):
        return True

    if isinstance(error, APIStatusError):
        status_code = error.status_code

        return status_code in OPENAI_RETRYABLE_STATUS_CODES or status_code >= 500

    if isinstance(error, PineconeException):
        status_code = getattr(error, "status", None)

        return isinstance(status_code, int) and (
            status_code in PINECONE_RETRYABLE_STATUS_CODES or status_code >= 500
        )

    return False


EXTERNAL_SERVICE_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=4.0,
    jitter=True,
    retry_on=is_transient_external_error,
)
