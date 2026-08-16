from typing import NotRequired, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from pinecone.exceptions import PineconeException

from job_dedup_rag.retry_policy import EXTERNAL_SERVICE_RETRY_POLICY


class RetryTestState(TypedDict):
    request: str
    result: NotRequired[str]


class FakePineconeError(PineconeException):
    def __init__(self, status: int) -> None:
        self.status = status


def test_transient_node_retries_until_success() -> None:
    attempts = 0

    def transient_node(state: RetryTestState) -> dict[str, str]:
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise FakePineconeError(503)

        return {"result": f"Processed {state['request']}"}

    builder = StateGraph(RetryTestState)
    builder.add_node(
        "transient_node", transient_node, retry_policy=EXTERNAL_SERVICE_RETRY_POLICY
    )
    builder.add_edge(START, "transient_node")
    builder.add_edge("transient_node", END)

    result = builder.compile().invoke({"request": "test request"})

    assert attempts == 3
    assert result["result"] == "Processed test request"


def test_validation_error_is_not_retried() -> None:
    attempts = 0

    def validation_node(state: RetryTestState) -> dict[str, str]:
        nonlocal attempts
        attempts += 1
        raise ValueError(f"Invalid request: {state['request']}")

    builder = StateGraph(RetryTestState)
    builder.add_node(
        "validation_node", validation_node, retry_policy=EXTERNAL_SERVICE_RETRY_POLICY
    )
    builder.add_edge(START, "validation_node")
    builder.add_edge("validation_node", END)

    graph = builder.compile()

    with pytest.raises(ValueError, match="Invalid request: bad request"):
        graph.invoke({"request": "bad request"})

    assert attempts == 1
