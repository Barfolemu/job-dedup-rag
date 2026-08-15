from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from pinecone.exceptions import PineconeException

from job_dedup_rag.retry_policy import (
    EXTERNAL_SERVICE_RETRY_POLICY,
)


class RetryTestState(TypedDict):
    request: str
    result: NotRequired[str]


class FakePineconeError(PineconeException):
    def __init__(self, status: int) -> None:
        self.status = status


transient_attempts = 0


def transient_node(
    state: RetryTestState,
) -> dict[str, str]:
    global transient_attempts

    transient_attempts += 1

    if transient_attempts < 3:
        raise FakePineconeError(503)

    return {
        "result": f"Processed {state['request']}",
    }


transient_builder = StateGraph(RetryTestState)

transient_builder.add_node(
    "transient_node",
    transient_node,
    retry_policy=EXTERNAL_SERVICE_RETRY_POLICY,
)

transient_builder.add_edge(START, "transient_node")
transient_builder.add_edge("transient_node", END)

transient_graph = transient_builder.compile()

transient_result = transient_graph.invoke(
    {
        "request": "test request",
    }
)

assert transient_attempts == 3
assert transient_result["result"] == "Processed test request"

print("Transient attempts:", transient_attempts)
print("Transient result:", transient_result["result"])


validation_attempts = 0


def validation_node(
    state: RetryTestState,
) -> dict[str, str]:
    global validation_attempts

    validation_attempts += 1
    raise ValueError(f"Invalid request: {state['request']}")


validation_builder = StateGraph(RetryTestState)

validation_builder.add_node(
    "validation_node",
    validation_node,
    retry_policy=EXTERNAL_SERVICE_RETRY_POLICY,
)

validation_builder.add_edge(START, "validation_node")
validation_builder.add_edge("validation_node", END)

validation_graph = validation_builder.compile()

try:
    validation_graph.invoke(
        {
            "request": "bad request",
        }
    )
except ValueError as error:
    print("Expected validation error:", error)
else:
    raise AssertionError("Validation error should escape the graph")

assert validation_attempts == 1

print("Validation attempts:", validation_attempts)
