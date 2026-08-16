from typing import Literal

from langgraph.graph import END, START, StateGraph

from job_dedup_rag.nodes import (
    advance_candidate,
    check_existing_job_id,
    compare_current_candidate,
    create_document,
    extract_job_features,
    mark_already_exists,
    mark_possible_duplicate,
    retrieve_candidates,
    store_document,
)
from job_dedup_rag.retry_policy import (
    EXTERNAL_SERVICE_RETRY_POLICY,
)
from job_dedup_rag.state import IngestionState


def build_ingestion_graph():
    graph_builder = StateGraph(IngestionState)

    graph_builder.add_node(
        "extract_job_features",
        extract_job_features,
        retry_policy=EXTERNAL_SERVICE_RETRY_POLICY,
    )
    graph_builder.add_node("create_document", create_document)
    graph_builder.add_node(
        "store_document",
        store_document,
        retry_policy=EXTERNAL_SERVICE_RETRY_POLICY,
    )
    graph_builder.add_node(
        "retrieve_candidates",
        retrieve_candidates,
        retry_policy=EXTERNAL_SERVICE_RETRY_POLICY,
    )
    graph_builder.add_node(
        "compare_current_candidate",
        compare_current_candidate,
        retry_policy=EXTERNAL_SERVICE_RETRY_POLICY,
    )
    graph_builder.add_node(
        "advance_candidate",
        advance_candidate,
    )
    graph_builder.add_node(
        "mark_possible_duplicate",
        mark_possible_duplicate,
    )
    graph_builder.add_node(
        "check_existing_job_id",
        check_existing_job_id,
        retry_policy=EXTERNAL_SERVICE_RETRY_POLICY,
    )
    graph_builder.add_node(
        "mark_already_exists",
        mark_already_exists,
    )

    graph_builder.add_edge(START, "check_existing_job_id")

    graph_builder.add_conditional_edges(
        "check_existing_job_id",
        route_after_exact_id_check,
        {
            "already_exists": "mark_already_exists",
            "extract": "extract_job_features",
        },
    )
    graph_builder.add_edge("extract_job_features", "create_document")
    graph_builder.add_edge("create_document", "retrieve_candidates")

    graph_builder.add_conditional_edges(
        "retrieve_candidates",
        route_after_retrieval,
        {
            "compare": "compare_current_candidate",
            "store": "store_document",
        },
    )

    graph_builder.add_conditional_edges(
        "compare_current_candidate",
        route_after_comparison,
        {
            "duplicate": "mark_possible_duplicate",
            "advance": "advance_candidate",
            "store": "store_document",
        },
    )

    graph_builder.add_edge(
        "advance_candidate",
        "compare_current_candidate",
    )

    graph_builder.add_edge(
        "mark_possible_duplicate",
        END,
    )

    graph_builder.add_edge("store_document", END)
    graph_builder.add_edge("mark_already_exists", END)

    compiled_graph = graph_builder.compile()

    return compiled_graph.with_config(
        {
            "run_name": "job_dedup_ingestion",
            "tags": [
                "job-dedup",
                "ingestion-workflow",
            ],
        }
    )


def route_after_retrieval(
    state: IngestionState,
) -> Literal["compare", "store"]:
    if state["candidates"]:
        return "compare"

    return "store"


def route_after_comparison(
    state: IngestionState,
) -> Literal["duplicate", "advance", "store"]:
    if state["comparison"].is_duplicate:
        return "duplicate"

    next_index = state["candidate_index"] + 1

    if next_index < len(state["candidates"]):
        return "advance"

    return "store"


def route_after_exact_id_check(
    state: IngestionState,
) -> Literal["already_exists", "extract"]:
    exact_id_exists = state.get("exact_id_exists")

    if exact_id_exists is True:
        return "already_exists"

    if exact_id_exists is False:
        return "extract"

    raise ValueError("Exact-ID routing requires an exact_id_exists result")
