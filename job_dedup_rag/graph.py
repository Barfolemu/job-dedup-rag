from langgraph.graph import END, START, StateGraph

from job_dedup_rag.nodes import (
    advance_candidate,
    compare_current_candidate,
    create_document,
    extract_job_features,
    mark_possible_duplicate,
    retrieve_candidates,
    store_document,
)
from typing import Literal
from job_dedup_rag.state import IngestionState


def build_ingestion_graph():
    graph_builder = StateGraph(IngestionState)

    graph_builder.add_node("extract_job_features", extract_job_features,)
    graph_builder.add_node("create_document", create_document)
    graph_builder.add_node("store_document", store_document)
    graph_builder.add_node("retrieve_candidates", retrieve_candidates)
    graph_builder.add_node("compare_current_candidate", compare_current_candidate, )
    graph_builder.add_node("advance_candidate", advance_candidate,)
    graph_builder.add_node("mark_possible_duplicate", mark_possible_duplicate, )

    graph_builder.add_edge(START, "extract_job_features")
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

    return graph_builder.compile()


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