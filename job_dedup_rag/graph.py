from langgraph.graph import END, START, StateGraph

from job_dedup_rag.nodes import (
    create_document,
    retrieve_candidates,
    store_document,
    extract_job_features,
)

from job_dedup_rag.state import IngestionState


def build_ingestion_graph():
    graph_builder = StateGraph(IngestionState)

    graph_builder.add_node("extract_job_features", extract_job_features,)
    graph_builder.add_node("create_document", create_document)
    graph_builder.add_node("store_document", store_document)
    graph_builder.add_node("retrieve_candidates", retrieve_candidates)

    graph_builder.add_edge(START, "extract_job_features")
    graph_builder.add_edge("extract_job_features", "create_document")
    graph_builder.add_edge("create_document", "retrieve_candidates")
    graph_builder.add_edge("retrieve_candidates", "store_document")
    graph_builder.add_edge("store_document", END)

    return graph_builder.compile()