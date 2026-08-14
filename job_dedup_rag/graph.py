from langgraph.graph import END, START, StateGraph

from job_dedup_rag.nodes import create_document, store_document
from job_dedup_rag.state import IngestionState


def build_ingestion_graph():
    graph_builder = StateGraph(IngestionState)

    graph_builder.add_node("create_document", create_document)
    graph_builder.add_node("store_document", store_document)

    graph_builder.add_edge(START, "create_document")
    graph_builder.add_edge("create_document", "store_document")
    graph_builder.add_edge("store_document", END)

    return graph_builder.compile()