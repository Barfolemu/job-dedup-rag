from job_dedup_rag.graph import build_ingestion_graph
from job_dedup_rag.state import IngestionState
from job_dedup_rag.vector_store import build_vector_store

initial_state: IngestionState = {
    "job": {
        "job_id": "test:ingestion-graph",
        "company_name": "Example Company",
        "role_title": "Software Engineering Manager",
        "found_by": "manual",
        "job_description": (
            "Lead a software engineering team building cloud-native "
            "applications on AWS."
        ),
    }
}

graph = build_ingestion_graph()
result = graph.invoke(initial_state)

print(f"Final state keys: {list(result.keys())}")
print(f"Stored IDs: {result['stored_ids']}")
print(f"Document metadata: {result['document'].metadata}")

vector_store = build_vector_store()
vector_store.delete(ids=result["stored_ids"])

print(f"Deleted test records: {result['stored_ids']}")

assert result["result_status"] == "stored"
assert result["stored_ids"] == ["test:ingestion-graph"]
assert result["candidate_index"] == len(result["candidates"]) - 1
assert not result["comparison"].is_duplicate
