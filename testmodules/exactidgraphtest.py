from job_dedup_rag.graph import build_ingestion_graph
from job_dedup_rag.state import IngestionState

initial_state: IngestionState = {
    "job": {
        "job_id": "linkedin:4413856088",
        "company_name": "Affirm",
        "role_title": "Manager, Software Engineering",
        "found_by": "linkedin",
        "job_description": ("This description must never reach feature extraction."),
    },
}

graph = build_ingestion_graph()
result = graph.invoke(initial_state)

assert result["result_status"] == "already_exists"
assert result["matched_job_id"] == "linkedin:4413856088"
assert result["exact_id_exists"] is True
assert "extracted_features" not in result
assert "document" not in result
assert "candidates" not in result
assert "stored_ids" not in result

print("Result status:", result["result_status"])
print("Matched job ID:", result["matched_job_id"])
print("Feature extraction bypassed:", "extracted_features" not in result)
