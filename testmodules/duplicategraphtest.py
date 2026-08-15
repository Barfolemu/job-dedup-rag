from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from job_dedup_rag.graph import build_ingestion_graph


graph = build_ingestion_graph()


job_description = Path(
    "data/jobs/query_cases/affirm-indeed.txt"
).read_text(encoding="utf-8")

initial_state = {
    "job": {
        "job_id": "indeed:test-affirm",
        "company_name": "Affirm",
        "role_title": "Engineering Manager, Developer Environments",
        "found_by": "indeed",
        "job_description": job_description,
    }
}

result = graph.invoke(initial_state)

print(f"Result status: {result['result_status']}")
print(f"Matched job ID: {result.get('matched_job_id')}")
print(f"Candidate index: {result['candidate_index']}")
print(f"Confidence: {result['comparison'].confidence}")
print(f"Explanation: {result['comparison'].explanation}")

assert result["result_status"] == "possible_duplicate"
assert result["matched_job_id"] == "linkedin:4413856088"
assert result["candidate_index"] == 0
assert result["comparison"].is_duplicate
assert "stored_ids" not in result

print("Duplicate graph checks passed")