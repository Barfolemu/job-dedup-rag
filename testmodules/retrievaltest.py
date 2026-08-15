from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest
from job_dedup_rag.nodes import extract_job_features
from job_dedup_rag.state import IngestionState
from job_dedup_rag.vector_store import build_vector_store

manifest_path = Path("data/jobs/manifest.json")
jobs = load_jobs_from_manifest(manifest_path)
query_job = jobs[0]

initial_state: IngestionState = {
    "job": query_job,
}

extraction_update = extract_job_features(initial_state)
features = extraction_update["extracted_features"]
query_text = features.to_search_text()

vector_store = build_vector_store()

results = vector_store.similarity_search_with_score(
    query_text,
    k=5,
)

ranked_results = sorted(
    results,
    key=lambda result: result[1],
    reverse=True,
)

print(f"Query job: {query_job['company_name']} — {query_job['role_title']}")

for position, (document, score) in enumerate(
    ranked_results,
    start=1,
):
    print(
        f"\n{position}. "
        f"{document.metadata['company_name']} — "
        f"{document.metadata['role_title']}"
    )
    print(f"   Job ID: {document.metadata['job_id']}")
    print(f"   Similarity score: {score:.4f}")

assert ranked_results
assert ranked_results[0][0].metadata["job_id"] == query_job["job_id"]

scores = [score for _, score in ranked_results]

assert scores == sorted(scores, reverse=True)

print("\nDirect structured retrieval checks passed")
