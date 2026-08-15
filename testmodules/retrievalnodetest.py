from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest
from job_dedup_rag.nodes import (
    create_document,
    extract_job_features,
    retrieve_candidates,
)
from job_dedup_rag.state import IngestionState

manifest_path = Path("data/jobs/manifest.json")
jobs = load_jobs_from_manifest(manifest_path)
query_job = jobs[0]

initial_state: IngestionState = {
    "job": query_job,
}

state_with_features: IngestionState = {
    **initial_state,
    **extract_job_features(initial_state),
}

state_with_document: IngestionState = {
    **state_with_features,
    **create_document(state_with_features),
}

candidate_update = retrieve_candidates(state_with_document)
candidates = candidate_update["candidates"]

print(f"Query job: {query_job['company_name']} — {query_job['role_title']}")
print(f"Candidates returned: {len(candidates)}")

for position, candidate in enumerate(candidates, start=1):
    document = candidate["document"]
    score = candidate["similarity_score"]

    print(
        f"\n{position}. "
        f"{document.metadata['company_name']} — "
        f"{document.metadata['role_title']}"
    )
    print(f"   Job ID: {document.metadata['job_id']}")
    print(f"   Similarity score: {score:.4f}")

scores = [candidate["similarity_score"] for candidate in candidates]

assert len(candidates) <= 5
assert scores == sorted(scores, reverse=True)
assert all(
    candidate["document"].metadata.get("job_id") != query_job["job_id"]
    for candidate in candidates
)
assert set(candidate_update) == {
    "candidates",
    "candidate_index",
}
assert candidate_update["candidate_index"] == 0

print("\nRetrieval node checks passed")
