from pathlib import Path

from job_dedup_rag.nodes import (
    create_document,
    extract_job_features,
    retrieve_candidates,
)
from job_dedup_rag.state import IngestionState


query_path = Path(
    "data/jobs/query_cases/affirm-indeed.txt"
)

initial_state: IngestionState = {
    "job": {
        "job_id": "indeed:test-affirm",
        "company_name": "Affirm",
        "role_title": "Engineering Manager, Developer Environments",
        "found_by": "indeed",
        "job_description": query_path.read_text(encoding="utf-8"),
    }
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

print(
    "Query job: "
    f"{initial_state['job']['company_name']} — "
    f"{initial_state['job']['role_title']}"
)
print(f"Query source: {initial_state['job']['found_by']}")
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

if not candidates:
    raise AssertionError(
        "No candidates returned. Ensure structured-v1 was populated "
        "by running: uv run main.py"
    )

top_candidate = candidates[0]
top_document = top_candidate["document"]

scores = [
    candidate["similarity_score"]
    for candidate in candidates
]

assert scores == sorted(scores, reverse=True)

assert top_document.metadata["job_id"] == "linkedin:4413856088"
assert "job_description" in top_document.metadata

print("\nCross-source duplicate retrieval checks passed")