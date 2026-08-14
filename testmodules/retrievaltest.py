from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest
from job_dedup_rag.vector_store import build_vector_store


manifest_path = Path("data/jobs/manifest.json")
jobs = load_jobs_from_manifest(manifest_path)

query_job = jobs[0]
vector_store = build_vector_store()

reformatted_query = " ".join(
    query_job["job_description"].split()
)

results = vector_store.similarity_search_with_score(
    reformatted_query,
    k=5,
)

print(
    f"Query job: {query_job['company_name']} — "
    f"{query_job['role_title']}\n"
)

for rank, (document, score) in enumerate(results, start=1):
    print(
        f"{rank}. {document.metadata['company_name']} — "
        f"{document.metadata['role_title']}"
    )
    print(f"   Job ID: {document.metadata['job_id']}")
    print(f"   Similarity score: {score:.4f}")