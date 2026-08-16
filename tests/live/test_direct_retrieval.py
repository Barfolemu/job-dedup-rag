from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest
from job_dedup_rag.nodes import extract_job_features
from job_dedup_rag.vector_store import build_vector_store

from .conftest import requires_private_manifest

MANIFEST_PATH = Path("data/jobs/manifest.json")


@requires_private_manifest
def test_direct_similarity_search_ranks_matching_record_first() -> None:
    jobs = load_jobs_from_manifest(MANIFEST_PATH)
    query_job = jobs[0]

    extraction_update = extract_job_features({"job": query_job})
    query_text = extraction_update["extracted_features"].to_search_text()

    vector_store = build_vector_store()
    results = vector_store.similarity_search_with_score(query_text, k=5)

    ranked_results = sorted(results, key=lambda result: result[1], reverse=True)

    assert ranked_results
    assert ranked_results[0][0].metadata["job_id"] == query_job["job_id"]

    scores = [score for _, score in ranked_results]
    assert scores == sorted(scores, reverse=True)
