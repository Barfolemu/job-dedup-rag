from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest
from job_dedup_rag.nodes import extract_job_features
from job_dedup_rag.state import IngestionState

manifest_path = Path("data/jobs/manifest.json")
jobs = load_jobs_from_manifest(manifest_path)
query_job = jobs[0]

initial_state: IngestionState = {
    "job": query_job,
}

extraction_update = extract_job_features(initial_state)
features = extraction_update["extracted_features"]

print(f"Source job: {query_job['company_name']} — {query_job['role_title']}\n")
print(features.model_dump_json(indent=2))

assert features.company_name is not None
assert features.role_title is not None
assert len(features.responsibilities) > 0

print("\nExtraction node checks passed")
