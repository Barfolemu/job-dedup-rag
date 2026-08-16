from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest

manifest_path = Path("data/jobs/manifest.json")
jobs = load_jobs_from_manifest(manifest_path)

print(f"Jobs loaded: {len(jobs)}")

for job in jobs:
    print(f"Job ID: {job['job_id']}")
    print(f"Company: {job['company_name']}")
    print(f"Role: {job['role_title']}")
    print(f"Found by: {job['found_by']}")
    print(f"JD characters: {len(job['job_description'])}")
    print(f"JD preview: {job['job_description'][:100]!r}")
