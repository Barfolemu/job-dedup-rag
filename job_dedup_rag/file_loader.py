import json
from pathlib import Path

from job_dedup_rag.state import JobPosting


def load_jobs_from_manifest(manifest_path: Path) -> list[JobPosting]:
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_entries = json.loads(manifest_text)

    jobs: list[JobPosting] = []

    for entry in manifest_entries:
        job_description_path = manifest_path.parent / entry["file"]
        job_description = job_description_path.read_text(encoding="utf-8")

        job: JobPosting = {
            "job_id": entry["job_id"],
            "company_name": entry["company_name"],
            "role_title": entry["role_title"],
            "found_by": entry["found_by"],
            "job_description": job_description,
        }

        jobs.append(job)

    return jobs
