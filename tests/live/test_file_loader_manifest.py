from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest

from .conftest import requires_private_manifest

MANIFEST_PATH = Path("data/jobs/manifest.json")


@requires_private_manifest
def test_load_jobs_from_private_manifest_parses_all_entries() -> None:
    jobs = load_jobs_from_manifest(MANIFEST_PATH)

    assert jobs

    for job in jobs:
        assert job["job_id"]
        assert job["company_name"]
        assert job["role_title"]
        assert job["found_by"]
        assert job["job_description"]
