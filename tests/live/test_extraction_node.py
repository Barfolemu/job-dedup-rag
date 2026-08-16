from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest
from job_dedup_rag.nodes import extract_job_features
from job_dedup_rag.state import IngestionState

from .conftest import requires_private_manifest

MANIFEST_PATH = Path("data/jobs/manifest.json")


@requires_private_manifest
def test_extraction_produces_populated_features_for_first_manifest_job() -> None:
    jobs = load_jobs_from_manifest(MANIFEST_PATH)
    query_job = jobs[0]

    initial_state: IngestionState = {"job": query_job}
    extraction_update = extract_job_features(initial_state)
    features = extraction_update["extracted_features"]

    assert features.company_name is not None
    assert features.role_title is not None
    assert len(features.responsibilities) > 0
