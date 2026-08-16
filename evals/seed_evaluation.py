import os
from pathlib import Path

from dotenv import load_dotenv

from job_dedup_rag.evaluation import validate_evaluation_namespace
from job_dedup_rag.file_loader import load_jobs_from_manifest
from job_dedup_rag.nodes import (
    create_document,
    extract_job_features,
    store_document,
)
from job_dedup_rag.state import IngestionState


def main() -> None:
    load_dotenv()

    namespace = validate_evaluation_namespace(os.environ["PINECONE_NAMESPACE"])

    manifest_path = Path("evaluation_data/seed_manifest.json")
    jobs = load_jobs_from_manifest(manifest_path)

    print(f"Evaluation namespace: {namespace}")
    print(f"Seed jobs: {len(jobs)}")

    for job in jobs:
        initial_state: IngestionState = {
            "job": job,
        }

        state_with_features: IngestionState = {
            **initial_state,
            **extract_job_features(initial_state),
        }

        state_with_document: IngestionState = {
            **state_with_features,
            **create_document(state_with_features),
        }

        storage_update = store_document(state_with_document)

        print(f"Stored {job['job_id']}: {storage_update['stored_ids']}")


if __name__ == "__main__":
    main()
