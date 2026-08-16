from pathlib import Path

from job_dedup_rag.file_loader import load_jobs_from_manifest
from job_dedup_rag.graph import build_ingestion_graph
from job_dedup_rag.state import IngestionState


def main():
    manifest_path = Path("data/jobs/manifest.json")
    jobs = load_jobs_from_manifest(manifest_path)
    graph = build_ingestion_graph()

    for job in jobs:
        initial_state: IngestionState = {
            "job": job,
        }

        result = graph.invoke(initial_state)

        result_status = result["result_status"]

        if result_status == "already_exists":
            print(
                f"Already exists {job['job_id']}: "
                f"{job['company_name']} — {job['role_title']}"
            )
            print(f"Existing job ID: {result['matched_job_id']}")

        elif result_status == "possible_duplicate":
            comparison = result["comparison"]

            print(f"Possible duplicate: {job['company_name']} — {job['role_title']}")
            print(f"Matched job ID: {result['matched_job_id']}")
            print(f"Confidence: {comparison.confidence:.2f}")
            print(f"Explanation: {comparison.explanation}")

        elif result_status == "stored":
            print(
                f"Stored {job['job_id']}: {job['company_name']} — {job['role_title']}"
            )
            print(f"Pinecone IDs: {result['stored_ids']}")

        else:
            raise ValueError(f"Unexpected ingestion result status: {result_status}")


if __name__ == "__main__":
    main()
