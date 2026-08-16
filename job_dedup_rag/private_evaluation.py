from pathlib import Path

from job_dedup_rag.evaluation import (
    run_evaluation_case,
    skip_document_storage,
    summarize_evaluations,
)
from job_dedup_rag.file_loader import load_evaluation_cases
from job_dedup_rag.graph import build_ingestion_graph


def main():
    manifest_path = Path("data/jobs/evaluation_manifest.json")
    evaluation_cases = load_evaluation_cases(manifest_path)

    graph = build_ingestion_graph(
        storage_node=skip_document_storage,
    )

    print("Private live evaluation")
    print("Pinecone writes: disabled")
    print(f"Cases: {len(evaluation_cases)}")

    observations = []

    for evaluation_case in evaluation_cases:
        observation = run_evaluation_case(
            graph,
            evaluation_case,
        )
        observations.append(observation)

        status_correct = observation.actual_status == observation.expected_status
        match_correct = observation.expected_match_job_id == observation.matched_job_id

        print(f"\nCase: {observation.case_id}")
        print(f"Expected status: {observation.expected_status}")
        print(f"Actual status: {observation.actual_status}")
        print(f"Status correct: {status_correct}")
        print(f"Expected matched job ID: {observation.expected_match_job_id}")
        print(f"Actual matched job ID: {observation.matched_job_id}")
        print(f"Matched job correct: {match_correct}")
        print(f"Retrieved job IDs: {observation.retrieved_job_ids}")
        print(f"Comparisons: {observation.comparison_count}")
        print(f"Latency seconds: {observation.latency_seconds:.3f}")

    summary = summarize_evaluations(observations)

    print("\nEvaluation summary")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
