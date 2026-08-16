from pathlib import Path

from job_dedup_rag.evaluation_runner import run_evaluation_manifest


def main() -> None:
    run_evaluation_manifest(
        Path("data/jobs/evaluation_manifest.json"),
        title="Private live evaluation",
    )


if __name__ == "__main__":
    main()
