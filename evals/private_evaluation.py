from pathlib import Path

from evals.evaluation_runner import run_evaluation_manifest


def main() -> None:
    run_evaluation_manifest(
        Path("data/jobs/evaluation_manifest.json"),
        title="Private live evaluation",
    )


if __name__ == "__main__":
    main()
