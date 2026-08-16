import os
from pathlib import Path

from dotenv import load_dotenv

from job_dedup_rag.evaluation import validate_evaluation_namespace
from job_dedup_rag.evaluation_runner import run_evaluation_manifest


def main() -> None:
    load_dotenv()

    namespace = validate_evaluation_namespace(os.environ["PINECONE_NAMESPACE"])

    print(f"Evaluation namespace: {namespace}")

    run_evaluation_manifest(
        Path("evaluation_data/evaluation_manifest.json"),
        title="Synthetic evaluation",
    )


if __name__ == "__main__":
    main()
