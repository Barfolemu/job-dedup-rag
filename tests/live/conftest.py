import os
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

from job_dedup_rag.evaluation import validate_evaluation_namespace


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/live/" in str(item.path).replace("\\", "/"):
            item.add_marker(pytest.mark.live)


PRIVATE_MANIFEST = Path("data/jobs/manifest.json")
PRIVATE_QUERY_CASES = Path("data/jobs/query_cases")

requires_private_manifest = pytest.mark.skipif(
    not PRIVATE_MANIFEST.exists(),
    reason=(
        "requires the private, gitignored data/jobs/manifest.json corpus "
        "(and its production Pinecone records) that only exists locally"
    ),
)

requires_private_query_cases = pytest.mark.skipif(
    not PRIVATE_QUERY_CASES.exists(),
    reason=(
        "requires the private, gitignored data/jobs/query_cases directory "
        "(and its production Pinecone records) that only exists locally"
    ),
)


def require_mutating_namespace() -> str:
    """Refuse to proceed unless PINECONE_NAMESPACE is a guarded evaluation-* value.

    Rejects an unset/empty namespace, the production `structured-v1` namespace,
    and any other namespace that doesn't start with the `evaluation-` prefix
    (matching the same rule the evaluation runners already enforce).
    """
    load_dotenv()
    raw_namespace = os.environ.get("PINECONE_NAMESPACE", "")

    try:
        return validate_evaluation_namespace(raw_namespace)
    except ValueError as error:
        pytest.fail(
            "Refusing to run a mutating live test outside a guarded "
            f"evaluation-* namespace: {error}. Set PINECONE_NAMESPACE, e.g. "
            "PINECONE_NAMESPACE=evaluation-live-tests.",
            pytrace=False,
        )


def unique_test_job_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4()}"
