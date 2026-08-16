from pathlib import Path

from langchain_core.documents import Document

from job_dedup_rag.nodes import compare_current_candidate
from job_dedup_rag.state import IngestionState

cedar_description = Path(
    "evaluation_data/cases/cedar-cloud-platform-manager.txt"
).read_text(encoding="utf-8")

redwood_description = Path(
    "evaluation_data/seeds/redwood-platform-manager.txt"
).read_text(encoding="utf-8")

state: IngestionState = {
    "job": {
        "job_id": "synthetic-query:cedar-cloud-platform-manager",
        "company_name": "Cedar Works",
        "role_title": "Engineering Manager, Cloud Platform",
        "found_by": "manual",
        "job_description": cedar_description,
    },
    "candidates": [
        {
            "document": Document(
                page_content=redwood_description,
                metadata={
                    "job_id": "synthetic:redwood-platform-manager-v1",
                    "company_name": "Redwood Analytics",
                    "role_title": "Engineering Manager, Cloud Platform",
                    "found_by": "manual",
                    "job_description": redwood_description,
                },
            ),
            "similarity_score": 0.99,
        }
    ],
    "candidate_index": 0,
}

comparison = compare_current_candidate(state)["comparison"]

print(comparison.model_dump_json(indent=2))

assert comparison.is_duplicate is False

print("\nDifferent-company comparison regression check passed")
