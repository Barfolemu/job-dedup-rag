from langchain_core.documents import Document

from job_dedup_rag.evaluation import (
    EvaluationCase,
    run_evaluation_case,
)
from job_dedup_rag.state import IngestionState


class FakeGraph:
    def invoke(
        self,
        state: IngestionState,
    ) -> IngestionState:
        assert state["job"]["job_id"] == "indeed:test-query"

        return {
            **state,
            "candidates": [
                {
                    "document": Document(
                        page_content="Stored candidate",
                        metadata={
                            "job_id": "linkedin:test-match",
                        },
                    ),
                    "similarity_score": 0.95,
                }
            ],
            "candidate_index": 0,
            "matched_job_id": "linkedin:test-match",
            "result_status": "possible_duplicate",
        }


evaluation_case = EvaluationCase(
    case_id="cross-source-test",
    job={
        "job_id": "indeed:test-query",
        "company_name": "Example Company",
        "role_title": "Engineering Manager",
        "found_by": "indeed",
        "job_description": "Example job description",
    },
    expected_status="possible_duplicate",
    expected_match_job_id="linkedin:test-match",
)

observation = run_evaluation_case(
    FakeGraph(),
    evaluation_case,
)

assert observation.case_id == "cross-source-test"
assert observation.actual_status == "possible_duplicate"
assert observation.matched_job_id == "linkedin:test-match"
assert observation.retrieved_job_ids == ["linkedin:test-match"]
assert observation.comparison_count == 1
assert observation.latency_seconds >= 0.0

print("Evaluation runner checks passed")
