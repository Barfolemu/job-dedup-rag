import json
from pathlib import Path

from job_dedup_rag.evaluation import EvaluationCase


def load_evaluation_cases(
    manifest_path: Path,
) -> list[EvaluationCase]:
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_entries = json.loads(manifest_text)

    evaluation_cases: list[EvaluationCase] = []

    for entry in manifest_entries:
        job_description_path = manifest_path.parent / entry["file"]
        job_description = job_description_path.read_text(encoding="utf-8")

        evaluation_case = EvaluationCase(
            case_id=entry["case_id"],
            job={
                "job_id": entry["job_id"],
                "company_name": entry["company_name"],
                "role_title": entry["role_title"],
                "found_by": entry["found_by"],
                "job_description": job_description,
            },
            expected_status=entry["expected_status"],
            expected_match_job_id=entry.get("expected_matched_job_id"),
        )

        evaluation_cases.append(evaluation_case)

    return evaluation_cases
