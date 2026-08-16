from pathlib import Path

from evals.file_loader import load_evaluation_cases


def test_load_evaluation_cases_parses_manifest_and_description(tmp_path: Path) -> None:
    description_path = tmp_path / "job.txt"
    description_path.write_text("Example job description", encoding="utf-8")

    manifest_path = tmp_path / "evaluation_manifest.json"
    manifest_path.write_text(
        """
[
  {
    "case_id": "example-case",
    "file": "job.txt",
    "job_id": "test:example",
    "company_name": "Example Company",
    "role_title": "Engineering Manager",
    "found_by": "manual",
    "expected_status": "stored",
    "expected_matched_job_id": null
  }
]
""",
        encoding="utf-8",
    )

    cases = load_evaluation_cases(manifest_path)

    assert len(cases) == 1

    evaluation_case = cases[0]

    assert evaluation_case.case_id == "example-case"
    assert evaluation_case.job["job_id"] == "test:example"
    assert evaluation_case.job["job_description"] == "Example job description"
    assert evaluation_case.expected_status == "stored"
    assert evaluation_case.expected_match_job_id is None
