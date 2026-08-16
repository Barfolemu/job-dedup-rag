from job_dedup_rag.models import ExtractedJobFeatures


def test_to_search_text_formats_all_sections() -> None:
    features = ExtractedJobFeatures(
        company_name="Example Company",
        role_title="Software Engineering Manager",
        requisition_id=None,
        location="Remote, United States",
        workplace_type="remote",
        employment_type="unspecified",
        seniority_level="Manager",
        team_or_domain="Developer Productivity",
        responsibilities=[
            "Lead a software engineering team",
            "Improve developer productivity",
        ],
        required_qualifications=[
            "Software engineering leadership experience",
        ],
        preferred_qualifications=[],
        technologies=[
            "AWS",
            "Kubernetes",
        ],
    )

    search_text = features.to_search_text()

    assert "Company: Example Company" in search_text
    assert "Requisition ID: Unspecified" in search_text
    assert "Preferred qualifications:\n- None stated" in search_text
    assert "- Kubernetes" in search_text
