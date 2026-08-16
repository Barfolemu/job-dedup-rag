from time import time

import pytest

from job_dedup_rag.models import ExtractedJobFeatures
from job_dedup_rag.nodes import create_document
from job_dedup_rag.state import IngestionState


def test_create_document_builds_search_text_and_retention_metadata() -> None:
    job_description = (
        "Lead a software engineering team building cloud-native applications on AWS."
    )

    features = ExtractedJobFeatures(
        company_name="Example Company",
        role_title="Software Engineering Manager",
        requisition_id=None,
        location="Remote, United States",
        workplace_type="remote",
        employment_type="full_time",
        seniority_level="Manager",
        team_or_domain="Cloud Engineering",
        responsibilities=[
            "Lead a software engineering team",
            "Guide cloud architecture",
        ],
        required_qualifications=["Software engineering leadership experience"],
        preferred_qualifications=[],
        technologies=["AWS"],
    )

    state: IngestionState = {
        "job": {
            "job_id": "test:document-node",
            "company_name": "Example Company",
            "role_title": "Software Engineering Manager",
            "found_by": "manual",
            "job_description": job_description,
        },
        "extracted_features": features,
    }

    before_creation_epoch = int(time())
    node_update = create_document(state)
    after_creation_epoch = int(time())

    document = node_update["document"]

    assert document.page_content != job_description
    assert "Company: Example Company" in document.page_content
    assert "Team or domain: Cloud Engineering" in document.page_content
    assert document.metadata["job_description"] == job_description
    assert document.metadata["technologies"] == ["AWS"]

    indexed_at_epoch = document.metadata["indexed_at_epoch"]
    expires_at_epoch = document.metadata["expires_at_epoch"]

    assert isinstance(indexed_at_epoch, int)
    assert isinstance(expires_at_epoch, int)
    assert before_creation_epoch <= indexed_at_epoch <= after_creation_epoch
    assert expires_at_epoch - indexed_at_epoch == 90 * 24 * 60 * 60


def test_create_document_rejects_empty_job_description() -> None:
    state: IngestionState = {
        "job": {
            "job_id": "test:empty-description",
            "company_name": "Example Company",
            "role_title": "Software Engineering Manager",
            "found_by": "manual",
            "job_description": "   ",
        },
        "extracted_features": ExtractedJobFeatures(
            company_name=None,
            role_title=None,
            requisition_id=None,
            location=None,
            workplace_type="unspecified",
            employment_type="unspecified",
            seniority_level=None,
            team_or_domain=None,
            responsibilities=[],
            required_qualifications=[],
            preferred_qualifications=[],
            technologies=[],
        ),
    }

    with pytest.raises(ValueError, match="Job description cannot be empty"):
        create_document(state)
