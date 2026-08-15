from job_dedup_rag.models import ExtractedJobFeatures
from job_dedup_rag.nodes import create_document
from job_dedup_rag.state import IngestionState


job_description = (
    "Lead a software engineering team building cloud-native "
    "applications on AWS."
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
    required_qualifications=[
        "Software engineering leadership experience",
    ],
    preferred_qualifications=[],
    technologies=[
        "AWS",
    ],
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

node_update = create_document(state)
document = node_update["document"]

print(f"Original state keys: {list(state.keys())}")
print(f"Node update keys: {list(node_update.keys())}")
print(f"Searchable content:\n{document.page_content}")
print(f"\nMetadata: {document.metadata}")

assert document.page_content != job_description
assert "Company: Example Company" in document.page_content
assert "Team or domain: Cloud Engineering" in document.page_content
assert document.metadata["job_description"] == job_description
assert document.metadata["technologies"] == ["AWS"]

print("\nDocument node checks passed")