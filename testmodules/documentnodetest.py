from time import time

from job_dedup_rag.models import ExtractedJobFeatures
from job_dedup_rag.nodes import create_document
from job_dedup_rag.state import IngestionState

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

before_creation_epoch = int(time())

node_update = create_document(state)

after_creation_epoch = int(time())

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

indexed_at_epoch = document.metadata["indexed_at_epoch"]
expires_at_epoch = document.metadata["expires_at_epoch"]

assert isinstance(indexed_at_epoch, int)
assert isinstance(expires_at_epoch, int)

assert before_creation_epoch <= indexed_at_epoch <= after_creation_epoch

assert expires_at_epoch - indexed_at_epoch == 90 * 24 * 60 * 60

print("\nDocument node checks passed")
print("Indexed at epoch:", indexed_at_epoch)
print("Expires at epoch:", expires_at_epoch)
print(
    "Retention days:",
    (expires_at_epoch - indexed_at_epoch) // (24 * 60 * 60),
)
