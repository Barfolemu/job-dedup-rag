from job_dedup_rag.nodes import create_document
from job_dedup_rag.state import IngestionState


state: IngestionState = {
    "job": {
        "job_id": "test:document-node",
        "company_name": "Example Company",
        "role_title": "Software Engineering Manager",
        "found_by": "manual",
        "job_description": (
            "Lead a software engineering team building cloud-native "
            "applications on AWS."
        ),
    }
}

update = create_document(state)
document = update["document"]

print(f"Original state keys: {list(state.keys())}")
print(f"Node update keys: {list(update.keys())}")
print(f"Page content: {document.page_content}")
print(f"Metadata: {document.metadata}")