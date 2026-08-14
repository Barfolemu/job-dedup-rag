from langchain_core.documents import Document
from job_dedup_rag.vector_store import build_vector_store
from job_dedup_rag.state import IngestionState


def create_document(state: IngestionState) -> dict[str, Document]:
    job = state["job"]
    job_description = job["job_description"].strip()

    if not job_description:
        raise ValueError("Job description cannot be empty")

    document = Document(
        page_content=job_description,
        metadata={
            "job_id": job["job_id"],
            "company_name": job["company_name"],
            "role_title": job["role_title"],
            "found_by": job["found_by"],
        },
    )

    return {"document": document}


def store_document(state: IngestionState) -> dict[str, list[str]]:
    document = state["document"]
    job_id = state["job"]["job_id"]

    vector_store = build_vector_store()

    stored_ids = vector_store.add_documents(
        documents=[document],
        ids=[job_id],
    )

    return {"stored_ids": stored_ids}