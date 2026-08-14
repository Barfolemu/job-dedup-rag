from typing import NotRequired, TypedDict

from langchain_core.documents import Document


class JobPosting(TypedDict):
    job_id: str
    company_name: str
    role_title: str
    found_by: str
    job_description: str


class IngestionState(TypedDict):
    job: JobPosting
    document: NotRequired[Document]
    stored_ids: NotRequired[list[str]]