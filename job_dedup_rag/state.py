from typing import NotRequired, TypedDict

from langchain_core.documents import Document

from job_dedup_rag.models import ExtractedJobFeatures


class JobPosting(TypedDict):
    job_id: str
    company_name: str
    role_title: str
    found_by: str
    job_description: str


class RetrievalCandidate(TypedDict):
    document: Document
    similarity_score: float


class IngestionState(TypedDict):
    job: JobPosting
    extracted_features: NotRequired[ExtractedJobFeatures]
    document: NotRequired[Document]
    candidates: NotRequired[list[RetrievalCandidate]]
    stored_ids: NotRequired[list[str]]
