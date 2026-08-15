from typing import NotRequired, TypedDict, Literal

from langchain_core.documents import Document

from job_dedup_rag.models import ExtractedJobFeatures, DuplicateComparison


class JobPosting(TypedDict):
    job_id: str
    company_name: str
    role_title: str
    found_by: str
    job_description: str


class RetrievalCandidate(TypedDict):
    document: Document
    similarity_score: float


class RetrievalUpdate(TypedDict):
    candidates: list[RetrievalCandidate]
    candidate_index: int

class PossibleDuplicateUpdate(TypedDict):
    matched_job_id: str
    result_status: Literal["possible_duplicate"]


class IngestionState(TypedDict):
    job: JobPosting
    extracted_features: NotRequired[ExtractedJobFeatures]
    document: NotRequired[Document]
    candidates: NotRequired[list[RetrievalCandidate]]
    candidate_index: NotRequired[int]
    comparison: NotRequired[DuplicateComparison]
    matched_job_id: NotRequired[str]
    result_status: NotRequired[
        Literal[
            "possible_duplicate",
            "stored",
        ]
    ]
    stored_ids: NotRequired[list[str]]

class StoredUpdate(TypedDict):
    stored_ids: list[str]
    result_status: Literal["stored"]
