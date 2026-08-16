from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

from job_dedup_rag.graph import IngestionGraph, build_ingestion_graph
from job_dedup_rag.state import IngestionState, JobPosting

ResultStatus = Literal["already_exists", "possible_duplicate", "stored"]

_REQUIRED_FIELDS = (
    "job_id",
    "company_name",
    "role_title",
    "found_by",
    "job_description",
)


def _stripped_nonblank(value: str, field_name: str) -> str:
    stripped = value.strip()

    if not stripped:
        raise ValueError(f"{field_name} must not be blank")

    return stripped


class DeduplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    company_name: str
    role_title: str
    found_by: str
    job_description: str

    @field_validator(*_REQUIRED_FIELDS)
    @classmethod
    def _reject_blank(cls, value: str, info: ValidationInfo) -> str:
        return _stripped_nonblank(value, info.field_name)


class DeduplicationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ResultStatus
    incoming_job_id: str
    matched_job_id: str | None = None
    stored_ids: list[str] | None = None

    @field_validator("incoming_job_id")
    @classmethod
    def _validate_incoming_job_id(cls, value: str) -> str:
        return _stripped_nonblank(value, "incoming_job_id")

    @field_validator("matched_job_id")
    @classmethod
    def _validate_matched_job_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return _stripped_nonblank(value, "matched_job_id")

    @field_validator("stored_ids")
    @classmethod
    def _validate_stored_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        return [_stripped_nonblank(item, "stored_ids entry") for item in value]

    @model_validator(mode="after")
    def _validate_status_specific_shape(self) -> "DeduplicationResponse":
        if self.status == "already_exists":
            if self.matched_job_id != self.incoming_job_id:
                raise ValueError(
                    "already_exists requires matched_job_id to equal incoming_job_id"
                )

            if self.stored_ids is not None:
                raise ValueError("already_exists must not include stored_ids")

        elif self.status == "possible_duplicate":
            if not self.matched_job_id:
                raise ValueError("possible_duplicate requires a matched_job_id")

            if self.matched_job_id == self.incoming_job_id:
                raise ValueError(
                    "possible_duplicate requires matched_job_id to differ from "
                    "incoming_job_id"
                )

            if self.stored_ids is not None:
                raise ValueError("possible_duplicate must not include stored_ids")

        elif self.status == "stored":
            if self.matched_job_id is not None:
                raise ValueError("stored must not include a matched_job_id")

            if self.stored_ids != [self.incoming_job_id]:
                raise ValueError(
                    "stored requires stored_ids to equal [incoming_job_id], "
                    "matching the current one-document storage contract"
                )

        return self


def _run_graph(
    request: DeduplicationRequest,
    graph: IngestionGraph,
) -> DeduplicationResponse:
    job: JobPosting = {
        "job_id": request.job_id,
        "company_name": request.company_name,
        "role_title": request.role_title,
        "found_by": request.found_by,
        "job_description": request.job_description,
    }
    initial_state: IngestionState = {"job": job}

    result = graph.invoke(initial_state)

    return DeduplicationResponse(
        status=result.get("result_status"),
        incoming_job_id=request.job_id,
        matched_job_id=result.get("matched_job_id"),
        stored_ids=result.get("stored_ids"),
    )


# Compiled once at import time and reused across calls. Building the graph
# has no I/O or env-var side effects of its own (those happen lazily inside
# node functions when invoked), so this is safe at module import.
_PRODUCTION_GRAPH = build_ingestion_graph()


def deduplicate_job(request: DeduplicationRequest) -> DeduplicationResponse:
    """The single public entry point for the JobTracker integration boundary.

    Runs the real production graph topology against the configured Pinecone
    namespace. There is no parameter for injecting a different graph or
    storage node here — that seam exists only as the module-private
    `_run_graph`, used by tests.
    """
    return _run_graph(request, _PRODUCTION_GRAPH)
