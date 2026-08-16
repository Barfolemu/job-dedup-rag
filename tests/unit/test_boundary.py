import pytest

from job_dedup_rag import boundary
from job_dedup_rag.boundary import (
    DeduplicationRequest,
    DeduplicationResponse,
    deduplicate_job,
)
from job_dedup_rag.state import IngestionState

VALID_REQUEST_KWARGS = {
    "job_id": "indeed:test-job",
    "company_name": "Example Company",
    "role_title": "Engineering Manager",
    "found_by": "indeed",
    "job_description": "Lead a platform engineering team.",
}


class FakeGraph:
    def __init__(self, extra_state: dict) -> None:
        self._extra_state = extra_state

    def invoke(self, state: IngestionState) -> IngestionState:
        return {**state, **self._extra_state}


def test_run_graph_maps_already_exists() -> None:
    request = DeduplicationRequest(**VALID_REQUEST_KWARGS)
    graph = FakeGraph(
        {"matched_job_id": request.job_id, "result_status": "already_exists"}
    )

    response = boundary._run_graph(request, graph)

    assert response == DeduplicationResponse(
        status="already_exists",
        incoming_job_id=request.job_id,
        matched_job_id=request.job_id,
        stored_ids=None,
    )


def test_run_graph_maps_possible_duplicate() -> None:
    request = DeduplicationRequest(**VALID_REQUEST_KWARGS)
    graph = FakeGraph(
        {"matched_job_id": "linkedin:match", "result_status": "possible_duplicate"}
    )

    response = boundary._run_graph(request, graph)

    assert response.status == "possible_duplicate"
    assert response.matched_job_id == "linkedin:match"
    assert response.stored_ids is None


def test_run_graph_maps_stored() -> None:
    request = DeduplicationRequest(**VALID_REQUEST_KWARGS)
    graph = FakeGraph({"stored_ids": [request.job_id], "result_status": "stored"})

    response = boundary._run_graph(request, graph)

    assert response.status == "stored"
    assert response.stored_ids == [request.job_id]
    assert response.matched_job_id is None


def test_deduplicate_job_delegates_to_the_module_level_production_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = DeduplicationRequest(**VALID_REQUEST_KWARGS)
    graph = FakeGraph({"stored_ids": [request.job_id], "result_status": "stored"})

    # deduplicate_job() takes no graph/storage_node parameter by design; this
    # module-level singleton is the only seam, and it exists purely so tests
    # never have to hit a real compiled production graph or real APIs.
    monkeypatch.setattr(boundary, "_PRODUCTION_GRAPH", graph)

    response = deduplicate_job(request)

    assert response.status == "stored"
    assert response.stored_ids == [request.job_id]


@pytest.mark.parametrize(
    "malformed_state",
    [
        {"result_status": "stored"},
        {"result_status": "stored", "stored_ids": []},
        {"result_status": "possible_duplicate"},
        {"result_status": "already_exists"},
        {"result_status": "not_a_real_status"},
        {
            "result_status": "stored",
            "stored_ids": ["x"],
            "matched_job_id": "y",
        },
        {"result_status": "already_exists", "matched_job_id": "some-other-id"},
        {
            "result_status": "possible_duplicate",
            "matched_job_id": VALID_REQUEST_KWARGS["job_id"],
        },
        {"result_status": "stored", "stored_ids": ["totally-different-id"]},
        {"result_status": "stored", "stored_ids": ["   "]},
    ],
    ids=[
        "stored-missing-stored-ids",
        "stored-empty-stored-ids",
        "possible-duplicate-missing-matched-id",
        "already-exists-missing-matched-id",
        "unknown-status",
        "stored-with-matched-job-id",
        "already-exists-different-matched-id",
        "possible-duplicate-self-match",
        "stored-unrelated-stored-id",
        "stored-blank-stored-id-entry",
    ],
)
def test_run_graph_rejects_malformed_graph_results(malformed_state: dict) -> None:
    request = DeduplicationRequest(**VALID_REQUEST_KWARGS)
    graph = FakeGraph(malformed_state)

    with pytest.raises(ValueError):
        boundary._run_graph(request, graph)


@pytest.mark.parametrize(
    "field",
    ["job_id", "company_name", "role_title", "found_by", "job_description"],
)
def test_deduplication_request_rejects_blank_required_fields(field: str) -> None:
    kwargs = {**VALID_REQUEST_KWARGS, field: "   "}

    with pytest.raises(ValueError):
        DeduplicationRequest(**kwargs)


def test_deduplication_request_strips_surrounding_whitespace() -> None:
    kwargs = {key: f"  {value}  " for key, value in VALID_REQUEST_KWARGS.items()}

    request = DeduplicationRequest(**kwargs)

    assert request.job_id == VALID_REQUEST_KWARGS["job_id"]
    assert request.company_name == VALID_REQUEST_KWARGS["company_name"]
    assert request.role_title == VALID_REQUEST_KWARGS["role_title"]
    assert request.found_by == VALID_REQUEST_KWARGS["found_by"]
    assert request.job_description == VALID_REQUEST_KWARGS["job_description"]


def test_run_graph_strips_whitespace_from_a_padded_exact_id_lookup() -> None:
    """A job ID like '  linkedin:123  ' must not reach the exact-ID lookup
    with spaces — DeduplicationRequest strips it before the graph ever sees
    the job."""
    padded_kwargs = {**VALID_REQUEST_KWARGS, "job_id": "  linkedin:123  "}
    request = DeduplicationRequest(**padded_kwargs)

    seen_job_ids: list[str] = []

    class RecordingGraph:
        def invoke(self, state: IngestionState) -> IngestionState:
            seen_job_ids.append(state["job"]["job_id"])
            return {
                **state,
                "matched_job_id": state["job"]["job_id"],
                "result_status": "already_exists",
            }

    boundary._run_graph(request, RecordingGraph())

    assert seen_job_ids == ["linkedin:123"]


def test_deduplication_request_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        DeduplicationRequest(**VALID_REQUEST_KWARGS, unexpected="nope")


def test_deduplication_response_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        DeduplicationResponse(
            status="stored",
            incoming_job_id="x",
            stored_ids=["x"],
            unexpected="nope",
        )


def test_deduplication_response_strips_and_rejects_blank_incoming_job_id() -> None:
    response = DeduplicationResponse(
        status="stored",
        incoming_job_id="  job-1  ",
        stored_ids=["  job-1  "],
    )
    assert response.incoming_job_id == "job-1"
    assert response.stored_ids == ["job-1"]

    with pytest.raises(ValueError):
        DeduplicationResponse(status="stored", incoming_job_id="   ", stored_ids=["x"])


def test_deduplication_response_rejects_blank_matched_job_id() -> None:
    with pytest.raises(ValueError):
        DeduplicationResponse(
            status="already_exists",
            incoming_job_id="job-1",
            matched_job_id="   ",
        )


def test_deduplication_response_rejects_blank_stored_id_entry() -> None:
    with pytest.raises(ValueError):
        DeduplicationResponse(
            status="stored",
            incoming_job_id="job-1",
            stored_ids=["   "],
        )
