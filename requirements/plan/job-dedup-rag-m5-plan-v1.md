# Finish job-dedup-rag: structural cleanup + Milestone 5 integration boundary

## Context

This repo was built as a learning project (LangGraph + RAG) with heavy help from
OpenAI's assistant. That assistant wrote a handoff document at
`requirements/job-dedup-rag-claude-handoff-m5.md` describing two remaining chunks
of work: (1) a structural cleanup pass, since evaluation/dev-test code is
currently mixed into the application package, and (2) "Milestone 5" — a stable
integration boundary so this project can eventually be called from Ashley's
`jobtracker` app (a separate AWS SAM/Lambda project on GitHub at
`Barfolemu/jobtracker`).

I reviewed the actual source (`job_dedup_rag/`, `testmodules/`, `evaluation_data/`,
`main.py`, `pyproject.toml`) against the handoff doc's claims and confirmed it's
accurate: `ruff check`, `ruff format --check`, and `compileall` are all currently
clean, and the 5-case synthetic evaluation manifest matches what the doc
describes. I also pulled the `jobtracker` repo's `models.py`, `api.py`, `db.py`,
and `gmail_agent.py` to ground the integration boundary in the real `JobItem`
shape and jobtracker's existing dedup behavior (today: exact `job_id` match only,
via a DynamoDB conditional-put and a pre-check in the Gmail ingestion agent — no
cross-source semantic dedup exists there yet, which is exactly the gap this
project fills).

Per Ashley's decision, this round of work stays entirely inside `job-dedup-rag`.
It will not touch the `jobtracker` repo, its AWS stack, or decide a transport
(HTTP/Lambda-layer/vendoring) — that's explicitly deferred to a later,
separately-scoped milestone.

The work happens in two reviewable phases, committed separately, on a new branch
`feature/structure-and-integration-boundary`, matching the handoff doc's
requested working method (plan → structural cleanup → commit → M5 boundary →
commit → validate → show diff before push).

## Phase A — Structural cleanup

**Goal:** separate application/domain code, evaluation tooling, and tests
without changing runtime behavior.

Keep `job_dedup_rag/` as a flat top-level package (no `src/` migration — it's
already import-clean and a `src/` move is pure churn with no behavior benefit
here; the handoff doc explicitly leaves this to judgment).

Target layout:

```text
job_dedup_rag/            core domain package (unchanged import path)
  graph.py, nodes.py, models.py, state.py, vector_store.py,
  retry_policy.py, exceptions.py
  file_loader.py            keep load_jobs_from_manifest (used by main.py + evals)
  evaluation.py              keep as-is: EvaluationCase/Observation/Summary,
                              summarize_evaluations, run_evaluation_case,
                              skip_document_storage, validate_evaluation_namespace
                              — reusable, no I/O side effects, already importable

evals/                     NEW top-level package: command entry points only
  evaluation_runner.py       moved from job_dedup_rag/ (does print()-based reporting)
  seed_evaluation.py         moved from job_dedup_rag/
  synthetic_evaluation.py    moved from job_dedup_rag/
  private_evaluation.py      moved from job_dedup_rag/

tests/                     NEW pytest suite, replacing testmodules/
  unit/                      no network, no credentials, run by default
  live/                      @pytest.mark.live — real OpenAI/Pinecone calls, opt-in

main.py                    stays at repo root, unchanged (private ingestion entry)
evaluation_data/           unchanged
```

**`file_loader.py` split (production vs. evaluation responsibilities):**
`job_dedup_rag/file_loader.py` currently combines `load_jobs_from_manifest`
(production — used by `main.py`, returns `list[JobPosting]`, no dependency on
evaluation types) with `load_evaluation_cases` (evaluation-only — returns
`list[EvaluationCase]`, depends on `job_dedup_rag.evaluation.EvaluationCase`,
only ever called from evaluation entry points). These are split, not justified
as combined:
- `job_dedup_rag/file_loader.py` keeps only `load_jobs_from_manifest`.
- `evals/file_loader.py` gets `load_evaluation_cases`, imported by the moved
  `evaluation_runner.py`.

This mirrors the same production/evaluation boundary the rest of Phase A draws
elsewhere (`evaluation.py`'s reusable models/metrics stay in core; command-level
I/O and evaluation-specific loading move to `evals/`).

**File-by-file disposition of `testmodules/*.py` (25 files):**
Convert each to a `pytest` test function/module under `tests/unit/` or
`tests/live/`. A first pass (grepping for `ChatOpenAI`, `build_vector_store`,
`job_id_exists`, `OpenAIEmbeddings`, or direct node calls that hit those) shows
roughly half touch real external clients and half are pure (state shape, retry
classification, search-text formatting, evaluation-summary math, file loading,
graph routing via injected fakes). The grep is a starting signal, not the final
call — read each file during the move and place it by what it *actually*
exercises, not just what it imports. Where a test currently makes a real
OpenAI/Pinecone call but could instead use a fake/monkeypatched client to test
the same logic (e.g. node wiring, error translation), prefer converting it to a
unit test with a fake rather than leaving it live — reserve `live/` for tests
that need genuine model/vector-store behavior (extraction quality, real
similarity search, the full graph run, the synthetic evaluation regression
check).

**Command changes to document in README:**
- `uv run python -m testmodules.X` commands → `uv run pytest tests/unit` (default)
  and `uv run pytest tests/live -m live` (opt-in, costs money, needs `.env`).
- `uv run python -m job_dedup_rag.seed_evaluation` → `uv run python -m evals.seed_evaluation`
- `uv run python -m job_dedup_rag.synthetic_evaluation` → `uv run python -m evals.synthetic_evaluation`
- `uv run python -m job_dedup_rag.private_evaluation` → `uv run python -m evals.private_evaluation`
- Validation commands updated: `uv run python -m compileall job_dedup_rag evals tests main.py`

**pyproject.toml changes:**
- Add `pytest` to `[dependency-groups] dev`.
- Add `[tool.pytest.ini_options]` with `markers = ["live: requires OPENAI/PINECONE credentials and may incur real API cost"]` and `addopts = "-m 'not live'"` so a bare `uv run pytest` is always safe/free by default.
- No packaging-exclusion changes needed yet — this project isn't currently built/shipped as a distributable artifact (no wheel build, no Lambda packaging step exists here), so "ensure packaging can exclude tests/evals" is satisfied simply by them living outside `job_dedup_rag/` already. Revisit if/when this becomes a real deployable dependency for jobtracker.

**Validation for Phase A:** `uv run ruff check .`, `uv run ruff format --check .`,
`uv run python -m compileall job_dedup_rag evals tests main.py`, `git diff --check`,
`uv run pytest tests/unit` (must pass with zero credentials/network). Commit
separately once green.

## Phase B — Milestone 5 integration boundary

**Goal:** one stable, reviewable seam that a future JobTracker integration can
call, without leaking LangGraph/Pinecone internals and without this package ever
importing anything from the `jobtracker` repo.

New module: `job_dedup_rag/boundary.py`

```python
class DeduplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    company_name: str
    role_title: str
    found_by: str
    job_description: str
    # validators: reject empty/whitespace-only required fields


class DeduplicationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["already_exists", "possible_duplicate", "stored"]
    incoming_job_id: str
    matched_job_id: str | None = None
    stored_ids: list[str] | None = None
    # model_validator(mode="after"): enforce status-specific shape —
    #   already_exists / possible_duplicate require matched_job_id set and
    #   stored_ids None; stored requires stored_ids set (non-empty) and
    #   matched_job_id None. Raises on any other combination.


def deduplicate_job(request: DeduplicationRequest) -> DeduplicationResponse:
    return _run_graph(request, _PRODUCTION_GRAPH)


_PRODUCTION_GRAPH = build_ingestion_graph()  # compiled once at import time, reused


def _run_graph(
    request: DeduplicationRequest, graph: IngestionGraph
) -> (
    DeduplicationResponse
): ...  # module-private seam: tests inject a fake/no-write graph here directly
```

- `deduplicate_job(request)` is the single public entry point (matches M5
  requirement #3: "one public service/function entry point") and takes no
  `storage_node` or graph parameter — that knob is not part of the stable
  contract. It always runs against `_PRODUCTION_GRAPH`, a module-level graph
  compiled once via `build_ingestion_graph()` (default production topology,
  real `store_document`) and reused across calls rather than rebuilt per call.
- The graph-to-response mapping is factored into a module-private `_run_graph`
  that takes the compiled graph/invoker as a parameter. Unit tests call
  `_run_graph` directly with a graph built via
  `build_ingestion_graph(storage_node=skip_document_storage)` (the same
  no-write pattern `evaluation.py` already uses) or a hand-built fake — so
  tests never hit real Pinecone/OpenAI and never need to monkeypatch the
  production singleton.
- `_run_graph` maps the resulting `IngestionState` onto `DeduplicationResponse`
  and does not expose `Document`, candidate lists, comparison objects, or
  graph/node names. Malformed/inconsistent graph results (e.g. a `stored`
  status with no `stored_ids`, or a duplicate status with no `matched_job_id`)
  raise via the response model's validator rather than silently producing an
  inconsistent `DeduplicationResponse` — covered by dedicated tests that feed
  `_run_graph` a fake graph returning deliberately malformed `IngestionState`
  results.
- **Confidence/explanation:** deliberately excluded from `DeduplicationResponse`
  v1. `DuplicateComparison.confidence`/`.explanation` only exist on the
  `possible_duplicate` path (a real `DuplicateComparison`) — `already_exists`
  and `stored` never produce one. The handoff doc says only expose these "if the
  graph can provide them consistently across all result paths" — it can't, so
  leaving them out avoids a leaky, inconsistently-populated contract. Revisit
  later if a caller actually needs it.

**JobTracker adapter — stays inside job-dedup-rag, structurally decoupled:**
Add `job_dedup_rag/integrations/jobtracker.py` defining:
- A `JobTrackerLikeItem` `Protocol` limited to exactly the five fields the RAG
  request actually needs — `job_id`, `company_name`, `role_title`, `found_by`,
  `job_description` — not the full `JobItem` shape (which also has `status`,
  `job_url`, `date_found`, `date_posted`, `notes`). Narrower interface, and it
  won't need to change if `JobItem` grows fields this boundary doesn't care
  about. Structurally compatible with the real `jobtracker.models.JobItem`
  (matches those five field names/types), but this file never imports it.
- `map_job_item_to_request(item: JobTrackerLikeItem) -> DeduplicationRequest`,
  raising a clear `ValueError` for missing/blank required fields.

This satisfies the handoff's hard constraint ("do not make the RAG repository
import `jobtracker.models.JobItem`") while still shipping a concrete, tested
mapping function. When jobtracker is later wired up for real, its own repo adds
a thin call site that imports the real `JobItem`, converts to this protocol
shape (trivially — the fields already match name-for-name against the real
dataclass pulled from `Barfolemu/jobtracker`), and calls `deduplicate_job`. That
wiring, plus how jobtracker actually depends on this package at runtime
(vendored, git dependency, Lambda layer — jobtracker already has a working
precedent for isolating heavy deps into a separate Lambda layer for its Gmail
agent, which is the natural pattern to reuse later), is out of scope now per
Ashley's decision.

**Tests:** `tests/unit/test_boundary.py` covering:
- Valid request → each of the three result statuses (using fake/injected
  storage + a fake graph or monkeypatched nodes, no real API calls) — reuse the
  `skip_document_storage` / no-write pattern already established in
  `job_dedup_rag/evaluation.py`.
- Invalid/missing required fields on `DeduplicationRequest` (Pydantic
  validation errors).
- `map_job_item_to_request` valid mapping and rejection of missing/blank
  required fields.

**Documentation:** add a "JobTracker integration boundary" section to README
covering the request/response contract, `deduplicate_job`, the adapter's
protocol-based decoupling rationale, and explicitly noting no transport exists
yet.

**Validation for Phase B:** same ruff/compileall/pytest-unit checks as Phase A,
plus this new test file. Commit separately.

## Final validation before declaring done

1. `uv run ruff check .`, `uv run ruff format --check .`,
   `uv run python -m compileall job_dedup_rag evals tests main.py`, `git diff --check`.
2. `uv run pytest tests/unit` — must pass with no credentials.
3. Rerun the guarded synthetic evaluation and confirm no regression from the
   Milestone 4 baseline (`status_accuracy=1.0`, `duplicate_precision=1.0`,
   `duplicate_recall=1.0`, no false positives/negatives):
   `PINECONE_NAMESPACE=evaluation-m4-20260816 uv run python -m evals.synthetic_evaluation`
   — **this makes real, paid OpenAI/Pinecone calls; get explicit go-ahead
   immediately before running it**, same as the handoff doc's own guidance not
   to run paid checks casually.
4. Show Ashley the full diff, the two commits, and validation output before
   pushing or opening a PR — nothing gets pushed without explicit go-ahead.

## Open items flagged, not pre-decided

- The handoff doc's evaluation baseline (36.56s latency, 5/5 cases) is the
  target to not regress — this plan doesn't touch duplicate-detection
  thresholds, prompts, or the comparison model's behavior anywhere.
- `evals/` vs keeping evaluation commands nested under `job_dedup_rag/` is a
  judgment call made here per the handoff doc's own "use judgment" language —
  flagging it in case Ashley would rather keep the eval CLI entry points nested
  under `job_dedup_rag.evals` instead of a top-level `evals/` package; either is
  a small rename if preferred.
