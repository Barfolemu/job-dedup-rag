# Job Dedup RAG — Claude Handoff for Structural Cleanup and Milestone 5

## Purpose

The primary learning goal of this project was to understand LangGraph and RAG by building the workflow incrementally. That goal has been achieved. The next phase can be implementation-focused: clean up the Python project structure and complete the defined Milestone 5 integration boundary.

Do not redesign the deduplication behavior or expand this into full production integration without Ashley's explicit approval.

## Repository state

- Repository: `Barfolemu/job-dedup-rag`
- Current branch: `main`
- Current `main` commit: `1f94df5`
- Merged pull request: `https://github.com/Barfolemu/job-dedup-rag/pull/3`
- PR title: `Milestone 4: reliability and evaluation`
- Working tree was clean after pulling the merge.
- Milestones 1–4 are complete.

Before changing anything:

```bash
git switch main
git pull --ff-only
git status
git log -5 --oneline
```

Create a new branch for this work. Suggested name:

```bash
git switch -c feature/structure-and-integration-boundary
```

## Current system behavior

The LangGraph workflow:

1. Checks Pinecone for the incoming exact job ID.
2. Returns `already_exists` immediately when that ID exists.
3. Extracts typed job features for new IDs.
4. Builds normalized searchable text.
5. Retrieves the five closest semantic candidates.
6. Treats retrieval of the incoming exact ID as an invariant failure.
7. Compares candidates serially with the comparison model.
8. Returns `possible_duplicate` when a different ID represents the same opening.
9. Otherwise stores the job and returns `stored`.

The exact-ID path bypasses extraction, embeddings, retrieval, and comparison.

Important result semantics:

- `already_exists`: the same source/job ID is already stored.
- `possible_duplicate`: a different ID appears to represent the same opening.
- `stored`: no duplicate was confirmed, so the incoming job was stored.

## Milestone 4 capabilities that must be preserved

- Exact-ID lookup through Pinecone `fetch`.
- Explicit same-ID semantic-retrieval invariant.
- Three total attempts for transient external failures with exponential backoff and jitter.
- No retries for internal programming, validation, or invariant failures.
- OpenAI client retries disabled to avoid stacked retry policies.
- `ExternalServiceOperationError` includes operation and job-ID context.
- Separate extraction and comparison model settings.
- Optional LangSmith tracing with safe run metadata.
- 90-day `indexed_at_epoch` and `expires_at_epoch` metadata on new vectors.
- Storage-node dependency injection so evaluations use the production graph topology without writing query cases.
- Evaluation namespaces must start with `evaluation-` and contain a suffix.
- Seed evaluation intentionally writes fixed-ID records; evaluation runners do not write query records.
- Cross-company policy: different company names are not duplicates without explicit evidence that the employers are equivalent.
- Uncollected token and cost measurements serialize as `null`, not zero.
- Aggregate matched-job-ID accuracy is reported separately from status accuracy.

## Final Milestone 4 evaluation baseline

Synthetic namespace: `evaluation-m4-20260816`

- Total cases: 5
- Correct statuses: 5
- Status accuracy: `1.0`
- Duplicate precision: `1.0`
- Duplicate recall: `1.0`
- Retrieval recall at five: `1.0`
- Mean matching candidate rank: `1.0`
- Matched-job-ID eligible cases: 2
- Correct matched-job-ID cases: 2
- Matched-job-ID accuracy: `1.0`
- Mean comparison count: `2.0`
- False positives: none
- False negatives: none
- Final measured latency: approximately `36.56` seconds
- Token and cost fields: `null` because usage is not collected

The private evaluation also previously passed 2/2 cases: exact-ID Affirm and a cross-source Indeed/LinkedIn version of the same Affirm opening.

Do not relabel evaluation cases to make results pass.

## Privacy and external-service constraints

- `.env`, `.venv`, and `data/jobs/` are private and ignored.
- Never commit real job descriptions or credentials.
- `main.py` processes the private manifest, can make paid calls, and can write to the production namespace. Do not run it as a routine test.
- Production/private Pinecone namespace: `structured-v1`.
- Public synthetic evaluation data lives under `evaluation_data/`.
- Synthetic evaluation queries are no-write, but `seed_evaluation` intentionally writes.
- LangSmith tracing may transmit complete job descriptions, prompts, candidates, model outputs, and graph state. Keep it disabled unless explicitly approved.
- Pinecone does not enforce `expires_at_epoch`; cleanup remains unimplemented.

## Structural cleanup objective

Ashley correctly noticed that evaluation commands and development tests are mixed too closely with application code. Reorganize the repository into a conventional, production-readable Python structure while preserving behavior.

Target separation should be conceptually similar to:

```text
src/job_dedup_rag/     application and reusable domain code
tests/                 automated unit/integration tests
evals/                 evaluation and seeding command entry points
evaluation_data/       public synthetic fixtures
```

Use judgment about whether a full `src/` migration is worth the churn. The required outcome is a clear boundary, not a particular directory name.

Structural cleanup requirements:

1. Keep reusable evaluation models and metric calculation as importable code.
2. Move private, synthetic, and seed command entry points out of the core runtime surface or organize them in a clearly named evaluation package.
3. Replace the ad hoc executable `testmodules/` organization with conventional automated tests, preferably `pytest`.
4. Separate tests that are completely local from tests requiring OpenAI, Pinecone, private data, or paid calls.
5. Add markers, names, or documented commands that make live tests opt-in.
6. Preserve the no-write evaluation guarantee.
7. Preserve existing module or CLI behavior where practical; document intentional command changes.
8. Ensure packaging/deployment can exclude tests, private data, and evaluation-only commands where appropriate.
9. Keep Ruff and compilation checks clean.
10. Update the README to match the final organization.

Perform the structural cleanup as a distinct, reviewable phase before implementing the Milestone 5 interface.

## Milestone 5 scope

Milestone 5 is an integration boundary, not the full JobTracker deployment integration.

### Required deliverables

1. Define a stable Pydantic request model around the deduplication workflow.
2. Define a stable Pydantic response model that exposes only integration-relevant results, not the internal LangGraph state.
3. Provide one public service/function entry point that accepts the request model, invokes the graph, and returns the response model.
4. Define a JobTracker-to-RAG mapping without importing JobTracker internals into the RAG domain package.
5. Add boundary tests for valid mappings, invalid/missing fields, and every result status.
6. Document how JobTracker would call this boundary.
7. Keep transport concerns separate. Do not add HTTP, Lambda, queues, or cross-repository deployment unless Ashley explicitly approves a transport design.

### JobTracker record shape

The supplied JobTracker snapshot defines this record:

```python
@dataclass
class JobItem:
    job_id: str
    company_name: str
    role_title: str
    status: Status = Status.NEW
    job_url: str = ""
    found_by: str = "manual"
    date_found: str = ""
    date_posted: str = ""
    job_description: str = ""
    notes: str = ""
```

The current RAG graph requires:

- `job_id`
- `company_name`
- `role_title`
- `found_by`
- `job_description`

The adapter should explicitly map those fields and reject invalid required values. Do not make the RAG repository import `jobtracker.models.JobItem`. Prefer a primitive mapping, protocol, or adapter owned at the integration edge.

### Suggested request semantics

The exact names are open to review, but the request should represent one incoming job posting with typed required fields. Avoid exposing Pinecone documents, graph node names, candidate indexes, or other orchestration details.

### Suggested response semantics

At minimum, expose:

- result status: `already_exists`, `possible_duplicate`, or `stored`
- incoming job ID
- matched job ID when applicable
- stored IDs when applicable

Only expose confidence or explanation if the graph can provide them consistently across all result paths and the API contract defines when they are absent.

## Non-goals unless Ashley expands scope

- Do not deploy the RAG project.
- Do not connect the live JobTracker Lambda to it.
- Do not modify JobTracker AWS infrastructure.
- Do not add a UI.
- Do not implement retention cleanup.
- Do not backfill old Pinecone vectors.
- Do not change embedding models, Pinecone indexes, namespaces, or duplicate thresholds without evaluation evidence and approval.
- Do not broaden the synthetic corpus simply to make structural work look larger.

## Expected working method

This phase is implementation-focused rather than a line-by-line tutorial, but preserve reviewability:

1. Inspect both repositories and propose the concrete target structure and interface before editing.
2. Identify compatibility risks, especially import paths and evaluation commands.
3. Implement structural cleanup first and run local tests.
4. Commit structural cleanup separately.
5. Implement the Milestone 5 request/response boundary and adapter tests.
6. Commit Milestone 5 separately.
7. Run the full local suite and the minimum necessary opt-in live evaluation.
8. Update documentation.
9. Show Ashley the final diff, commits, and validation results before pushing or opening a PR.

Do not silently commit, push, merge, deploy, delete Pinecone records, or change AWS resources without Ashley's explicit instruction.

## Validation expectations

At minimum, retain equivalents of:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall src tests
git diff --check
```

Adjust paths if the final structure does not use `src/`. Local automated tests must pass without credentials or network access. Live tests must be explicitly separated and documented.

Before declaring completion, rerun the five-case synthetic evaluation in the guarded namespace and confirm no regression from the Milestone 4 baseline:

```bash
PINECONE_NAMESPACE=evaluation-m4-20260816 \
uv run python -m job_dedup_rag.synthetic_evaluation
```

If structural cleanup changes that command, document and use its supported replacement.

## First response requested from Claude

Before writing code, provide:

1. The proposed final directory structure.
2. Which current files move, split, or remain.
3. The proposed Pydantic request and response contracts.
4. Where the JobTracker adapter belongs and why.
5. The test split between local and live tests.
6. Any questions or scope conflicts requiring Ashley's decision.

After Ashley approves that plan, implement the work on the new branch.
