# Job Dedup RAG

An early-stage duplicate-detection pipeline for job postings collected from
different sources. It uses LangGraph to orchestrate structured LLM extraction
and retrieval-augmented generation (RAG), with the goal of recognizing the same
role even when aggregators reformat or lightly rewrite the original posting.

## Milestone 3: serial duplicate comparison

Milestone 3 adds LLM duplicate comparison to structured retrieval:

```text
JobPosting
    -> structured LLM extraction (gpt-5.4-mini)
    -> normalized search text
    -> OpenAI embedding (text-embedding-3-small, 1536 dimensions)
    -> Pinecone top-five retrieval (cosine similarity, structured-v1 namespace)
    -> serial LLM comparison in similarity order
         |-> duplicate found: possible_duplicate
         `-> all candidates rejected: store job -> stored
```

The extraction step converts each job description into typed identifying
features: company, role, requisition ID, location, workplace and employment
types, seniority, team/domain, responsibilities, qualifications, and
technologies. Those fields become consistent search text before embedding,
reducing noise caused by source-specific formatting.

The searchable Pinecone document contains the normalized text. Its metadata
retains the original job description and source fields so the comparison stage
can evaluate candidates against the complete JD rather than the compact
retrieval representation.

Candidates are compared serially in descending similarity order. Comparison
stops immediately when the LLM identifies a duplicate. The graph returns
`possible_duplicate` and does not add that job as a new Pinecone record. If all
candidates are rejected—or retrieval returns none—the unique job is stored and
the graph returns `stored`. Retrieval itself still creates a transient query
embedding; “not stored” means a possible duplicate is not embedded and written
as a new vector record.

The current LangGraph workflow is:

```text
extract_job_features -> create_document -> retrieve_candidates
    -> compare candidates serially
         |-> mark_possible_duplicate -> END
         `-> store_document -> END
```

## Technology choices

- Python 3.12+
- LangGraph and LangChain
- `gpt-5.4-mini` for schema-constrained feature extraction
- `text-embedding-3-small` with 1536-dimensional vectors
- Pinecone cosine-similarity index
- Pinecone namespace `structured-v1`
- Pydantic for the extracted feature schema

The Pinecone index must already exist with 1536 dimensions and cosine as its
distance metric.

## Setup

Install the locked dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Copy the supplied environment template and add your credentials:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-5.4-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

PINECONE_API_KEY=
PINECONE_INDEX_NAME=job-dedup-rag
PINECONE_NAMESPACE=structured-v1

# Optional LangSmith tracing (set to false to disable)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=job-dedup-rag-development
```

Both `.env` and `data/jobs/` are intentionally ignored by Git. The job-data
directory contains real job descriptions and must remain local.

LangSmith tracing is optional. When enabled, prompts, job descriptions, and
model outputs are sent to the configured LangSmith workspace. Leave tracing
disabled if that workspace is not approved to receive the job-posting data.

## Run the workflow

Place a local `manifest.json` and its referenced JD text files under
`data/jobs/`, then populate the retrieval index:

```bash
uv run python main.py
```

Each manifest entry supplies `job_id`, `company_name`, `role_title`, `found_by`,
and the filename containing the original description.

## Development checks

`testmodules/` contains focused Python scripts used during development, not a
formal pytest suite. Run them from the repository root in module form:

```bash
# Local transformation checks
uv run python -m testmodules.fileloadertest
uv run python -m testmodules.searchtexttest
uv run python -m testmodules.documentnodetest

# OpenAI and Pinecone checks
uv run python -m testmodules.extractionnodetest
uv run python -m testmodules.embeddingtest
uv run python -m testmodules.retrievaltest
uv run python -m testmodules.retrievalnodetest
uv run python -m testmodules.comparisonnodetest
uv run python -m testmodules.graphtest
uv run python -m testmodules.duplicategraphtest
```

The milestone acceptance check uses an Indeed-style reformatted copy of an
existing LinkedIn JD. It verifies that structured retrieval ranks the original
LinkedIn posting first across sources and that its original JD is available in
metadata:

```bash
uv run python -m testmodules.crosssourceduplicatetest
```

Run `uv run python main.py` first so the `structured-v1` namespace contains the
source postings.

## Result statuses

- `possible_duplicate`: comparison found a matching candidate; the new job is
  not stored.
- `stored`: every retrieved candidate was rejected, or none were returned; the
  new job is stored after comparison completes.

## Current limitations

- An exact-ID precheck is not implemented.
- Candidate retrieval currently requests six results and experimentally filters
  out a result with the same job ID before returning up to five candidates.
  This same-ID filtering must be replaced with a proper invariant after the
  exact-ID precheck is added; it is not production-ready deduplication logic.
- JobTracker integration is not implemented.
