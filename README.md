# Job Dedup RAG

An early-stage duplicate-detection pipeline for job postings collected from
different sources. It uses LangGraph to orchestrate structured LLM extraction
and retrieval-augmented generation (RAG), with the goal of recognizing the same
role even when aggregators reformat or lightly rewrite the original posting.

## Milestone 2: structured retrieval

Milestone 2 implements candidate retrieval:

```text
JobPosting
    -> structured LLM extraction (gpt-5.4-mini)
    -> normalized search text
    -> OpenAI embedding (text-embedding-3-small, 1536 dimensions)
    -> Pinecone top-five retrieval (cosine similarity, structured-v1 namespace)
```

The extraction step converts each job description into typed identifying
features: company, role, requisition ID, location, workplace and employment
types, seniority, team/domain, responsibilities, qualifications, and
technologies. Those fields become consistent search text before embedding,
reducing noise caused by source-specific formatting.

The searchable Pinecone document contains the normalized text. Its metadata
retains the original job description and source fields so a later comparison
stage can evaluate candidates against the complete JD rather than the compact
retrieval representation.

The current LangGraph workflow is:

```text
extract_job_features -> create_document -> retrieve_candidates -> store_document
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
```

Both `.env` and `data/jobs/` are intentionally ignored by Git. The job-data
directory contains real job descriptions and must remain local.

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
uv run python -m testmodules.graphtest
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

## Current limitations

- There is no final LLM duplicate comparison or duplicate classification yet.
- The graph still stores every ingested job after candidate retrieval.
- An exact-ID precheck is not implemented.
- Candidate retrieval currently requests six results and experimentally filters
  out a result with the same job ID before returning up to five candidates.
  This same-ID filtering must be replaced with a proper invariant after the
  exact-ID precheck is added; it is not production-ready deduplication logic.
- JobTracker integration is not implemented.

## Next milestone

Add serial LLM comparison of the query job against the top five retrieved
candidates. The comparison will inspect each candidate's original JD and exit
early as soon as a duplicate is confirmed, avoiding unnecessary model calls.
