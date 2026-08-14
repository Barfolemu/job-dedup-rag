# Job Dedup RAG

A small Python project for loading job postings, embedding their descriptions,
and storing them in Pinecone so semantically similar postings can be retrieved.
The ingestion workflow is orchestrated with LangGraph.

The project currently provides the ingestion and similarity-search foundation
for job deduplication. It does **not** yet classify results as duplicates or
apply a duplicate-score threshold automatically.

## How it works

Each entry in `data/jobs/manifest.json` points to a text file containing a job
description. The application:

1. Loads the manifest and corresponding description files.
2. Converts each posting into a LangChain `Document`.
3. Adds job metadata (`job_id`, company, role, and source) to the document.
4. Creates an embedding with an OpenAI embedding model.
5. Stores the vector and metadata in a Pinecone index using the job ID as the
   vector ID.

Using a stable job ID means ingesting the same source record again updates that
ID rather than intentionally creating a second ID.

```text
manifest.json + job text files
              |
              v
       load job postings
              |
              v
   create LangChain Document
              |
              v
 OpenAI embedding -> Pinecone index
              |
              v
      similarity retrieval
```

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended for dependency management)
- An OpenAI API key
- A Pinecone API key and an existing Pinecone index

The Pinecone index dimension must match the output dimension of the configured
OpenAI embedding model. Use the same embedding model for both ingestion and
retrieval.

## Setup

Install the locked dependencies:

```bash
uv sync
```

Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=your-embedding-model
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-index-name
```

The `.env` file is ignored by Git. Do not commit API keys.

## Input data

Place job-description text files in `data/jobs/` and describe them in
`data/jobs/manifest.json`:

```json
[
  {
    "file": "123456.txt",
    "job_id": "linkedin:123456",
    "company_name": "Example Company",
    "role_title": "Engineering Manager",
    "found_by": "linkedin"
  }
]
```

The `file` value is resolved relative to the manifest. Each referenced file
should contain the complete plain-text job description. The `data/jobs/`
directory is ignored by Git because it is local test/input data.

## Run ingestion

Run commands from the repository root:

```bash
uv run python main.py
```

The program ingests every manifest entry and prints the Pinecone IDs returned
for each stored posting. It does not delete those records afterward.

## Retrieve similar postings

After ingestion, run the retrieval smoke test:

```bash
uv run python -m testmodules.retrievaltest
```

This uses the first manifest posting as a query and prints the five closest
records returned by Pinecone. The meaning of the returned score depends on the
metric configured for the Pinecone index; establish a duplicate threshold from
representative labeled examples rather than assuming a universal cutoff.

## Development test scripts

The files under `testmodules/` are informal Python scripts for testing pieces of
the project as it is developed. They are not a pytest suite and are not intended
as a production test harness. Run them in module form from the repository root
so imports resolve correctly:

```bash
# Tests that do not call external APIs
uv run python -m testmodules.fileloadertest
uv run python -m testmodules.documentnodetest

# Tests that use the configured OpenAI or Pinecone services
uv run python -m testmodules.pineconetest
uv run python -m testmodules.embeddingtest
uv run python -m testmodules.vectorstoretest
uv run python -m testmodules.graphtest
uv run python -m testmodules.retrievaltest
```

`vectorstoretest` and `graphtest` create temporary test vectors and delete them
at the end of a successful run. If either script exits before cleanup, remove
the test vector manually before rerunning if needed.

Avoid invoking package-dependent scripts as paths, for example
`python testmodules/fileloadertest.py`; that can produce
`ModuleNotFoundError: No module named 'job_dedup_rag'`. Use the `python -m ...`
commands above instead.

## Project layout

```text
.
|-- main.py                       # Loads and ingests all manifest jobs
|-- job_dedup_rag/
|   |-- file_loader.py            # Reads manifest entries and text files
|   |-- state.py                  # Typed job and graph state definitions
|   |-- nodes.py                  # Document creation and storage nodes
|   |-- graph.py                  # LangGraph ingestion workflow
|   `-- vector_store.py           # OpenAI/Pinecone client construction
|-- testmodules/                  # Informal scripts used during development
|-- data/jobs/                    # Local manifest and descriptions
|-- pyproject.toml                # Project metadata and dependencies
`-- uv.lock                       # Locked dependency versions
```

## Troubleshooting

### `python: command not found` or missing packages

Use `uv run python ...` so the command runs with the project environment. If
using the virtual environment directly, run `.venv/bin/python` on Linux/macOS.

### Missing environment variable

An error such as `KeyError: 'PINECONE_INDEX_NAME'` means the corresponding
value is absent from `.env` or the process environment. Confirm all four setup
variables are present.

### Cannot connect to Pinecone or OpenAI

DNS errors, connection timeouts, and `MaxRetryError` indicate that the process
cannot reach the external API. Check network access, proxy/firewall settings,
and service availability.

### Pinecone dimension error

The selected embedding model and Pinecone index must use the same vector
dimension. Create a compatible index or change `OPENAI_EMBEDDING_MODEL` to the
model used when the index was created.

### Authentication or index-not-found error

Verify the API key and confirm `PINECONE_INDEX_NAME` names an existing index in
the project associated with that key.
