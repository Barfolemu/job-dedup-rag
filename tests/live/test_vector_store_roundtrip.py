import os

import pytest
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from .conftest import require_mutating_namespace, unique_test_job_id

pytestmark = pytest.mark.mutating


def test_vector_store_write_search_and_delete_round_trip() -> None:
    """Writes and deletes one throwaway record in a guarded evaluation-* namespace."""
    namespace = require_mutating_namespace()

    index_name = os.environ["PINECONE_INDEX_NAME"]
    model_name = os.environ["OPENAI_EMBEDDING_MODEL"]

    pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pinecone.Index(index_name)
    embeddings = OpenAIEmbeddings(model=model_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
    )

    job_id = unique_test_job_id("test:engineering-manager")

    document = Document(
        page_content=(
            "Lead a software engineering team building cloud-native applications "
            "on AWS. Manage engineers, improve delivery practices, and guide the "
            "architecture of distributed systems."
        ),
        metadata={
            "job_id": job_id,
            "company_name": "Example Company",
            "role_title": "Software Engineering Manager",
            "source": "test",
        },
    )

    try:
        stored_ids = vector_store.add_documents(documents=[document], ids=[job_id])
        assert stored_ids == [job_id]

        results = vector_store.similarity_search_with_score(
            "Engineering leader managing teams and AWS cloud applications",
            k=1,
        )

        assert results
        assert results[0][0].metadata["job_id"] == job_id
    finally:
        # Deleting a non-existent ID is a no-op, so this is safe even if
        # add_documents() never succeeded or an assertion failed first.
        vector_store.delete(ids=[job_id])
