import os

from dotenv import load_dotenv
from pinecone import Pinecone


def test_configured_index_exists_and_is_read_only_described() -> None:
    load_dotenv()

    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ["PINECONE_INDEX_NAME"]

    pinecone = Pinecone(api_key=api_key)
    index_description = pinecone.describe_index(index_name)

    assert index_description.name == index_name
    assert index_description.dimension == 1536
    assert index_description.metric == "cosine"
