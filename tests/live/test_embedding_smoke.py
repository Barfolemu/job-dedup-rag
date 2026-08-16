import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


def test_embedding_model_returns_expected_dimensions() -> None:
    load_dotenv()

    model_name = os.environ["OPENAI_EMBEDDING_MODEL"]
    embeddings = OpenAIEmbeddings(model=model_name)

    vector = embeddings.embed_query(
        "Senior Software Engineering Manager leading cloud platform teams"
    )

    assert len(vector) == 1536
