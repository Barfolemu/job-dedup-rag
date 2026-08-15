import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


def build_vector_store() -> PineconeVectorStore:
    load_dotenv(override=True)

    index_name = os.environ["PINECONE_INDEX_NAME"]
    model_name = os.environ["OPENAI_EMBEDDING_MODEL"]
    api_key = os.environ["PINECONE_API_KEY"]
    namespace = os.environ["PINECONE_NAMESPACE"]

    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index(index_name)
    embeddings = OpenAIEmbeddings(model=model_name)

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
    )