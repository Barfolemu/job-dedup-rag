import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

index_name = os.environ["PINECONE_INDEX_NAME"]
model_name = os.environ["OPENAI_EMBEDDING_MODEL"]

pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pinecone.Index(index_name)

embeddings = OpenAIEmbeddings(model=model_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

job_id = "test:engineering-manager"

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

stored_ids = vector_store.add_documents(
    documents=[document],
    ids=[job_id],
)

print(f"Stored IDs: {stored_ids}")

results = vector_store.similarity_search_with_score(
    "Engineering leader managing teams and AWS cloud applications",
    k=1,
)

for matching_document, score in results:
    print(f"Score: {score}")
    print(f"Metadata: {matching_document.metadata}")
    print(f"Content: {matching_document.page_content}")

vector_store.delete(ids=[job_id])
print(f"Deleted test record: {job_id}")
