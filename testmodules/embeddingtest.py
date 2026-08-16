import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

model_name = os.environ["OPENAI_EMBEDDING_MODEL"]

embeddings = OpenAIEmbeddings(model=model_name)

vector = embeddings.embed_query(
    "Senior Software Engineering Manager leading cloud platform teams"
)

print(f"Model: {model_name}")
print(f"Dimensions: {len(vector)}")
print(f"First five values: {vector[:5]}")
