import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.environ["PINECONE_API_KEY"]
index_name = os.environ["PINECONE_INDEX_NAME"]

pinecone = Pinecone(api_key=api_key)
index_description = pinecone.describe_index(index_name)

print(index_description)
