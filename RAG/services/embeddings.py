import os

from langchain_huggingface import HuggingFaceEmbeddings


# all-MiniLM-L6-v2 is a small (80MB, 384-dim) bi-encoder — much faster to load
# and embed than bge-m3, with far lower memory. Override via EMBEDDING_MODEL_NAME.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def create_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
