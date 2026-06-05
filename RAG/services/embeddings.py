from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


def create_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
