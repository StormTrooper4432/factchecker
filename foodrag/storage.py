import os

import chromadb

from . import config
from .embeddings import FoodEmbeddingFunction


def get_client():
    os.makedirs(config.CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_PATH)


def get_collection(client):
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=FoodEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def recreate_collection(client):
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    return get_collection(client)
