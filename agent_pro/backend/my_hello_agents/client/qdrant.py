import os
from qdrant_client import QdrantClient


def get_qdrant_client() -> QdrantClient:
    """
    获取 Qdrant 客户端
    :return: QdrantClient
    """
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

qdrant_client = get_qdrant_client()