from langchain_openai import ChatOpenAI
from memory.base import BaseMemory
from client.rag_client import rag_service
from rag.rag_service import Chunk


class SemanticMemory(BaseMemory):
    """
    语义记忆，用于存储和检索语义信息。
    """
    def __init__(self):
        self.rag_service = rag_service

    def add(self, user_id: str, file_path: str, password: str = None):
        """
        添加语义记忆。
        :param user_id: 用户id
        :param file_path: 文件路径
        :param password: 密码
        """
        rag_service.add_to_neo4j_qdrant(user_id, file_path, password)


    def get(self, user_id: str, query: str) -> list[Chunk]:
        """
        从语义记忆中检索信息。
        :param user_id: 用户id
        :param query: 查询文本
        :return: 包含查询结果的列表
        """
        return rag_service.retriever_chunks(user_id, query)

