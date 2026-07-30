import os
from typing import List, Any

import redis
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from qdrant_client.http.models import ScoredPoint

from client.model import model
from memory.preference import PreferenceMemory
from memory.semantic import SemanticMemory
from my_hello_agents.memory.episodic import EpisodicMemory
from my_hello_agents.memory.history import HistoryMemory
from my_hello_agents.memory.work import WorkMemory
from rag.rag_service import Chunk


class MemoryManager:
    """
    内存管理器，用于存储和检索信息。
    """
    def __init__(self):
        self.model = model
        self.redis_client = redis.Redis(host='192.168.150.101', port=6379, db=0, password='123321')
        # 工作记忆
        self.workMemory = WorkMemory()
        # 历史记忆
        self.historyMemory = HistoryMemory(redis_client=self.redis_client)

        # 情景记忆
        self.episodicMemory = EpisodicMemory(summary_model=self.model)
        # 语义记忆
        self.semanticMemory = SemanticMemory()
        # 偏好记忆
        self.preferenceMemory = PreferenceMemory(redis_client=self.redis_client, model=self.model)

    def add_work_memory(self, key: str, value: List[BaseMessage]):
        """
        将信息添加到工作记忆中。
        :param key: 记忆键 user_id + session_id
        :param value: 记忆值，包含消息的列表
        """
        self.workMemory.add(key, value)

    def get_work_memory(self, key: str) -> List[BaseMessage]:
        """
        从工作记忆中检索信息。
        :param key: 记忆键 user_id + session_id
        :return: 记忆值，包含消息的列表
        """
        return self.workMemory.get(key)

    def add_history_memory(self, key: str, message: BaseMessage):
        """
        添加历史记忆
        """
        self.historyMemory.add(key, message)

    def get_history_memory(self, key: str) -> List[BaseMessage]:
        """
        从历史记忆中检索信息。
        :param key: 记忆键 user_id + session_id
        :return: 记忆值，包含消息的列表
        """
        return self.historyMemory.get(key)

    def add_episodic_memory(self, user_id: str, messages: List[BaseMessage]):
        """
        添加情景记忆
        """
        self.episodicMemory.add(user_id, messages)

    def get_episodic_memory(self, user_id: str, query: str) -> List[ScoredPoint]:
        """
        从情景记忆中检索信息。
        :param user_id
        :param query: 检索内容
        :return: 检索结果
        """
        return self.episodicMemory.get(user_id, query)

    def add_semantic_memory(self, user_id: str, query: str, response: str):
        """
        添加语义记忆
        """
        self.semanticMemory.add(user_id, query, response)

    def get_semantic_memory(self, user_id: str, query: str) -> List[Chunk]:
        """
        从语义记忆中检索信息。
        :param user_id
        :param query: 检索内容
        :return: 检索结果
        """
        return self.semanticMemory.get(user_id, query)

    def add_preference_memory(self, user_id: str, value: str):
        """
        添加偏好记忆
        """
        self.preferenceMemory.add(user_id, value=value)

    def get_preference_memory(self, user_id: str, query: str) -> str | list[Any]:
        """
        从偏好记忆中检索信息。
        :param user_id
        :param query: 检索内容
        :return: 检索结果
        """
        return self.preferenceMemory.get(user_id, query)
