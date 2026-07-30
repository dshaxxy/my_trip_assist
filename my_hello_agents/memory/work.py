from typing import List
from langchain_core.messages import BaseMessage
from my_hello_agents.memory.base import BaseMemory

class WorkMemory(BaseMemory):
    """
    工作记忆
    """
    def __init__(self):
        self.data = {}

    def add(self, key: str, value: List[BaseMessage]):
        """
        将信息添加到工作记忆中。
        :param key: 记忆键 user_id + session_id
        :param value: 记忆值，包含消息的列表
        """
        if key not in self.data:
            self.data[key] = value
        else:
            self.data[key].extend(value)

    def get(self, key: str) -> List[BaseMessage]:
        """
        从工作记忆中检索信息。
        :param key: 记忆键 user_id + session_id
        """
        if key not in self.data:
            return []
        else:
            return self.data[key]
