import json
import redis
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from my_hello_agents.memory.base import BaseMemory


class HistoryMemory(BaseMemory):
    """
    历史记忆
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.prefix = "history:"

    def add(self, key: str, message: BaseMessage):
        """
        添加历史记忆
        """
        self.redis_client.rpush(self.prefix + key, json.dumps({"role": message.type, "content": message.content, "time": message.additional_kwargs["timestamp"]}))

    def get(self, key: str) -> list[BaseMessage]:
        """
        获取历史记忆
        """
        history = self.redis_client.lrange(self.prefix + key, -10, -1)
        messages = []
        if not history:
            return messages
        for i in history:
            message = json.loads(i)
            if message["role"] == "human":
                messages.append(HumanMessage(content=message["content"], additional_kwargs={"timestamp": message["time"]}))
            else:
                messages.append(AIMessage(content=message["content"]))
        return messages

if __name__ == '__main__':
    memory = HistoryMemory()
    messages = memory.get("666_999")
    for i in messages:
        print(i)