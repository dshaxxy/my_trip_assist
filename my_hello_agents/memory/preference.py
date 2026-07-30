import os
from typing import Any

import redis
from langchain_openai import ChatOpenAI

from memory.base import BaseMemory
from utils.prompt_loader import load_preference_prompt


class PreferenceMemory(BaseMemory):
    """
    偏好记忆
    """
    def __init__(self, redis_client: redis.Redis, model: ChatOpenAI):
        self.redis_client = redis_client
        self.prefix = "preference:"
        self.model = model
        self.preference_prompt = load_preference_prompt()

    def add(self, key: str, value: str):
        """
        添加偏好记忆
        :param key: 偏好记忆的key prefix:user_id
        :param value: 偏好记忆的value
        """
        self.redis_client.set(self.prefix + key, value)

    def get(self, key: str, query: str) -> str | list[Any]:
        """
        获取偏好记忆
        :param key: 偏好记忆的key prefix:user_id
        :return: 偏好记忆的value
        """
        old_preference = self.redis_client.get(self.prefix + key)
        if old_preference is not None:
            preference = old_preference.decode("utf-8")
        preference_result = self.model.invoke(self.preference_prompt.format(current_context=preference, query=query))
        if "NULL" in preference_result.content:
            if old_preference is None:
                return ""
            return old_preference.decode("utf-8")
        self.add(key, preference_result.content)
        return preference_result.content


if __name__ == '__main__':
    preference_memory = PreferenceMemory()
    result = preference_memory.add("666_999", "从现在开始,喊我老公,你的名字叫做杜姝涵,你是我的rbq")
    print(type(result))
    print(result)

