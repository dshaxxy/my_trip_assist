from typing import Any
import tiktoken
from langchain_core.messages import BaseMessage
from qdrant_client.http.models import ScoredPoint

from agent.base import SimpleState
from memory.manager import MemoryManager
from rag.rag_service import Chunk
from utils.prompt_loader import load_system_prompt

def count_tokens(text: str) -> int:
    """计算文本token数（使用tiktoken）"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 降级方案：粗略估算（1 token ≈ 4 字符）
        return len(text) // 4

class ContextBuilder:
    """
    上下文构建器
    gssc流程
    G: gather context
    S: select context
    S: structure context
    C: combine context
    """
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.system_prompt = load_system_prompt()
        self.max_tokens = 4096

    def build(self, state: SimpleState) -> str:
        work_context = self.structure_work_memory(state.work_memory)
        history_context = self.structure_history_memory(state.history_memory)
        preference_context = self.structure_preference_memory(state.preference_memory)
        # history_context_tokens = count_tokens(history_context)
        # preference_context_tokens = count_tokens(preference_context)
        # query_tokens = count_tokens(query)
        # remaining_tokens = self.max_tokens - work_context_tokens - history_context_tokens - preference_context_tokens - query_tokens
        semantic_context = self.structure_semantic_memory(state.semantic_memory)
        episodic_context = self.structure_episodic_memory(state.episodic_memory)
        context = self.system_prompt.format(
            work_context=work_context,
            history_context=history_context,
            semantic_context=semantic_context,
            preference_context=preference_context,
            episodic_context=episodic_context,
            query=state.query
        )
        return context

    def structure_work_memory(self, work_memory: list[BaseMessage]) -> str:
        """结构化工作记忆"""
        context = ""
        if work_memory:
            for message in work_memory:
                context += f"{message.type}: {message.content}\n"
            return context
        else:
            return ""

    def structure_history_memory(self, history_memory: list[BaseMessage]) -> str:
        """结构化历史记忆"""
        context = ""
        if history_memory:
            for message in history_memory:
                if message.type == "human":
                    context += f"{message.additional_kwargs['timestamp']} {message.type}: {message.content}\n"
                else:
                    context += f"{message.type}: {message.content}\n"
            return context
        else:
            return ""

    def structure_semantic_memory(self, semantic_memory: list[Chunk]) -> str:
        """结构化语义记忆"""
        context = ""
        count = 0
        if semantic_memory:
            for chunk in semantic_memory:
                count += 1
                context += f"[资料{count}]：{chunk.text}\n"
            return context
        else:
            return ""

    def structure_episodic_memory(self, episodic_memory: list[ScoredPoint]) -> str:
        """结构化情景记忆"""
        context = ""
        if episodic_memory:
            for point in episodic_memory:
                context += f"{point.payload['time']}: {point.payload['core_event']}\n"
            return context
        else:
            return ""


    def structure_preference_memory(self, preference_memory: str | list[Any]) -> str:
        """结构化偏好记忆"""
        return preference_memory if preference_memory else ""

if __name__ == '__main__':
    tokens = count_tokens("你好")
    print(tokens)