from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


class SimpleState(BaseModel):
    """
    简单的状态类
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    query: str
    work_memory: list = None
    history_memory: list = None
    semantic_memory: list = None
    episodic_memory: list = None
    preference_memory: str = ""
    system_prompt: str = ""
