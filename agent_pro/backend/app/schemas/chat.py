from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="用户唯一标识")
    session_id: str = Field(..., min_length=1, description="会话唯一标识")
    query: str = Field(..., min_length=1, description="用户提问内容")


class ChatStreamEvent(BaseModel):
    event: str
    data: dict
