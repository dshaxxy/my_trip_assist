from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies import get_chat_service
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("/chat")
async def chat(
    body: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    聊天接口（SSE 流式输出）。

    请求体: user_id, session_id, query
    响应: text/event-stream
    """
    return StreamingResponse(
        chat_service.stream_chat(
            user_id=body.user_id,
            session_id=body.session_id,
            query=body.query,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
