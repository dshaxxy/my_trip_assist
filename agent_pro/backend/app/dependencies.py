import sys
from functools import lru_cache
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
MY_AGENTS_ROOT = BACKEND_ROOT / "my_hello_agents"

if str(MY_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(MY_AGENTS_ROOT))

from agent.my_agent import get_agent  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402


@lru_cache
def get_compiled_agent():
    return get_agent()


def get_chat_service() -> ChatService:
    return ChatService(agent=get_compiled_agent())
