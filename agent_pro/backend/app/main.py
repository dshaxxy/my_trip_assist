from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时连接 MCP server 并注册工具; 失败不影响主服务。"""
    try:
        from client.mcp_client import mcp_client
        await mcp_client.connect()
        await mcp_client.register_tools()
    except Exception as e:
        print(f"[MCP] 连接/注册失败(不影响主服务): {type(e).__name__}: {e}")
    yield
    try:
        from client.mcp_client import mcp_client
        await mcp_client.close()
    except Exception:
        pass


app = FastAPI(
    title="My Trip Assist API",
    description="基于多层记忆的聊天机器人后端",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
