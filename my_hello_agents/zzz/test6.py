import datetime
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph

from my_hello_agents.memory.manager import MemoryManager


class SimpleState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str


load_dotenv()

# 初始化 LLM
llm = ChatOpenAI(
    model=os.getenv('LLM_MODEL_ID', 'qwen3.7-plus'),
    openai_api_key=os.getenv('LLM_API_KEY'),
    openai_api_base=os.getenv('LLM_BASE_URL', 'https://ws-06luduvpfh4renzg.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'),
    temperature=0
)
user_id = "666"
session_id = "999"
memory_id = f"{user_id}_{session_id}"

memory = MemoryManager()

def system_prompt_node(state: SimpleState) -> SimpleState:
    """系统提示节点：生成系统提示"""
    messages = memory.get_history_memory(memory_id)
    print("============================redis_history======================================")
    for i in messages:
        print(type(i), i.content)
    request = HumanMessage(content=state["query"], additional_kwargs={"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    memory.add_history_memory(memory_id, request)
    return {"messages": [SystemMessage(content="你是一个 helpful 的助手。"), *messages, request]}

def llm_node(state: SimpleState) -> SimpleState:
    """LLM 节点：调用 LLM 生成回复"""
    response = llm.invoke(state["messages"])
    response.additional_kwargs["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"messages": [response]}

def memory_node(state: SimpleState) -> SimpleState:
    """记忆节点：将消息保存到记忆中"""
    memory.add_history_memory(memory_id, state["messages"][-1])
    memory.add_work_memory(memory_id, state["messages"])
    return state

builder = StateGraph(SimpleState)
builder.add_node("system_prompt", system_prompt_node)
builder.add_node("llm", llm_node)
builder.add_node("memory", memory_node)
builder.add_edge(START, "system_prompt")
builder.add_edge("system_prompt", "llm")
builder.add_edge("llm", "memory")
builder.add_edge("memory", END)
graph = builder.compile()

result = graph.invoke({"messages": [], "query": "那12个人呢"})
print("============================result======================================")
for i in result["messages"]:
    print(i)
print("============================memory======================================")
for i in memory.get_history_memory(memory_id):
    print(i)



