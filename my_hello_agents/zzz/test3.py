from typing import Annotated
from dotenv import load_dotenv
from langgraph.constants import END, START
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field
from my_hello_agents.core.llm import MyLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

load_dotenv()

llm = MyLLM()


class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)


def system_prompt_node(state: AgentState) -> dict:
    """系统提示节点：生成系统提示"""
    return {"messages": [SystemMessage(content="你是一个有帮助的助手。")]}


def llm_node(state: AgentState) -> dict:
    # 将系统提示与对话历史合并
    messages = state.messages + [HumanMessage(content="你好")]

    # 调用 LLM
    response = llm.think(messages)

    return {"messages": [HumanMessage(content="你好"),AIMessage(content=response)]}


builder = StateGraph(AgentState)
builder.add_node("system_prompt", system_prompt_node)
builder.add_node("llm", llm_node)

builder.add_edge(START, "system_prompt")
builder.add_edge("system_prompt", "llm")
builder.add_edge("llm", END)

graph = builder.compile()
result = graph.invoke({"messages": []})
print(result)
# # 添加条件边
# builder.add_conditional_edges(
#     "llm",  # 源节点
#     route_after_llm,  # 路由函数
#     {
#         "tools": "tool_executor",  # 返回 "tools" 时 -> tool_executor 节点
#         END: END  # 返回 END 时 -> 结束
#     }
# )
