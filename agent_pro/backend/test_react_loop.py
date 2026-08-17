"""隔离测试 ReAct 循环: 不依赖记忆/向量服务, 只跑 react_llm -> my_tool 循环 + 真实模型 API。"""
import asyncio
import os
import sys

sys.path.insert(0, "my_hello_agents")
from dotenv import load_dotenv

load_dotenv(os.path.join("..", "..", ".env"))

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from agent.base import SimpleState
from agent.react import get_final_answer, known_tool_names, my_tool_node, react_llm_node, route_after_llm
from langchain_core.messages import AIMessage, HumanMessage
from skills import render_skills_manifest
from tools.executor import get_available_tool_names, render_tools_section

SYSTEM_TPL = open("my_hello_agents/prompts/react_system_prompt.txt", encoding="utf-8").read()


def build_system_prompt(query: str) -> str:
    return SYSTEM_TPL.format(
        work_context="",
        episodic_context="",
        semantic_context="",
        preference_context="",
        history_context="",
        available_tools=render_tools_section(get_available_tool_names([])),
        skills_manifest=render_skills_manifest(),
        query=query,
    )


def build_graph():
    builder = StateGraph(SimpleState)
    builder.add_node("react_llm", react_llm_node)
    builder.add_node("my_tool", my_tool_node)
    builder.add_edge(START, "react_llm")
    builder.add_conditional_edges(
        "react_llm", route_after_llm, {"final": END, "tool": "my_tool"}
    )
    builder.add_edge("my_tool", "react_llm")
    return builder.compile()


graph = build_graph()


async def run(query: str):
    print("=" * 50)
    print(f"QUERY: {query}")
    state = SimpleState(
        user_id="test",
        session_id="test",
        query=query,
        messages=[HumanMessage(content=query)],
        system_prompt=build_system_prompt(query),
    )
    final = await graph.ainvoke(state, config={"recursion_limit": 30})
    print("最终状态类型:", type(final))
    answer = get_final_answer(final, known_tool_names("test", []))
    print("最终回答:", answer)
    return answer


async def run_skill_flow():
    """确定性验证激活链路: 直接调用 activate_skill_node(模拟模型选择激活),
    验证: 加载 skill.md -> active_tools 更新 -> skill_instructions 注入 -> 真实图里模型调用 calculate -> final。"""
    from agent.react import activate_skill_node
    print("=" * 50)
    query = "帮我精确计算 12 * 17 等于多少"
    print(f"SKILL FLOW: {query}")
    state = SimpleState(
        user_id="test",
        session_id="test",
        query=query,
        messages=[HumanMessage(content=query)],
        system_prompt=build_system_prompt(query),
    )
    # 模拟模型在 ReAct 循环中选择了 activate_skill 并进入 my_tool_node
    res = activate_skill_node(state, {"skill_name": "calculator"})
    state.messages.append(res["messages"][0])  # 观察回灌
    print("激活后 active_tools:", state.active_tools)
    print("激活后 active_skills:", state.active_skills)
    print("skill_instructions 长度:", len(state.skill_instructions))

    # 真实图继续: 模型应看到 calculate 工具并调用它
    final = await graph.ainvoke(state, config={"recursion_limit": 30})
    answer = get_final_answer(final, known_tool_names("test", final.get("active_tools", [])))
    print("最终 active_tools:", final.get("active_tools"))
    print("最终回答:", answer)
    return answer


if __name__ == "__main__":
    async def main():
        from client.mcp_client import mcp_client
        await mcp_client.connect()
        await mcp_client.register_tools()
        # 改进三: professional_qa skill 联网搜索链路(无 TAVILY key 时返回未配置错误, 链路仍通)
        await run("请激活专业知识问答 skill, 然后联网搜索 2026 年上海中考时间")
        # 工具轮 + 最终回答
        await run("现在几点钟了? 请用工具获取准确时间")
        # 纯直接回答(不调用工具)
        await run("1+1等于几? 直接回答")
        # skill 自然激活(模型主动调用 activate_skill)
        await run("请激活计算器 skill, 然后精确计算 12 * 17")
        # skill 激活链路(确定性, 直接走 activate_skill_node)
        await run_skill_flow()
        # 纯逻辑: final JSON 路径
        from langchain_core.messages import AIMessage
        st = SimpleState(
            user_id="test", session_id="test", query="q",
            messages=[AIMessage(content='{"thought": "t", "action": "final", "action_input": {"content": "final答案"}}')],
        )
        print("=" * 50)
        print("final JSON 提取:", get_final_answer(st, known_tool_names("test", [])))

    asyncio.run(main())
