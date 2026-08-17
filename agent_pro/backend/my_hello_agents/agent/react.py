"""ReAct 循环机制: 模型输出解析、工具执行、路由。不依赖记忆/外部存储, 便于独立测试。"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.base import SimpleState
from client.model import get_model
from skills import skill_manager
from tools.executor import create_tool_executor


def parse_react_json(content: str) -> dict:
    """解析模型输出, 兼容 ```json 代码块包裹与前后多余文字。解析失败返回空 dict。"""
    if not content:
        return {}
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def is_tool_round(content: str, known_tools: set[str]) -> bool:
    """判断模型输出是否为工具调用轮: 能解析出已注册工具名(且非 final)的 action。"""
    parsed = parse_react_json(content)
    action = parsed.get("action")
    return bool(action) and action != "final" and action in known_tools


def get_final_answer(state, known_tools: set[str]) -> str:
    """从消息中提取最终回答:
    - final JSON -> action_input.content
    - 纯文本(非JSON) -> 原文
    - 工具调用轮 JSON -> 跳过继续向上找
    state 兼容 SimpleState 与 langgraph 返回的 dict。
    """
    messages = state.messages if hasattr(state, "messages") else state.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        parsed = parse_react_json(msg.content)
        if not parsed:
            return msg.content
        action = parsed.get("action")
        if action is None:
            return msg.content
        if action == "final":
            return parsed.get("action_input", {}).get("content", "")
        if action in known_tools:
            continue
        return msg.content
    return ""


def known_tool_names(user_id: str, active_tools: list[str]) -> set[str]:
    """当前请求实际可执行的工具名集合(基础工具 + 已激活 skill 工具)。"""
    return create_tool_executor(user_id, active_tools).tool_names()


async def react_llm_node(state: SimpleState) -> dict:
    """
    LLM 节点(ReAct): 组装 SystemMessage + 对话历史调用模型。
    模型按契约输出: 工具轮 JSON(thought/action/action_input) 或 最终回答纯文本。
    已激活 skill 的指令段(全文 + 工具描述)以追加的 SystemMessage 注入, 模型即可"看到"新工具。
    """
    model = get_model()
    messages = [SystemMessage(content=state.system_prompt)]
    if getattr(state, "skill_instructions", ""):
        messages.append(SystemMessage(content=state.skill_instructions))
    messages.extend(list(state.messages))
    result = await model.ainvoke(messages)
    content = result.content
    print("[模型输出]：", content)
    return {
        "messages": [AIMessage(content=content)]
    }


def activate_skill_node(state: SimpleState, action_input: dict) -> dict:
    """处理 activate_skill: 加载 skill.md 全文, 工具名加入 active_tools,
    注入 skill 指令段, 使下一轮模型可见新工具。"""
    skill_name = str(action_input.get("skill_name", "")).strip()
    skill = skill_manager.load_skill(skill_name)
    if skill is None:
        observation = f"错误: 未找到 skill「{skill_name}」, 请检查清单中的技能名或用 final 直接回答。"
        return {
            "messages": [HumanMessage(content=f"[工具观察 activate_skill] {observation}")]
        }

    if skill_name not in state.active_skills:
        state.active_skills.append(skill_name)
        for tool in skill.tools:
            if tool not in state.active_tools:
                state.active_tools.append(tool)
        skill_manager.register_skill_tools(skill)
        instruction = skill_manager.render_skill_instructions(skill)
        state.skill_instructions = (
            (state.skill_instructions + "\n\n" + instruction).strip()
            if state.skill_instructions else instruction
        )
    observation = f"已激活 skill「{skill_name}」, 当前可用工具: {state.active_tools}"
    print(f"[Skill 激活] {skill_name}, active_tools={state.active_tools}")
    return {
        "messages": [HumanMessage(content=f"[工具观察 activate_skill] {observation}")],
        "active_tools": state.active_tools,
        "active_skills": state.active_skills,
        "skill_instructions": state.skill_instructions,
    }


async def my_tool_node(state: SimpleState) -> dict:
    """
    工具执行节点: 解析模型输出的 action, 交给 ToolExecutor 执行,
    将 observation 以 HumanMessage 回灌进消息流, 再送回大模型。
    activate_skill 需更新 state(active_tools/指令注入), 因此特判处理。
    """
    last = state.messages[-1]
    parsed = parse_react_json(last.content)
    action = parsed.get("action", "")
    action_input = parsed.get("action_input", {}) or {}

    if action == "activate_skill":
        return activate_skill_node(state, action_input)

    executor = create_tool_executor(state.user_id, state.active_tools)
    observation = await executor.execute(action, action_input)
    print(f"[工具执行] {action} <- {action_input}")
    print(f"[观察结果] {observation}")
    return {
        "messages": [HumanMessage(content=f"[工具观察 {action}] {observation}")]
    }


def route_after_llm(state: SimpleState) -> str:
    """ReAct 循环路由: 工具轮 -> my_tool, 否则(final或纯文本) -> memory_update。"""
    last = state.messages[-1]
    if isinstance(last, AIMessage):
        executor = create_tool_executor(state.user_id, state.active_tools)
        if is_tool_round(last.content, executor.tool_names()):
            return "tool"
    return "final"
