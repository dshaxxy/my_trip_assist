import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent.base import SimpleState
from agent.react import parse_react_json
from tools.executor import create_tool_executor


class ChatService:
    """封装 Agent 调用与 SSE 事件生成。"""

    RECURSION_LIMIT = 60

    def __init__(self, agent: Any):
        self._agent = agent

    @staticmethod
    def build_initial_state(user_id: str, session_id: str, query: str) -> SimpleState:
        return SimpleState(
            user_id=user_id,
            session_id=session_id,
            query=query,
            messages=[],
        )

    @staticmethod
    def _format_sse(event: str, data: dict) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"

    @staticmethod
    def _node_messages(node_update: Any) -> list:
        """兼容节点返回 dict 或 SimpleState。"""
        if isinstance(node_update, dict):
            return node_update.get("messages", [])
        return getattr(node_update, "messages", [])

    async def stream_chat(
            self,
            user_id: str,
            session_id: str,
            query: str,
    ) -> AsyncIterator[str]:
        input_state = self.build_initial_state(user_id, session_id, query)
        yield self._format_sse(
            "start",
            {"user_id": input_state.user_id, "session_id": input_state.session_id},
        )

        try:
            # 运行时 active_tools 随 activate_skill 更新, 初始取 input_state
            active_tools = list(input_state.active_tools)
            async for update in self._agent.astream(
                    input_state,
                    stream_mode="updates",
                    config={"recursion_limit": self.RECURSION_LIMIT},
            ):
                for node_name, node_update in update.items():
                    if node_name == "react_llm":
                        for event in self._classify_llm_output(
                                node_update, input_state.user_id, active_tools
                        ):
                            yield event
                    elif node_name == "my_tool":
                        if isinstance(node_update, dict) and node_update.get("active_tools") is not None:
                            active_tools = list(node_update["active_tools"])
                        messages = self._node_messages(node_update)
                        last = messages[-1] if messages else None
                        if isinstance(last, HumanMessage):
                            yield self._format_sse("observe", {"observation": last.content})

            yield self._format_sse("message", {"content": "", "done": True})
        except Exception as e:
            yield self._format_sse("error", {"msg": str(e), "done": True})

    def _classify_llm_output(
            self,
            node_update: Any,
            user_id: str,
            active_tools: list[str],
    ) -> list[str]:
        """把 react_llm 节点的模型输出分类为 SSE 事件:
        - 工具轮 JSON(已知工具) -> action 事件(thought/action/action_input)
        - final JSON -> message 事件(action_input.content)
        - 纯文本 -> message 事件(原文)
        """
        messages = self._node_messages(node_update)
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return []

        parsed = parse_react_json(last.content)
        action = parsed.get("action")

        if not action or action == "final":
            content = parsed.get("action_input", {}).get("content") if action == "final" else last.content
            if content:
                return [self._format_sse("message", {"content": content, "done": False})]
            return []

        executor = create_tool_executor(user_id, active_tools)
        if action in executor.tool_names():
            return [self._format_sse("action", {
                "thought": parsed.get("thought", ""),
                "action": action,
                "action_input": parsed.get("action_input", {}),
            })]

        # 未知 action: 视作回答原文, 与 agent 侧 _get_final_answer 保持一致
        return [self._format_sse("message", {"content": last.content, "done": False})]
