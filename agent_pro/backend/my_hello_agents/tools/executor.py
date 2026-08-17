import datetime
import inspect
import json
from functools import partial
from typing import Any, Callable

from skills import skill_manager

# ============ 工具描述(spec)注册表 ============
# 供提示词渲染:name -> {description, parameters}
# 基础工具常驻;改进2 中 skill 激活后通过 register_tool_spec 追加
BASIC_TOOL_SPECS = {
    "current_time": {
        "description": "获取当前日期和时间,用于回答时效性相关的问题",
        "parameters": {},
    },
    "activate_skill": {
        "description": "加载并激活一个技能(skill), 激活后才能使用该 skill 提供的工具",
        "parameters": {
            "skill_name": "要激活的 skill 名称(必填), 从系统提示的可用技能清单中选择",
        },
    },
}

TOOL_SPECS: dict[str, dict] = dict(BASIC_TOOL_SPECS)


def register_tool_spec(name: str, description: str, parameters: dict | None = None):
    TOOL_SPECS[name] = {
        "description": description,
        "parameters": parameters or {},
    }


def render_tools_section(names: list[str]) -> str:
    """把工具名列表渲染成提示词里的可用工具段。"""
    if not names:
        return "当前没有可用工具。"
    lines = ["当前可用工具:", ""]
    for n in names:
        spec = TOOL_SPECS.get(n)
        if not spec:
            continue
        lines.append(f"### {n}")
        lines.append(f"用途: {spec['description']}")
        lines.append(f"action_input: {json.dumps(spec['parameters'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).strip()


def get_available_tool_names(active_tools: list[str]) -> list[str]:
    """基础工具 + 已激活工具(去重)。"""
    names = list(BASIC_TOOL_SPECS.keys())
    for n in active_tools:
        if n not in names:
            names.append(n)
    return names


class ToolExecutor:
    """工具执行器:action 名 -> 可调用函数。模型只负责"点菜",实际执行在这里。"""

    def __init__(self):
        self._funcs: dict[str, Callable] = {}
        self.register("current_time", self._current_time)
        # activate_skill 由 agent 节点的 my_tool_node 特判处理, 这里只保证路由可识别
        self.register("activate_skill", self._activate_skill)

    @staticmethod
    def _activate_skill(**kwargs) -> str:
        return "activate_skill 需由 agent 节点处理, 不应直接执行。"

    def register(self, name: str, func: Callable):
        self._funcs[name] = func

    def has(self, name: str) -> bool:
        return name in self._funcs

    def tool_names(self) -> set[str]:
        return set(self._funcs.keys())

    async def execute(self, action: str, action_input: dict | None) -> str:
        """执行工具, 返回 observation 字符串。支持同步与 async 实现; 任何异常转成可读观察结果。"""
        func = self._funcs.get(action)
        if func is None:
            return f"错误: 未知工具 {action},请检查工具名,或直接用 final 回答用户。"
        try:
            result = func(**(action_input or {}))
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except TypeError as e:
            return f"工具 {action} 参数错误: {e}"
        except Exception as e:
            return f"工具 {action} 执行出错: {type(e).__name__}: {e}"

    @staticmethod
    def _current_time() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_tool_executor(user_id: str, active_tools: list[str]) -> ToolExecutor:
    """按请求构建执行器。active_tools 里来自已激活 skill 的工具从 SkillManager 取实现注册;
    实现签名含 user_id 参数的工具(如 rag_retrieve)用 partial 绑定当前用户。"""
    executor = ToolExecutor()
    for tool_name in active_tools or []:
        impl = skill_manager.get_tool_impl(tool_name)
        if impl is None:
            continue
        sig = inspect.signature(impl)
        if "user_id" in sig.parameters:
            impl = partial(impl, user_id=user_id)
        executor.register(tool_name, impl)
    return executor
