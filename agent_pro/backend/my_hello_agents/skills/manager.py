"""Skill 管理器: 渐进式披露的核心。

- 启动时扫描 skills/ 目录, 解析每个 skill.md 的 frontmatter 得到清单(仅 name/description/tools)
- load_skill(name) 读取 skill.md 全文 + 导入 skills.{name} 模块收集工具 spec 与实现
- render_skill_instructions() 把 skill 全文与工具描述拼成注入提示词的指令段
"""
import importlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from utils.path_tool import get_abs_path


@dataclass
class Skill:
    name: str
    description: str
    tools: list[str]
    content: str          # skill.md 全文
    specs: dict = field(default_factory=dict)      # tool_name -> {"description", "parameters"}
    impls: dict = field(default_factory=dict)      # tool_name -> callable


class SkillManager:
    def __init__(self, skills_dir: str | None = None):
        self.skills_dir = Path(skills_dir or get_abs_path("skills"))
        self._catalog: dict[str, dict] = {}   # name -> {"description", "tools"}
        self._tool_specs: dict[str, dict] = {}  # tool_name -> spec(全局)
        self._tool_impls: dict[str, callable] = {}  # tool_name -> callable(全局)
        self._discover()

    # ============ 扫描与加载 ============

    def _discover(self):
        """扫描 skills/*/skill.md, 解析 frontmatter 并导入工具模块。"""
        if not self.skills_dir.exists():
            return
        for skill_md in self.skills_dir.glob("*/skill.md"):
            name = skill_md.parent.name
            meta = self._parse_frontmatter(skill_md)
            self._catalog[name] = meta
            try:
                module = importlib.import_module(f"skills.{name}")
                self._tool_specs.update(getattr(module, "TOOL_SPECS", {}))
                self._tool_impls.update(getattr(module, "TOOL_IMPLS", {}))
            except ImportError:
                print(f"[skill] 导入 {name} 工具模块失败, 工具将不可用")

    @staticmethod
    def _parse_frontmatter(skill_md: Path) -> dict:
        """解析 skill.md 的 --- 包裹的 frontmatter。"""
        text = skill_md.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                meta = yaml.safe_load(parts[1]) or {}
                return {
                    "description": str(meta.get("description", "")),
                    "tools": list(meta.get("tools", []) or []),
                }
        return {"description": "", "tools": []}

    def list_skills(self) -> list[dict]:
        """渐进披露清单: 只暴露 name/description/tools, 不读全文。"""
        return [
            {"name": name, **meta}
            for name, meta in self._catalog.items()
        ]

    def load_skill(self, name: str) -> Skill | None:
        """读取 skill.md 全文并组装 Skill(工具 spec/impl 来自已扫描的全局表)。"""
        skill_md = self.skills_dir / name / "skill.md"
        if not skill_md.exists():
            return None
        content = skill_md.read_text(encoding="utf-8")
        meta = self._catalog.get(name, {"description": "", "tools": []})
        tools = list(meta.get("tools", []))
        return Skill(
            name=name,
            description=meta.get("description", ""),
            tools=tools,
            content=content,
            specs={t: self._tool_specs[t] for t in tools if t in self._tool_specs},
            impls={t: self._tool_impls[t] for t in tools if t in self._tool_impls},
        )

    # ============ 工具注册与提示词渲染 ============

    def get_tool_impl(self, tool_name: str):
        return self._tool_impls.get(tool_name)

    def register_skill_tools(self, skill: Skill):
        """把 skill 声明且已实现的工具 spec 注册进提示词渲染用的 TOOL_SPECS。"""
        from tools.executor import TOOL_SPECS, register_tool_spec
        for tool, spec in skill.specs.items():
            register_tool_spec(
                tool,
                spec.get("description", ""),
                spec.get("parameters", {}),
            )

    def render_skill_instructions(self, skill: Skill) -> str:
        """拼接注入提示词的指令段: skill 全文 + 本 skill 可用工具描述。"""
        from tools.executor import render_tools_section
        lines = [
            f"# 已激活 Skill: {skill.name}",
            "",
            skill.content.strip(),
            "",
            "## 本 Skill 可用工具",
            render_tools_section(skill.tools),
        ]
        return "\n".join(lines)


skill_manager = SkillManager()


def render_skills_manifest(active_skills: list[str] | None = None) -> str:
    """渲染可用 skill 清单(渐进披露: 只暴露 name/description/tools, 不读全文)。"""
    catalog = skill_manager.list_skills()
    if not catalog:
        return "暂无可用技能。"
    lines = ["以下是可用的技能清单。当用户请求与某个技能的能力匹配时, 你必须先通过 activate_skill 激活该技能, 激活后才能看到并使用其工具:", ""]
    for skill in catalog:
        lines.append(f"- {skill['name']}: {skill['description']}")
    return "\n".join(lines)
