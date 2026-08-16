"""calculator skill 的工具实现: 基于 ast 的安全四则运算, 不执行任意代码。"""
import ast
import operator

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"不支持的表达式元素: {type(node).__name__}")


def calculate(expression: str) -> str:
    """安全计算四则运算表达式, 返回结果字符串。"""
    expr = str(expression).strip()
    if not expr:
        return "错误: 表达式为空"
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED):
                return f"错误: 不支持的表达式元素 {type(node).__name__}"
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                return "错误: 仅支持数值运算"
        result = _eval_node(tree.body)
        return str(result)
    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except SyntaxError:
        return "错误: 表达式语法不正确"
    except Exception as e:
        return f"错误: {e}"


TOOL_SPECS = {
    "calculate": {
        "description": "安全计算四则运算表达式, 支持 + - * / // % ** 和括号",
        "parameters": {
            "expression": "要计算的数学表达式字符串, 如 '12 * 17' 或 '(3 + 5) * 2'(必填)",
        },
    },
}

TOOL_IMPLS = {
    "calculate": calculate,
}
