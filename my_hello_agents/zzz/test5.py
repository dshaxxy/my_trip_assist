import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import ast
import operator

load_dotenv()


# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息。"""
    return f"关于 '{query}' 的搜索结果：这是模拟的搜索结果..."


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。"""
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def safe_eval(node):
        if isinstance(node, ast.Expression):
            return safe_eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            return ops[type(node.op)](operand)
        else:
            raise ValueError(f"不支持的表达式类型: {type(node)}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city} 今日天气：晴，温度 22C，湿度 60%"


tools = [search_web, calculate, get_weather]

# 初始化 LLM 并绑定工具
llm = ChatOpenAI(
    model=os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-pro'),
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
    temperature=0
)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


def agent_node(state: MessagesState) -> dict:
    """Agent 推理节点：调用 LLM 决定下一步行动"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 构建 ReAct 图
builder = StateGraph(MessagesState)

# 添加节点
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))  # 内置 ToolNode 自动处理工具调用

# 添加边
builder.add_edge(START, "agent")

# 条件路由：如果 LLM 请求工具则执行工具，否则结束
builder.add_conditional_edges(
    "agent",
    tools_condition,  # 内置路由函数
    {
        "tools": "tools",
        END: END
    }
)

# 工具执行完后返回 agent 继续推理
builder.add_edge("tools", "agent")

graph = builder.compile()

# 测试
result = graph.invoke({
    "messages": [HumanMessage(content="北京今天天气如何？另外帮我计算 1234 * 5678, 一次只能调用一个工具")]
})

for message in result["messages"]:
    print(f"[{message.type}]: {message.content[:200] if message.content else '(工具调用)'}")
#
# print(result["messages"])