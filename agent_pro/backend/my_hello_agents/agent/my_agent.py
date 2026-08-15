import os
from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from mq.rabbit_sync_producer import MemorySyncProducer
load_dotenv()
from agent.base import SimpleState
from agent.react import (
    get_final_answer,
    known_tool_names,
    react_llm_node,
    my_tool_node,
    route_after_llm,
)
from utils.context_tool import ContextBuilder, count_tokens
from client.model import get_model
from client.memory_client import memory_manager
from langchain_core.messages import HumanMessage, AIMessage

memory_manager = memory_manager
context_builder = ContextBuilder(memory_manager)
producer = MemorySyncProducer()


def start_node(state: SimpleState) -> SimpleState:
    """
    开始节点
    """
    key = f"{state.user_id}_{state.session_id}"
    memory_manager.add_history_memory(key, HumanMessage(content=state.query))
    return {
        "messages": [HumanMessage(content=state.query)]
    }

def work_memory_node(state: SimpleState) -> dict:
    """
    工作记忆节点, 使用llm判断工作记忆是否改变
    """
    key = f"{state.user_id}_{state.session_id}"
    work_memory = memory_manager.get_work_memory(key)
    model = get_model()
    topic_switch_simple_prompt = """
    判断用户最新问题是否导致话题转换。只回答“变”或“不变”。

    对话历史 ：
    {chat_history}

    用户最新问题：{query}

    你的回答（只输出“变”或“不变”）：
    """
    chat_history = "\n".join([f"{i.type}: {i.content}" for i in work_memory])
    result = model.invoke(topic_switch_simple_prompt.format(chat_history=chat_history, query=state.query))
    if "不变" in result.content:
        work_memory = work_memory
    else:
        producer.send_memory_task(user_id=state.user_id, chat_history=work_memory)
        memory_manager.clear_work_memory(key)
        work_memory = []
    print("[工作记忆]：", work_memory)
    return {
        "work_memory": work_memory
    }

def episodic_memory_node(state: SimpleState) -> dict:
    """
    情景记忆节点, 加载长期情景记忆
    :param state:
    :return:
    """
    episodic_memory = memory_manager.get_episodic_memory(user_id=state.user_id, query=state.query)
    print("[情景记忆]：", episodic_memory)
    return {
        "episodic_memory": episodic_memory
    }

def history_memory_node(state: SimpleState) -> dict:
    """
    历史记忆节点, 加载短期历史记忆
    :param state:
    :return:
    """
    key = f"{state.user_id}_{state.session_id}"
    history_memory = memory_manager.get_history_memory(key)
    print("[历史记忆]：", history_memory)
    return {
        "history_memory": history_memory
    }

def preference_memory_node(state: SimpleState) -> dict:
    """
    偏好记忆节点, 加载偏好记忆
    :param state:
    :return:
    """
    preference_memory = memory_manager.get_preference_memory(state.user_id, state.query)
    print("[偏好记忆]：", preference_memory)
    return {
        "preference_memory": preference_memory
    }

def semantic_memory_node(state: SimpleState) -> dict:
    """
    语义记忆节点, 加载语义记忆
    :param state:
    :return:
    """
    model = get_model()
    prompt = """
    # 角色
    你是一个查询意图分析专家。用户已上传一份知识文档，但检索该文档十分耗时。
    你的任务是快速判断：用户的提问是否“有必要”去检索这份文档才能回答。
    
    # 核心原则
    **默认不需要检索，除非有明确信号。** 宁可漏判（后续可用追问弥补），也不要无事检索，浪费资源。
    
    # 需要检索的情况（满足任一即可）
    1. **明确引用**：用户显式指向了文档或其内容。
       - 例：“文档里怎么说的？”、“根据我上传的文件……”、”上面提到的那个方法是什么？“
    2. **内容依赖**：问题看似通用，但回答需要文档中的具体数据、观点或案例。
       - 例：用户上传了一份《2024年某行业分析报告》，问“今年的市场份额是多少？” → 虽然问题通用，但“今年”的数据很可能在那份报告里。
    3. **指代消解**：用户用了“这个”、“那个”、“它”等代词，指代文档中的概念。
       - 例：刚上传完一份产品说明书，问“这个怎么设置？” → “这个”大概率指说明书中提到的产品。
    4. **延续性问答**：在已经检索过文档的多轮对话中，用户后续的问题如果是对文档已检索内容的追问，仍需继续检索。
       - 例：上一轮已从文档中提取了“三大核心观点”，用户接着问“那第二点能展开说说吗？”
    
    # 不需要检索的情况（即便文档可能包含答案，也暂不检索）
    1. **纯通用知识**：问题可以完全用公开知识回答，且用户没提文档。
       - 例：“什么是机器学习？”、“推荐几个好用的笔记软件”。
       - 即使文档里刚好也写了这些，只要用户没要求“按文档说”，就不检索。
    2. **与文档无关的请求**：创作、翻译、计算、闲聊等，显然不需查阅文档。
       - 例：“帮我写一首诗”、“翻译这段话：Hello”、“1+1等于几？”
    3. **文档范围的常识性外延**：用户问了文档相关领域的常识，但不是要从文档里找答案。
       - 例：上传了一份《Python入门教程》，用户问“Python之父是谁？” → 这是公开常识，不需要查文档。
    
    # 模糊情况的处理策略
    如果无法确定文档中是否包含答案，请做以下权衡：
    - 若问题与文档主题**高度相关**，且查了文档回答质量会显著提升 → **需要**
    - 若问题与文档主题**关联微弱**，更可能是随意提问 → **不需要**
    
    # 输出格式
    严格只输出“需要”或“不需要”
    
    # 待判断的用户查询
    {{query}}
    """
    result = model.invoke(prompt.format(query=state.query))
    print("[是否需要语义记忆判断结果]：", result.content)
    if "不需要" in result.content:
        semantic_memory = []
    else:
        if state.user_id == "666":
            key = "666_999"
        else:
            key = state.user_id
        semantic_memory = memory_manager.get_semantic_memory(key, state.query)
    print("[语义记忆]：", semantic_memory)
    return {
        "semantic_memory": semantic_memory
    }

def context_node(state: SimpleState) -> SimpleState:
    """
    上下文节点, 构建 ReAct 系统提示词(记忆 + 可用工具 + 输出契约)
    :param state:
    :return:
    """
    state.system_prompt = context_builder.build_react(state)
    context_tokens = count_tokens(state.system_prompt)
    print("[上下文]：", state.system_prompt)
    print("[上下文token数]：", context_tokens)
    return state

def memory_update_node(state: SimpleState) -> SimpleState:
    """
    记忆更新节点, 更新工作记忆
    :param state:
    :return:
    """
    key = f"{state.user_id}_{state.session_id}"
    final_answer = get_final_answer(state, known_tool_names(state.user_id, state.active_tools))
    if final_answer:
        memory_manager.add_history_memory(key, AIMessage(content=final_answer))
        # 工作记忆只保留"用户可见"的问答对, 不含工具轮次的中间消息
        memory_manager.add_work_memory(key, [
            HumanMessage(content=state.query),
            AIMessage(content=final_answer),
        ])
    messages = memory_manager.get_work_memory(key)

    if len(messages) > int(os.getenv("MAX_WORKING_MEMORY", 20)):
        # 切分：头部旧消息 + 保留最新消息
        shift_out_messages = messages[:int(os.getenv("SHIFT_NUM", 10))]
        remain_messages = messages[int(os.getenv("SHIFT_NUM", 10)):]

        print(f"【滑动窗口触发】用户{state.user_id}，移出{len(shift_out_messages)}条消息提取长期记忆")
        # 推送被切走的历史消息到MQ，异步提炼
        producer.send_memory_task(user_id=state.user_id, chat_history=shift_out_messages)

        # 更新state，内存只保留最新消息
        memory_manager.update_work_memory(key, remain_messages)
        state.messages = remain_messages

    return state

def end_node(state: SimpleState) -> SimpleState:
    """
    结束节点
    :param state:
    :return:
    """
    return state

builder = StateGraph(SimpleState)
builder.add_node("start", start_node)
builder.add_node("work_memory", work_memory_node)
builder.add_node("episodic_memory", episodic_memory_node)
builder.add_node("history_memory", history_memory_node)
builder.add_node("preference_memory", preference_memory_node)
builder.add_node("semantic_memory", semantic_memory_node)
builder.add_node("context", context_node)
builder.add_node("react_llm", react_llm_node)
builder.add_node("my_tool", my_tool_node)
builder.add_node("memory_update", memory_update_node)
builder.add_node("end", end_node)

builder.add_edge(START, "start")

builder.add_edge("start", "work_memory")
builder.add_edge("start", "episodic_memory")
builder.add_edge("start", "history_memory")
builder.add_edge("start", "preference_memory")
builder.add_edge("start", "semantic_memory")

builder.add_edge("work_memory", "context")
builder.add_edge("episodic_memory", "context")
builder.add_edge("history_memory", "context")
builder.add_edge("semantic_memory", "context")
builder.add_edge("preference_memory", "context")

builder.add_edge("context", "react_llm")
builder.add_conditional_edges(
    "react_llm",
    route_after_llm,
    {"final": "memory_update", "tool": "my_tool"},
)
builder.add_edge("my_tool", "react_llm")
builder.add_edge("memory_update", "end")
builder.add_edge("end", END)

graph = builder.compile()

def get_agent():
    return graph

import asyncio

async def test():
    input_state = SimpleState(
        user_id="666",
        session_id="999",
        query="给我写一篇500字的关于AI的文章",
        messages=[],
    )
    res = await graph.ainvoke(
        input_state,
        config={"recursion_limit": 60},
    )
    print("=" * 40)
    print("最终回答：", get_final_answer(res, known_tool_names(res.user_id, res.active_tools)))

if __name__ == "__main__":
    async def sse_chat():
        input_state = SimpleState(
            user_id="666",
            session_id="999",
            query="给我写一篇500字的关于AI的文章",
            messages=[],
        )
        async for chunk in graph.astream(input_state, config={"recursion_limit": 60}):
            for node_name, update in chunk.items():
                print(f"[{node_name}]")

    asyncio.run(test())
