import os
from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import StreamWriter

from mq.rabbit_sync_producer import MemorySyncProducer
load_dotenv()
from agent.base import SimpleState
from utils.context_tool import ContextBuilder, count_tokens
from client.model import get_model
from client.memory_client import memory_manager
from langchain_core.messages import HumanMessage

memory_manager = memory_manager
context_builder = ContextBuilder(memory_manager)
producer = MemorySyncProducer()


def start_node(state: SimpleState) -> SimpleState:
    """
    开始节点
    """
    key = f"{state.user_id}_{state.session_id}"
    memory_manager.add_history_memory(key, HumanMessage(content=state.query))
    return state

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
    请根据用户最新问题，判断是否需要加载语义记忆。
    用户最新问题：{query}
    你的回答（只输出“需要”或“不需要”）：
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
    上下文节点, 加载上下文
    :param state:
    :return:
    """
    state.system_prompt = context_builder.build(state)
    context_tokens = count_tokens(state.system_prompt)
    print("[上下文]：", state.system_prompt)
    print("[上下文token数]：", context_tokens)
    return state

# async def llm_node(state: SimpleState) -> SimpleState:
#     """
#     LLM 节点: 使用 LLM 生成回复
#     :param state:
#     :return:
#     """
#     model = get_model()
#     prompt = state.system_prompt
#     result = model.invoke(prompt.format(query=state.query))
#     state.messages.append(result)
#     return state

# 添加 stream_writer 参数！
# async def llm_node(state: SimpleState, stream_writer: StreamWriter) -> SimpleState:
#     model = get_model()
#     prompt = state.system_prompt.format(query=state.query)
#
#     full_response = ""
#     # 使用异步流式 astream
#     async for chunk in model.astream(prompt):
#         text_piece = chunk.content
#         if not text_piece:
#             continue
#         full_response += text_piece
#         # 向外推送增量文本，自定义你想要的数据格式
#         stream_writer.write({
#             "type": "llm_chunk",
#             "content": text_piece
#         })
#     # 最终完整消息存入state，供后续 memory_update 使用
#     from langchain_core.messages import AIMessage
#     state.messages.append(AIMessage(content=full_response))
#     return state

async def llm_node(state: SimpleState) -> SimpleState:
    model = get_model()
    prompt_text = state.system_prompt.format(query=state.query)

    full_response = ""
    async for chunk in model.astream(prompt_text):
        text_piece = chunk.content
        if not text_piece:
            continue
        full_response += text_piece

    from langchain_core.messages import AIMessage
    state.messages.append(AIMessage(content=full_response))
    return state

def memory_update_node(state: SimpleState) -> SimpleState:
    """
    记忆更新节点, 更新工作记忆
    :param state:
    :return:
    """
    key = f"{state.user_id}_{state.session_id}"
    memory_manager.add_history_memory(key, state.messages[-1])
    memory_manager.add_work_memory(key, state.messages)
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
builder.add_node("llm", llm_node)
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

builder.add_edge("context", "llm")
builder.add_edge("llm", "memory_update")
builder.add_edge("memory_update", "end")
builder.add_edge("end", END)

graph = builder.compile()

def get_agent():
    return graph

# import asyncio
#
# async def test():
#     res = await llm_node(SimpleState(user_id="666", session_id="999", query="给我写一篇500字的关于AI的文章", messages=[]))
#     print(res)
#
# if __name__ == "__main__":
#     async def sse_chat():
#         input_state = SimpleState(
#             user_id="666",
#             session_id="999",
#             query="给我写一篇500字的关于AI的文章",
#             messages=[]
#         )
#         config = {"configurable": {"thread_id": f"666_999"}}
#
#         async for event in graph.astream_events(input_state, config=config, version="v2"):
#             # 只捕获 llm 节点内部 LLM 的输出
#             node_name = event["metadata"].get("langgraph_node")
#             if (
#                     event["event"] == "on_chat_model_stream"
#                     and node_name == "llm"
#             ):
#                 chunk = event["data"]["chunk"]
#                 text_piece = chunk.content
#                 if text_piece:
#                     print(text_piece, end="")
#                     # yield build_sse("message", {"content": text_piece, "done": False})
#
#     asyncio.run(sse_chat())
