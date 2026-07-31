import os
from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from my_hello_agents.mq.rabbit_sync_producer import MemorySyncProducer
load_dotenv()
from my_hello_agents.agent.base import SimpleState
from my_hello_agents.utils.context_tool import ContextBuilder, count_tokens
from my_hello_agents.client.model import get_model
from my_hello_agents.client.memory_client import get_memory_manager
from langchain_core.messages import HumanMessage

memory_manager = get_memory_manager()
context_builder = ContextBuilder(memory_manager)
producer = MemorySyncProducer()


def start_node(state: SimpleState) -> SimpleState:
    """
    开始节点
    """
    key = f"{state.user_id}_{state.session_id}"
    memory_manager.add_history_memory(key, HumanMessage(content=state.query))
    return state

def work_memory_node(state: SimpleState) -> SimpleState:
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
        state.work_memory = work_memory
    else:
        producer.send_memory_task(user_id=state.user_id, chat_history=work_memory)
        memory_manager.clear_work_memory(key)
        state.work_memory = []
    print("[工作记忆]：", work_memory)
    return state

def episodic_memory_node(state: SimpleState) -> SimpleState:
    """
    情景记忆节点, 加载长期情景记忆
    :param state:
    :return:
    """
    episodic_memory = memory_manager.get_episodic_memory(user_id=state.user_id, query=state.query)
    print("[情景记忆]：", episodic_memory)
    state.episodic_memory = episodic_memory
    return state

def history_memory_node(state: SimpleState) -> SimpleState:
    """
    历史记忆节点, 加载短期历史记忆
    :param state:
    :return:
    """
    key = f"{state.user_id}_{state.session_id}"
    history_memory = memory_manager.get_history_memory(key)
    print("[历史记忆]：", history_memory)
    state.history_memory = history_memory
    return state

def preference_memory_node(state: SimpleState) -> SimpleState:
    """
    偏好记忆节点, 加载偏好记忆
    :param state:
    :return:
    """
    preference_memory = memory_manager.get_preference_memory(state.user_id, state.query)
    print("[偏好记忆]：", preference_memory)
    state.preference_memory = preference_memory
    return state

def semantic_memory_node(state: SimpleState) -> SimpleState:
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
        state.semantic_memory = []
    else:
        if state.user_id == "666":
            key = "666_999"
        else:
            key = state.user_id
        state.semantic_memory = memory_manager.get_semantic_memory(key, state.query)
    print("[语义记忆]：", state.semantic_memory)
    return state

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

def llm_node(state: SimpleState) -> SimpleState:
    """
    LLM 节点: 使用 LLM 生成回复
    :param state:
    :return:
    """
    model = get_model()
    prompt = state.system_prompt
    result = model.invoke(prompt.format(query=state.query))
    state.messages.append(result)
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

