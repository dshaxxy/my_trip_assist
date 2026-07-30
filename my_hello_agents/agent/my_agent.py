import datetime

from dotenv import load_dotenv

load_dotenv()
from agent.base import SimpleState
from utils.context_tool import ContextBuilder, count_tokens
from langchain_core.messages import HumanMessage
from client.model import get_model
from memory.manager import MemoryManager

memory_manager = MemoryManager()
context_builder = ContextBuilder(memory_manager)


def start_node(state: SimpleState) -> SimpleState:
    """
    开始节点
    """
    key = f"{state.user_id}_{state.session_id}"
    memory_manager.add_history_memory(key, HumanMessage(content=state.query, additional_kwargs={"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}))
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
        # todo 话题转变，清空工作记忆 发送消息到消息队列去异步转化为长期记忆
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
    # todo: 工作记忆相关操作  超出阈值转化为长期记忆

    return state

def end_node(state: SimpleState) -> SimpleState:
    """
    结束节点
    :param state:
    :return:
    """
    return state


if __name__ == '__main__':
    state = SimpleState(user_id="666", session_id="999", query="人体循环系统由什么组成？体循环和肺循环作用是什么？", messages=[])
    state1 = start_node(state)
    state2 = work_memory_node(state1)
    state3 = episodic_memory_node(state2)
    state4 = history_memory_node(state3)
    state5 = preference_memory_node(state4)
    state6 = semantic_memory_node(state5)
    context_node(state6)

