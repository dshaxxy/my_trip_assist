import datetime
import json

from dotenv import load_dotenv

load_dotenv()
import re
import uuid
from client.model import model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, \
    ScoredPoint
from client.qdrant import get_qdrant_client, qdrant_client
from my_hello_agents.memory.base import BaseMemory
from rag.embedding import EmbeddingService
from utils.prompt_loader import load_summary_prompt


class EpisodicMemory(BaseMemory):
    """
    情景记忆
    """

    def __init__(self, summary_model: ChatOpenAI):
        self.summary_model = summary_model
        self.qdrant_client = qdrant_client
        self.summary_prompt = load_summary_prompt()
        self.collection_name = "episodic_memory"
        # 创建集合（如果不存在）
        self._init_collection()

    def _init_collection(self):
        """初始化集合"""
        collections = self.qdrant_client.get_collections().collections
        if not self.qdrant_client.collection_exists(self.collection_name):
            print("集合不存在，正在创建...")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # 根据你的模型调整
                    distance=Distance.COSINE  # 余弦相似度
                )
            )

    def add(self, user_id: str, messages: list[BaseMessage]):
        """
        将工作记忆信息通过llm总结
        然后向量化
        最后添加到情景记忆中。
        :param user_id: 用户id
        :param messages: 记忆消息列表
        """
        messages1 = [i.model_dump() for i in messages]
        formatted_messages = json.dumps(messages1, ensure_ascii=False)
        summary = self.summary_model.invoke(self.summary_prompt.format(formatted_memory=formatted_messages))
        summary_dict = json.loads(summary.content)
        text_to_embed = f"{summary_dict['time']} {summary_dict['core_event']} "
        if "无" not in summary_dict['key_findings']:
            text_to_embed += f"{summary_dict['key_findings']}"
        if "无" not in summary_dict['action_items']:
            text_to_embed += f"{summary_dict['action_items']}"
        vector = EmbeddingService.encode(text_to_embed)
        point_id = str(uuid.uuid4())
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "time": summary_dict["time"],
                        "participants": summary_dict["participants"],
                        "core_event": summary_dict["core_event"],
                        "key_findings": summary_dict["key_findings"],
                        "action_items": summary_dict["action_items"],
                        "importance": summary_dict["importance"],
                    }
                )
            ]
        )
        print(f"情景记忆已成功存入 Qdrant，user_id={user_id}，point_id={point_id}")
        print(text_to_embed, f" importance={summary_dict['importance']}")

    def get(self, user_id: str, query: str) -> list[ScoredPoint]:
        """
        从情景记忆中检索信息。
        :param user_id: 用户id
        :param query: 查询内容
        :return: 检索到的信息列表
        """
        query_vector = EmbeddingService.encode(query)
        result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=8,
            with_payload=True,
            with_vectors=True,
        )
        if not result.points:
            return []
        # 重排, 根据importance和 时间近因性 重新计算score
        now = datetime.datetime.now()
        # 当前只保留年月日，统一对比日期
        today = datetime.date(year=now.year, month=now.month, day=now.day)
        half_life_day = 7  # 半衰期7天，只按天计算
        candidate_list = []

        for point in result.points:
            payload = point.payload
            sim_score = point.score
            importance = float(payload.get("importance", 0.0))
            time_str = payload.get("time", "未知")

            memory_dt = parse_memory_time_only_date(time_str)
            if memory_dt is not None:
                # 只计算相差天数
                delta_days = (today - memory_dt).days
                # 日期衰减因子：间隔天数越多，分数越低
                time_factor = pow(0.5, delta_days / half_life_day)
            else:
                time_factor = 0.1

            # 综合加权打分
            final_score = (
                    0.70 * sim_score
                    + 0.20 * importance
                    + 0.10 * time_factor
            )
            point.score = final_score
            candidate_list.append(point)
        # 综合分数降序排序
        candidate_list.sort(key=lambda p: p.score, reverse=True)
        return candidate_list

def parse_memory_time_only_date(time_text: str) -> datetime.date | None:
    if not time_text or time_text == "未知":
        return None
    # 格式2：2026年7月24日上午 / 2026年7月24日下午
    pattern = r"(\d{4})-(\d{1,2})-(\d{1,2})"
    match = re.search(pattern, time_text.split(" ")[0])
    if match:
        year, month, day = map(int, match.groups())
        return datetime.date(year=year, month=month, day=day)
    # 无法解析
    return None

if __name__ == '__main__':
    # history_memory = HistoryMemory()
    # messages = history_memory.get("666_999")
    # episodic_memory = EpisodicMemory()
    # episodic_memory.add("666_999", messages)
    model = model
    episodic_memory = EpisodicMemory(model)
    points = episodic_memory.get("666", "我记得之前问过你分苹果的事情是不是, 是什么时候来着")
    if points:
        for i in points:
            print(i)
