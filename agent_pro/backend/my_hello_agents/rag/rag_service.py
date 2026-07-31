import json
import uuid
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from client.model import model
from client.neo4j_client import neo4j_client
from client.qdrant import qdrant_client
from rag.embedding import EmbeddingService
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_extract_prompt


class Chunk(BaseModel):
    chunk_id: str
    type: str
    text: str
    score: float


class RagService:
    def __init__(self):
        self.collection_name = "vector_store"
        self.spilt = self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len
        )
        self.qdrant_client = qdrant_client
        self.neo4j_client = neo4j_client
        self.model = model
        self.system_prompt = load_extract_prompt()
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

    def add_to_neo4j_qdrant(self, user_id: str, file_path: str, password: str = None):
        path = get_abs_path(file_path)
        doc_name = path.split("/")[-1]
        documents = PyPDFLoader(path, password, mode="single").load()
        chunks = self.spliter.split_documents(documents)
        with self.neo4j_client.session() as session:
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id, user_id: $user_id})
                ON CREATE SET d.name = $doc_name, d.created_at = datetime()
                SET d.updated_at = datetime()
                """,
                doc_id=f"{user_id}_{doc_name}",
                doc_name=doc_name,
                user_id=user_id
            )
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            self._add_to_qdrant(user_id, chunk, chunk_id)
            chunks_with_entities = self._get_chunks_with_entities(chunk)
            entitys = chunks_with_entities.get("entities", [])
            for entity in entitys:
                print(entity)
            relations = chunks_with_entities.get("relations", [])
            for rel in relations:
                print(rel)
            print("===" * 30)
            self._add_to_neo4j(user_id, chunk, chunk_id, doc_name, chunks_with_entities)



    def _add_to_qdrant(self, user_id: str, chunk: Document, chunk_id: str):
        vector = EmbeddingService.encode(chunk.page_content)
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                )
            ]
        )

    def _add_to_neo4j(self, user_id: str, chunk: Document, chunk_id: str, doc_name: str, chunks_with_entities):
        # 2. 创建 Chunk 节点，并关联到 Document
        with self.neo4j_client.session() as session:
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id, user_id: $user_id})
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    text: $text,
                    page: $page,
                    order: $order,
                    user_id: $user_id
                })
                CREATE (d)-[:CONTAINS {user_id: $user_id}]->(c)
                """,
                doc_id=f"{user_id}_{doc_name}",
                chunk_id=chunk_id,
                text=chunk.page_content,
                page=chunk.metadata.get("page", 0),
                order=chunk.metadata.get("order", 0),
                user_id=user_id
            )
            # 3. 为当前 Chunk 创建或合并 Entity，并建立 MENTIONS 关系
            for entity in chunks_with_entities.get("entities", []):
                # 使用命名空间隔离，确保实体唯一
                full_entity_name = f"{user_id}_{entity['name']}"

                session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $chunk_id, user_id: $user_id})
                    MERGE (e:Entity {
                        entity_id: $entity_id,
                        user_id: $user_id
                    })
                    ON CREATE SET 
                        e.name = $full_name,
                        e.display_name = $display_name,
                        e.type = $type,
                        e.created_at = datetime()
                    ON MATCH SET 
                        e.last_seen_at = datetime()
                    MERGE (c)-[:MENTIONS {user_id: $user_id}]->(e)
                    """,
                    chunk_id=chunk_id,
                    entity_id=f"{user_id}_{entity['name']}",
                    full_name=full_entity_name,
                    display_name=entity["name"],
                    type=entity.get("type", "UNKNOWN"),
                    user_id=user_id
                )

            # 4. 处理 Entity 之间的关系（如果大模型提取了关系）
            for rel in chunks_with_entities.get("relations", []):
                from_entity_id = f"{user_id}_{rel['from']}"
                to_entity_id = f"{user_id}_{rel['to']}"

                session.run(
                    """
                    MATCH (e1:Entity {entity_id: $from_id, user_id: $user_id})
                    MATCH (e2:Entity {entity_id: $to_id, user_id: $user_id})
                    MERGE (e1)-[r:RELATES_TO {
                        rel_type: $rel_type,
                        user_id: $user_id
                    }]->(e2)
                    """,
                    from_id=from_entity_id,
                    to_id=to_entity_id,
                    rel_type=rel.get("type", "RELATED"),
                    user_id=user_id
                )

    # """
    # chunks_with_entities=[
    #             {
    #                 "entities": [
    #                     {"name": "巴黎", "type": "LOCATION"},
    #                     {"name": "法国", "type": "LOCATION"},
    #                     {"name": "埃菲尔铁塔", "type": "LANDMARK"}
    #                 ],
    #                 "relations": [
    #                     {"from": "巴黎", "to": "法国", "type": "CAPITAL_OF"},
    #                     {"from": "埃菲尔铁塔", "to": "巴黎", "type": "LOCATED_IN"}
    #                 ]
    #             }
    #         ]
    # """
    def _get_chunks_with_entities(self, chunk: Document) -> dict:
        message = self.model.invoke(self.system_prompt.format(text=chunk.page_content))
        chunks_with_entities = json.loads(message.content)
        return chunks_with_entities

    def retriever_chunks(self, user_id: str, query: str) -> list[Chunk]:
        qdrant_chunks = self._retriever_from_qdrant(user_id, query)
        neo4j_chunks = self._retriever_from_neo4j(user_id, query)
        # 重排
        rearranged_chunks = self._rearrange_chunks(qdrant_chunks, neo4j_chunks)
        return rearranged_chunks

    def _retriever_from_qdrant(self, user_id: str, query: str) -> list[Chunk]:
        vector = EmbeddingService.encode(query)
        result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=6,
            with_payload=True,
            with_vectors=True,
        )
        if not result.points:
            return []
        chunk_list = []
        for chunk in result.points:
            if chunk.score < 0.55:
                continue
            chunk_list.append(
                Chunk(chunk_id=chunk.id, type="qdrant_chunk", text=chunk.payload["content"], score=chunk.score))
        return chunk_list

    def _retriever_from_neo4j(self, user_id: str, query: str) -> list[Chunk]:
        chunks_with_entities = self._get_chunks_with_entities(Document(query))
        entities = chunks_with_entities.get("entities", [])
        print(entities)
        # 边界：没有提取任何实体，直接返回空列表
        if not entities:
            return []

        # 拼装当前用户命名空间的实体ID
        entity_ids = [f"{user_id}_{ent['name']}" for ent in entities]

        with self.neo4j_client.session() as session:
            cypher = """
                MATCH (seed:Entity)
                WHERE seed.entity_id IN $entity_ids AND seed.user_id=$user_id
                WITH seed
                MATCH (e:Entity)
                WHERE e.user_id=$user_id AND (e = seed OR (seed)-[:RELATES_TO]-(e))
                WITH DISTINCT seed, e, CASE WHEN e=seed THEN 'direct' ELSE 'related' END AS match_type
                MATCH (c:Chunk)-[:MENTIONS {user_id:$user_id}]->(e)
                RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, c.page AS page, c.order AS order, match_type
                """
            result = session.run(
                cypher,
                entity_ids=entity_ids,
                user_id=user_id
            )
            records = list(result)

        # 组装检索结果
        chunk_list = []
        for record in records:
            if record["match_type"] == "direct":
                score = 0.85
            else:
                score = 0.60
            chunk_info = Chunk(chunk_id=record["chunk_id"], type="neo4j_chunk", text=record["text"], score=score)
            chunk_list.append(chunk_info)
        return chunk_list

    def _rearrange_chunks(self, qdrant_chunks: list[Chunk], neo4j_chunks: list[Chunk]) -> list[Chunk]:
        cache = {}
        all_chunks = qdrant_chunks + neo4j_chunks
        for chunk in all_chunks:
            cid = chunk.chunk_id
            # 已存在则对比分数，只保留更高分
            if cid not in cache or chunk.score > cache[cid].score:
                cache[cid] = chunk
        # 排序
        res = sorted(cache.values(), key=lambda c: c.score, reverse=True)
        return res


if __name__ == '__main__':
    vector_store = RagService()
    chunks = vector_store.retriever_chunks("666_999", "人体循环系统由什么组成？体循环和肺循环作用是什么？")
    for chunk in chunks:
        print(chunk.chunk_id)
        print(chunk.type)
        print(chunk.text)
        print(chunk.score)
