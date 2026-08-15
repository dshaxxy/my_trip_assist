import json
import os
import re
import tempfile
import uuid

from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel
from langchain_core.documents import Document
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

    # ============ PDF 加载 / 清洗 / 分片 (pdfplumber) ============

    @staticmethod
    def _detect_scanned(pdf) -> bool:
        """扫描版检测:首页提取不到有效文本即判定为扫描版。"""
        first_page = pdf.pages[0]
        text = (first_page.extract_text() or "").strip()
        return len(text) < 20 or "�" in text

    @staticmethod
    def _ocr_page_images(file_path: str, password: str = None, dpi: int = 200) -> list[tuple[int, str]]:
        """用 PyMuPDF 把 PDF 每页渲染为临时 PNG,返回 [(page_no, image_path)]。"""
        import fitz

        doc = fitz.open(file_path)
        if password:
            doc.authenticate(password)
        images = []
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)          # Windows 上必须释放句柄才能写入
                pix.save(tmp_path)
                images.append((page.number + 1, tmp_path))
        finally:
            doc.close()
        return images

    def _ocr_pdf_with_dashscope(self, file_path: str, password: str = None) -> list[tuple[int, str]]:
        """扫描版 PDF 逐页 OCR:渲染为图片后调 qwen-vl-ocr,返回 [(page_no, ocr_text)]。"""
        from dashscope import MultiModalConversation

        images = self._ocr_page_images(file_path, password)
        pages = []
        try:
            for page_no, img_path in images:
                messages = [{"role": "user", "content": [
                    {"image": f"file://{img_path}"},
                    {"text": "请识别这张图片中的所有文字,保留段落结构,输出纯文本。"},
                ]}]
                resp = MultiModalConversation.call(model="qwen-vl-ocr", messages=messages)
                if resp.status_code == 200:
                    content = resp.output.choices[0].message.content
                    # content 可能是 str 或 list[dict]
                    if isinstance(content, str):
                        text = content
                    else:
                        text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
                else:
                    text = ""
                pages.append((page_no, text))
        finally:
            for _, img_path in images:
                if os.path.exists(img_path):
                    os.remove(img_path)
        return pages

    @staticmethod
    def _clean_text(text: str, merge_lines: bool = True) -> str:
        """清洗 PDF 提取文本:去控制字符/页码/装饰线,处理软换行,压缩空白。

        merge_lines=True(文本):去掉所有折行换行,仅保留句子结束标点后的换行,
        使段落完整不截断。merge_lines=False(表格):保留行结构。
        """
        if not text:
            return ""
        # 去掉控制字符(PDF 常把 SOH/FF 等当段落/分页分隔符)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # 独立成行的页码(1-3 位数字),兼容页首/页尾及末尾无换行
        text = re.sub(r"(?:^|\n)\s*\d{1,3}\s*(?:\n|$)", "\n", text)
        # 分隔线 / 装饰线
        text = re.sub(r"\n[—\-_＝=]{3,}\n", "\n", text)
        if merge_lines:
            # 去掉所有折行换行,仅保留句子结束标点后的换行作为段落边界
            text = re.sub(r"(?<![。！？；;])[\r\n]+", "", text)
        # 压缩连续空白
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _rows_to_markdown(rows: list[list]) -> str:
        """表格数据转 Markdown 表格字符串。"""
        if not rows:
            return ""
        # 补全行内空单元格,保证列对齐
        width = max((len(r) for r in rows), default=0)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(str(c).replace("|", "\\|").strip() for c in rows[0]) + " |",
                 "| " + " | ".join(["---"] * width) + " |"]
        for row in rows[1:]:
            lines.append("| " + " | ".join(str(c).replace("|", "\\|").strip() for c in row) + " |")
        return "\n".join(lines)

    @classmethod
    def _table_to_documents(cls, table_md: str, page: int, chapter: str = "") -> list[Document]:
        """把一张 Markdown 表格按行切成多个 Document,带重复表头与行 overlap。"""
        lines = table_md.split("\n")
        header = lines[0]
        sep = lines[1]
        body = lines[2:]
        # 每块最多容纳的行数(按总行数自适应)
        block_rows = 12 if len(body) > 12 else max(6, len(body))
        docs = []
        for start in range(0, len(body), block_rows):
            block = body[start:start + block_rows]
            block_text = "\n".join([header, sep, *block])
            if start > 0:
                # 衔接上一块最后一行,保持行间上下文
                block_text = "\n".join([header, sep, body[start - 1], *block])
            docs.append(Document(
                page_content=block_text,
                metadata={
                    "type": "table",
                    "page": page,
                    "chapter": chapter,
                    "start_index": start,
                }
            ))
        return docs

    def _extract_doc_chunk(self, file_path: str, password: str = None) -> list[Document]:
        """
        加载 PDF 并分片,返回 Document 列表。
        - 文本版 PDF:pdfplumber 提取,表格转 Markdown 块,文本走清洗+语义切分
        - 扫描版 PDF:检测到首页无文本层后,自动走 qwen-vl-ocr 逐页识别
        - 用 outside_bbox 裁掉表格区域,避免表格内容在文本块中重复
        """
        import logging
        import pdfplumber

        # pdfminer 对缺 FontBBox 元数据的字体打 WARNING,是已知无害噪音,静默掉
        logging.getLogger("pdfminer").setLevel(logging.ERROR)

        path = get_abs_path(file_path)
        docs: list[Document] = []

        with pdfplumber.open(path, password=password) as pdf:
            # 扫描版 PDF:首页无文本层,走 OCR
            if self._detect_scanned(pdf):
                print(f"[OCR] 检测到扫描版 PDF: {path},调用 qwen-vl-ocr 逐页识别...")
                ocr_pages = self._ocr_pdf_with_dashscope(path, password)
                all_text_pages = []
                for page_no, text in ocr_pages:
                    cleaned = self._clean_text(text)
                    if cleaned:
                        all_text_pages.append((page_no, cleaned))
                docs.extend(self._chunk_text_pages(all_text_pages))
                return docs

            all_text_pages = []          # [(page_no, cleaned_text)]
            table_docs: list[Document] = []

            for page in pdf.pages:
                page_no = page.page_number

                # ---- 表格:仅当页面竖线足够多(密集格线)时才检测,排除页框/装饰线误判 ----
                v_edges = [e for e in page.edges if e["orientation"] == "v"]
                tables = []
                if len(v_edges) >= 5:
                    tables = page.find_tables()
                if tables:
                    for table in tables:
                        rows = table.extract()
                        # 过滤混入 bbox 的正文行:真实表格行含多个短单元格,正文行是单个超长单元格
                        rows = [r for r in rows if sum(1 for c in r if str(c or "").strip()) >= 2]
                        if len(rows) < 2:
                            continue
                        table_md = self._clean_text(self._rows_to_markdown(rows), merge_lines=False)
                        if table_md:
                            table_docs.extend(self._table_to_documents(table_md, page_no))

                # ---- 文本:裁掉所有表格区域后提取,避免与表格块重复 ----
                text_area = page
                for table in tables:
                    text_area = text_area.outside_bbox(table.bbox)
                page_text = (text_area.extract_text() or "").strip()
                cleaned = self._clean_text(page_text)
                if cleaned:
                    all_text_pages.append((page_no, cleaned))

        docs.extend(self._chunk_text_pages(all_text_pages))
        # 表格块排在文本块之前,便于后续按类型区分处理
        return table_docs + docs

    def _chunk_text_pages(self, all_text_pages: list[tuple[int, str]]) -> list[Document]:
        """跨页拼接为连续文本,统一按语义切分,保留 page 元数据。"""
        docs: list[Document] = []
        if not all_text_pages:
            return docs
        full_text = ""
        page_offsets = []            # [(page_no, start, end)]
        for page_no, text in all_text_pages:
            start = len(full_text)
            full_text += text + "\n\n"
            page_offsets.append((page_no, start, len(full_text)))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            add_start_index=True,
        )
        full_doc = Document(page_content=full_text)
        text_chunks = splitter.split_documents([full_doc])

        def page_of(start_idx):
            for page_no, s, e in page_offsets:
                if start_idx < e:
                    return page_no
            return page_offsets[-1][0]

        for chunk in text_chunks:
            chunk.metadata["type"] = "text"
            chunk.metadata["page"] = page_of(chunk.metadata["start_index"])
            docs.append(chunk)
        return docs

    def add_to_neo4j_qdrant(self, user_id: str, file_path: str, password: str = None):
        path = get_abs_path(file_path)
        doc_name = path.split("/")[-1]
        # 用 pdfplumber 完整链路提取文本 + 表格,已做清洗/分片/去重
        chunks = self._extract_doc_chunk(path, password)
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
        text_order = 0
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            # 所有块(文本 + 表格)都进 Qdrant 做向量检索
            self._add_to_qdrant(user_id, chunk, chunk_id)
            # 表格块是平铺行数据,只进向量库;硬塞实体图反而制造噪音
            if chunk.metadata.get("type") == "table":
                print(f"[表格块] 仅入 Qdrant, page={chunk.metadata.get('page')}")
                continue
            chunks_with_entities = self._get_chunks_with_entities(chunk)
            entitys = chunks_with_entities.get("entities", [])
            for entity in entitys:
                print(entity)
            relations = chunks_with_entities.get("relations", [])
            for rel in relations:
                print(rel)
            print("===" * 30)
            self._add_to_neo4j(user_id, chunk, chunk_id, doc_name, chunks_with_entities, order=text_order)
            text_order += 1



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

    def _add_to_neo4j(self, user_id: str, chunk: Document, chunk_id: str, doc_name: str, chunks_with_entities, order: int = 0):
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
                order=order,
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
    rag_service = RagService()
    chunks = rag_service._extract_doc_chunk("rag/knowledge_base/888.pdf")
    for chunk in chunks:

        print("-----------------")
        print(chunk)
