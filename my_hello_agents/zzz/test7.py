import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path

path = get_abs_path("rag/knowledge_base/888.pdf")

# 安全打开PDF
with fitz.open(path) as doc:
    text_parts = []
    for page in doc:
        # 基于文字坐标提取，缓解段落碎片化
        words = page.get_text("words")
        lines = []
        line_buf = []
        last_y = None
        for item in words:
            x0, y0, x1, y1, word = item[:5]
            if last_y is not None and abs(y0 - last_y) > 9:
                lines.append(" ".join(line_buf))
                line_buf = []
            line_buf.append(word)
            last_y = y0
        if line_buf:
            lines.append(" ".join(line_buf))
        text_parts.append("\n".join(lines))
    full_text = "\n\n".join(text_parts)

# 【可选】在这里调用你的 _post_process_pdf_text 清洗文本

# 适配中文论文分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "。", "！", "？", "；", "\n", "，", " ", ""],
    keep_separator=True
)

chunks = splitter.split_text(full_text)
print(f"切分为 {len(chunks)} 个块")
for chunk in chunks:
    print("-" * 60)
    print(chunk.strip())
