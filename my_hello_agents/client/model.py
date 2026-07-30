import os
from langchain_openai import ChatOpenAI


def get_model():
    return ChatOpenAI(
            model=os.getenv('SUMMARIZER_MODEL_ID', 'qwen3.7-plus'),
            openai_api_key=os.getenv('SUMMARIZER_API_KEY'),
            openai_api_base=os.getenv('SUMMARIZER_BASE_URL', 'https://ws-06luduvpfh4renzg.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'),
            temperature=0
    )

model = get_model()
