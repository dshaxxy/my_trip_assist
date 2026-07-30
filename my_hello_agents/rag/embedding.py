import os
import dashscope
from dashscope import TextEmbedding

class EmbeddingService:

    @staticmethod
    def encode(text: str) -> list[float]:
        response = TextEmbedding.call(
            model="text-embedding-v4",
            input=text,
            parameters={"dimension": 1536}
        )
        return response.output["embeddings"][0]["embedding"]

    @staticmethod
    def encode_batch(texts: list[str]) -> list[list[float]]:
        response = TextEmbedding.call(
            model="text-embedding-v4",
            input=texts,
            parameters={"dimension": 1536}
        )
        return [item["embedding"] for item in response.output["embeddings"]]



if __name__ == '__main__':
    print(EmbeddingService.encode("hello world"))
