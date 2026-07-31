import asyncio
import json
from dotenv import load_dotenv
import os
load_dotenv()
from aio_pika import connect_robust, Message, IncomingMessage
from aio_pika.abc import AbstractQueue
from langchain_core.messages import messages_from_dict, BaseMessage

# 你的记忆提取、写入Qdrant工具函数
from my_hello_agents.client.memory_client import get_memory_manager

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://agent:123321@192.168.150.101:5672//agent")
QUEUE_NAME = "memory_extract_queue"
memory_manager = get_memory_manager()


async def process_memory_task(raw_task: dict):
    """核心：解析对话 + 提取长期记忆 + 入库向量库"""
    user_id = raw_task["user_id"]
    msg_dict_list = raw_task["chat_history"]
    # 字典还原 BaseMessage 列表
    chat_history: list[BaseMessage] = messages_from_dict(msg_dict_list)

    print(f"[消费者] 开始提取用户{user_id}长期记忆，对话长度：{len(chat_history)}")
    # 1. LLM提取结构化长期记忆（你之前的工作记忆提取prompt）
    # 2. 写入Qdrant向量库
    await asyncio.to_thread(
        memory_manager.add_episodic_memory,
        user_id=user_id,
        messages=chat_history
    )
    print(f"[消费者] 用户{user_id}长期记忆入库完成")


async def message_wrapper(message: IncomingMessage):
    async with message.process(requeue=True):
        try:
            body = json.loads(message.body.decode("utf-8"))
            await process_memory_task(body)
        except Exception as e:
            print(f"记忆提取异常，自动重入队列: {str(e)}")
            raise


async def consumer_main():
    connection = await connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue: AbstractQueue = await channel.declare_queue(QUEUE_NAME, durable=True)
        await queue.consume(message_wrapper)
        print(f"异步消费者启动，监听队列 {QUEUE_NAME}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(consumer_main())