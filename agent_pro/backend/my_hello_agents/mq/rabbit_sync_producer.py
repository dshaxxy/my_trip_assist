import json
from typing import List, Dict, Any
import pika
from pika.exceptions import AMQPConnectionError, StreamLostError
from dotenv import load_dotenv
import time
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict, HumanMessage, AIMessage

load_dotenv()


class MemorySyncProducer:
    def __init__(
        self,
        queue_name: str = "memory_extract_queue",
        retry_times: int = 3,
        retry_delay: int = 1
    ):
        self.queue_name = queue_name
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self.connection = None
        self.channel = None

    def _connect(self):
        if self.connection and self.connection.is_open:
            return
        # 解析url提取账号密码host，或者直接拆分参数，这里推荐直接拆分
        credentials = pika.PlainCredentials("agent", "123321")
        params = pika.ConnectionParameters(
            host="192.168.150.101",
            port=5672,
            virtual_host="/agent",  # 直接写正确vhost
            credentials=credentials
        )
        for i in range(self.retry_times):
            try:
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.queue_name, durable=True)
                self.channel.basic_qos(prefetch_count=1)
                return
            except (AMQPConnectionError, StreamLostError) as e:
                print(f"MQ连接失败 {i+1}/{self.retry_times}: {e}")
                time.sleep(self.retry_delay)
        raise ConnectionError("RabbitMQ连接失败")

    def send_memory_task(self, user_id: str, chat_history: List[BaseMessage]):
        """
        发送对话历史，异步提取长期记忆
        :param user_id: 用户唯一标识
        :param chat_history: list[BaseMessage] 对话消息列表
        """
        self._connect()
        # BaseMessage 转可序列化字典
        msg_dict_list = [message_to_dict(msg) for msg in chat_history]
        task_payload = {
            "user_id": user_id,
            "chat_history": msg_dict_list
        }
        body = json.dumps(task_payload, ensure_ascii=False).encode("utf-8")
        props = pika.BasicProperties(delivery_mode=2)
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=body,
            properties=props
        )
        print(f"[生产者] 用户{user_id}对话记忆提取任务已投递")

    def close(self):
        if self.connection and self.connection.is_open:
            self.connection.close()

if __name__ == '__main__':
    history = [
        HumanMessage(content="我喜欢旅行，每年7月去海边"),
        AIMessage(content="好的，我记下你的喜好"),
        HumanMessage(content="简历重点是Java后端实习")
    ]

    # 投递任务，立即返回，不阻塞主线程
    producer = MemorySyncProducer()
    producer.send_memory_task(user_id="666", chat_history=history)
    producer.close()