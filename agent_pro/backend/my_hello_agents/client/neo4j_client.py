import os

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver


def get_neo4j_client() -> Driver:
    return GraphDatabase.driver(
        uri=os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        max_connection_lifetime=300,          # 连接复用上限,低于服务端空闲回收时间
        connection_acquisition_timeout=60,    # 获取连接最久等待
        connection_timeout=30,                # 建立连接最久等待
    )

neo4j_client = get_neo4j_client()


if __name__ == '__main__':
    load_dotenv()
    client = get_neo4j_client()
