import os

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver


def get_neo4j_client() -> Driver:
    return GraphDatabase.driver(uri=os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

neo4j_client = get_neo4j_client()


if __name__ == '__main__':
    load_dotenv()
    client = get_neo4j_client()
