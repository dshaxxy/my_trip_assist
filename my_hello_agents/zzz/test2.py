from operator import add
from typing import Annotated, TypedDict, List

from langgraph.constants import END, START
from langgraph.graph import StateGraph


class SimpleState(TypedDict):
    message: Annotated[List[str], add]
    cur_id: str

def node_1(state: SimpleState) -> SimpleState:
    pre_id = state["cur_id"]
    return {
        "message": ["node_1 执行完毕"],
        "cur_id": pre_id + "node_1"
    }

def node_2(state: SimpleState) -> SimpleState:
    pre_id = state["cur_id"]
    return {
        "message": ["node_2 执行完毕"],
        "cur_id": pre_id + "node_2"
    }

builder = StateGraph(SimpleState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph = builder.compile()

result = graph.invoke({
    "message": [],
    "cur_id": ""
})
print(result)

raw_mermaid = graph.get_graph().draw_mermaid()
print(raw_mermaid)
