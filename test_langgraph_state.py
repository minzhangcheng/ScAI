from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import Dict, Any

def node(state: dict) -> dict:
    return {"val": state.get("val", 0) + 1}

graph = StateGraph(dict)
graph.add_node("node", node)
graph.set_entry_point("node")

conn = sqlite3.connect(":memory:", check_same_thread=False)
saver = SqliteSaver(conn)
compiled = graph.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "1"}}
compiled.invoke({"val": 1}, config)
snapshot = compiled.get_state(config)
print(snapshot.values)

