import asyncio
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

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

async def test():
    snapshot = await compiled.aget_state(config)
    print(snapshot.values)

asyncio.run(test())
