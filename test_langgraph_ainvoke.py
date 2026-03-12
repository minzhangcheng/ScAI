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

async def test():
    res = await compiled.ainvoke({"val": 1}, config)
    print(res)

asyncio.run(test())
