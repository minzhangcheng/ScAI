import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from langgraph.graph import StateGraph

def node(state: dict) -> dict: return {"val": state.get("val", 0) + 1}
graph = StateGraph(dict)
graph.add_node("node", node)
graph.set_entry_point("node")

async def main():
    async with aiosqlite.connect(":memory:") as conn:
        saver = AsyncSqliteSaver(conn)
        compiled = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "1"}}
        res = await compiled.ainvoke({"val": 1}, config)
        print(res)

asyncio.run(main())
