import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    async with AsyncSqliteSaver.from_conn_string(':memory:') as saver:
        print(saver)

asyncio.run(main())
