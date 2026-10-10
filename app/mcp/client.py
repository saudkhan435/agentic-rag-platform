import asyncio
import os
import shutil
from pathlib import Path

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MCP_COMMAND = shutil.which("mcp")

if not MCP_COMMAND:
    raise RuntimeError(
        "The 'mcp' command was not found in PATH"
    )

MCP_SERVER = (
    PROJECT_ROOT
    / "app"
    / "mcp"
    / "server.py"
)


async def search_documents_via_mcp(
    query: str,
    limit: int = 5
) -> list[dict]:

    server = StdioServerParameters(
    command=MCP_COMMAND,
    args=[
        "run",
        str(MCP_SERVER)
    ],
    cwd=str(PROJECT_ROOT),
    env={
        "DATABASE_URL": os.environ["DATABASE_URL"]
    }
)

    async with Client(server) as client:
        result = await client.call_tool(
            "search_documents",
            {
                "query": query,
                "limit": limit
            }
        )

        if result.is_error:
            raise RuntimeError(
                "MCP search_documents tool failed"
            )

        data = result.structured_content

        if isinstance(data, dict) and "result" in data:
            return data["result"]

        if isinstance(data, list):
            return data

        raise RuntimeError(
            "MCP tool returned no structured output"
        )


def search_documents_via_mcp_sync(
    query: str,
    limit: int = 5
) -> list[dict]:

    return asyncio.run(
        search_documents_via_mcp(
            query=query,
            limit=limit
        )
    )