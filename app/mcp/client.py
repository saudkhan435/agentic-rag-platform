import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv

from mcp import Client, StdioServerParameters

load_dotenv()

logger = logging.getLogger(__name__)



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
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set"
        )

    server = StdioServerParameters(
        command=MCP_COMMAND,
        args=[
            "run",
            str(MCP_SERVER)
        ],
        cwd=str(PROJECT_ROOT),
        env={
            "DATABASE_URL": database_url
        }
    )

    logger.info(
        "MCP search started | limit=%d",
        limit
    )

    start_time = time.perf_counter()

    try:
        async with Client(server) as client:
            logger.info(
                "MCP client connected"
            )

            result = await client.call_tool(
                "search_documents",
                {
                    "query": query,
                    "limit": limit
                }
            )

            if result.is_error:
                logger.error(
                    "MCP search_documents tool returned an error"
                )

                raise RuntimeError(
                    "MCP search_documents tool failed"
                )

            data = result.structured_content

            if isinstance(data, dict) and "result" in data:
                results = data["result"]

            elif isinstance(data, list):
                results = data

            else:
                logger.error(
                    "MCP tool returned no structured output"
                )

                raise RuntimeError(
                    "MCP tool returned no structured output"
                )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "MCP search completed | "
                "results=%d | duration=%.2fs",
                len(results),
                elapsed
            )

            return results

    except Exception:
        elapsed = time.perf_counter() - start_time

        logger.exception(
            "MCP search failed | duration=%.2fs",
            elapsed
        )

        raise


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