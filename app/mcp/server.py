import sys
from pathlib import Path

from mcp.server import MCPServer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


mcp = MCPServer("Agentic RAG MCP Server")


@mcp.tool()
def search_documents(
    query: str,
    limit: int = 5
) -> list[dict]:
    """
    Search the ingested document collection for passages
    relevant to a natural-language query.
    """

    # Import here instead of during MCP server startup.
    from app.retrieval.service import retrieve_context

    results = retrieve_context(
        query=query,
        final_limit=limit
    )

    return [
        {
            "document": result["document"],
            "chunk_index": result["chunk_index"],
            "content": result["content"],
            "rerank_score": result["rerank_score"]
        }
        for result in results
    ]