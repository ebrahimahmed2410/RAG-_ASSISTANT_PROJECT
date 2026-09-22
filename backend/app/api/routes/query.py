"""
FastAPI Route Handlers for /health and /query Endpoints
"""

from typing import List
from fastapi import APIRouter, Request, HTTPException, status
from app.schemas.query import QueryRequest, QueryResponse, SourceItem, HealthResponse
from app.utils.logging_config import logger

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Component Status Check",
    tags=["Health"]
)
async def health_check(request: Request) -> HealthResponse:
    """
    Returns the operational status of the RAG assistant, including
    the vector store connection, indexed chunk count, and Ollama connection.
    """
    vector_store = getattr(request.app.state, "vector_store", None)
    llm_service = getattr(request.app.state, "llm_service", None)

    components = {
        "vector_store_initialized": False,
        "embedding_model_loaded": False,
        "ollama_connected": False,
    }
    details = {}

    if vector_store:
        components["vector_store_initialized"] = vector_store.is_ready
        components["embedding_model_loaded"] = vector_store.embedding_fn is not None
        details["indexed_chunk_count"] = vector_store.get_document_count()
        details["persist_dir"] = vector_store.persist_dir
        details["collection_name"] = vector_store.collection_name

    if llm_service:
        ollama_status = await llm_service.check_connection()
        components["ollama_connected"] = ollama_status.get("reachable", False)
        details["ollama_host"] = llm_service.ollama_host
        details["ollama_model"] = llm_service.model_name
        details["ollama_model_available"] = ollama_status.get("model_available", False)

    overall_status = "healthy" if components["vector_store_initialized"] else "degraded"

    return HealthResponse(
        status=overall_status,
        components=components,
        details=details
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit question and receive grounded answer with citations",
    tags=["RAG Query"]
)
async def query_documents(
    payload: QueryRequest,
    request: Request
) -> QueryResponse:
    """
    Processes a natural language query against indexed document chunks:
    1. Embeds question and executes ChromaDB vector similarity search.
    2. Builds a strictly grounded prompt with retrieved context and citations.
    3. Calls local Ollama LLM to synthesize the factual answer.
    4. Returns the answer alongside exact source documents and page citations.
    """
    vector_store = getattr(request.app.state, "vector_store", None)
    llm_service = getattr(request.app.state, "llm_service", None)
    settings = getattr(request.app.state, "settings", None)

    if not vector_store or not vector_store.is_ready:
        logger.error("Vector store service is not available or not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database is currently unavailable or indexing is incomplete."
        )

    if not llm_service:
        logger.error("LLM generation service is not available.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation service is not initialized."
        )

    top_k = settings.top_k if settings else 4

    try:
        # 1. Retrieve most relevant chunks
        logger.info(f"Processing query: '{payload.question}' (top_k={top_k})")
        retrieved_chunks = vector_store.search(query=payload.question, top_k=top_k)

        # 2. Generate grounded answer
        answer = await llm_service.generate_answer(
            question=payload.question,
            retrieved_chunks=retrieved_chunks
        )

        # 3. Format citations
        sources: List[SourceItem] = []
        for chunk in retrieved_chunks:
            sources.append(
                SourceItem(
                    document=chunk.get("document", "Unknown"),
                    page=chunk.get("page", 1),
                    relevance_score=chunk.get("relevance_score"),
                    snippet=chunk.get("text", "")[:250] + "..." if len(chunk.get("text", "")) > 250 else chunk.get("text", "")
                )
            )

        return QueryResponse(
            answer=answer,
            sources=sources
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing /query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the query. Please try again."
        )
