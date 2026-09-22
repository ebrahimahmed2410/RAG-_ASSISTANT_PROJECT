"""
FastAPI Main Application Entry Point
Configures application lifespan, CORS, error handlers, and routes.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.utils.logging_config import logger
from app.services.retrieval import VectorStoreService
from app.services.generation import OllamaGenerationService
from app.api.routes.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager.
    Loads heavy resources (ChromaDB vector store, SentenceTransformers embedding model)
    once at application startup and shares them across requests via app.state.
    """
    logger.info("=== Starting RAG Assistant Backend Application ===")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Embedding Model: {settings.embedding_model_name}")
    logger.info(f"Ollama Target Host: {settings.ollama_host} (Model: {settings.ollama_model})")

    # Resolve vector store persist path relative to backend root
    persist_path = settings.chroma_persist_dir
    if not os.path.isabs(persist_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        persist_path = os.path.abspath(os.path.join(base_dir, persist_path))

    # 1. Initialize Vector Store Service
    vector_store = VectorStoreService(
        persist_dir=persist_path,
        collection_name=settings.chroma_collection_name,
        embedding_model_name=settings.embedding_model_name
    )
    vector_store.initialize()

    # 2. Initialize LLM Generation Service
    llm_service = OllamaGenerationService(
        ollama_host=settings.ollama_host,
        model_name=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds
    )

    # Attach instances to app.state
    app.state.vector_store = vector_store
    app.state.llm_service = llm_service
    app.state.settings = settings

    logger.info("Lifespan startup complete. Application is ready to accept requests.")

    yield

    logger.info("=== Shutting down RAG Assistant Backend Application ===")


def create_application() -> FastAPI:
    """Application factory for the FastAPI backend."""
    application = FastAPI(
        title="RAG-Powered Document Assistant API",
        description=(
            "Production-grade Retrieval-Augmented Generation (RAG) backend. "
            "Enables semantic question-answering over computer science educational documents "
            "with exact document and page-level source citations."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS Middleware configuration
    origins = settings.cors_origins
    if isinstance(origins, str):
        origins = [origins]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler to prevent leaking internal stack traces
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error on {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error occurred. Please try again later."}
        )

    # Root endpoint
    @application.get("/", tags=["Root"])
    async def root():
        return {
            "message": "RAG Document Assistant Backend API is running.",
            "documentation": "/docs",
            "health": "/health"
        }

    # Register routers
    application.include_router(query_router)

    return application


app = create_application()
