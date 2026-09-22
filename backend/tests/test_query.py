"""
Unit and Integration Tests for RAG Assistant Backend API
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.query import QueryRequest, QueryResponse
from app.services.retrieval import VectorStoreService
from app.services.generation import OllamaGenerationService


@pytest.fixture
def client():
    """Provides a FastAPI test client with mocked background services."""
    # Create mock VectorStoreService
    mock_vector_store = MagicMock(spec=VectorStoreService)
    mock_vector_store.is_ready = True
    mock_vector_store.collection_name = "cs_documents"
    mock_vector_store.persist_dir = "./data/vector_store"
    mock_vector_store.embedding_fn = MagicMock()
    mock_vector_store.get_document_count.return_value = 16
    mock_vector_store.search.return_value = [
        {
            "text": "Binary search has a time complexity of O(log n) on sorted arrays.",
            "document": "cs101_data_structures.pdf",
            "page": 4,
            "chunk_id": "cs101_data_structures.pdf_p4_c1",
            "distance": 0.15,
            "relevance_score": 0.925
        },
        {
            "text": "An array is a contiguous block of memory storing elements.",
            "document": "cs101_data_structures.pdf",
            "page": 1,
            "chunk_id": "cs101_data_structures.pdf_p1_c1",
            "distance": 0.40,
            "relevance_score": 0.800
        }
    ]

    # Create mock OllamaGenerationService
    mock_llm = MagicMock(spec=OllamaGenerationService)
    mock_llm.ollama_host = "http://localhost:11434"
    mock_llm.model_name = "llama3.2"
    mock_llm.check_connection = AsyncMock(return_value={
        "reachable": True,
        "model_available": True,
        "available_models": ["llama3.2:latest"]
    })
    mock_llm.generate_answer = AsyncMock(
        return_value="The time complexity of binary search is O(log n) according to cs101_data_structures.pdf (Page 4)."
    )

    with TestClient(app) as test_client:
        # Attach mocks to app.state inside context after lifespan executes
        app.state.vector_store = mock_vector_store
        app.state.llm_service = mock_llm
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns 200 and component statuses."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert data["components"]["vector_store_initialized"] is True
    assert data["components"]["embedding_model_loaded"] is True
    assert data["components"]["ollama_connected"] is True


def test_valid_query_request(client):
    """Test 1 from requirements: Valid /query request with mock generation."""
    payload = {"question": "What is the time complexity of binary search?"}
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "binary search" in data["answer"].lower()
    assert len(data["sources"]) == 2
    assert data["sources"][0]["document"] == "cs101_data_structures.pdf"
    assert data["sources"][0]["page"] == 4
    assert data["sources"][0]["relevance_score"] == 0.925


def test_invalid_empty_query_returns_422(client):
    """Test 2 from requirements: Invalid input returning 422."""
    payload = {"question": ""}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_invalid_whitespace_query_returns_422(client):
    """Test that query with only whitespace returns 422."""
    payload = {"question": "   \n\t  "}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_missing_question_field_returns_422(client):
    """Test that query payload missing the question field returns 422."""
    payload = {}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_too_short_question_returns_422(client):
    """Test that question shorter than minimum length returns 422."""
    payload = {"question": "ab"}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_prompt_formatting():
    """Unit test verifying grounded prompt construction rules."""
    service = OllamaGenerationService()
    chunks = [
        {"document": "test.pdf", "page": 2, "text": "CPU scheduling ensures fair allocation."}
    ]
    prompt = service.build_prompt("What is CPU scheduling?", chunks)

    assert "ONLY the factual information provided in the Context" in prompt
    assert "test.pdf (Page 2)" in prompt
    assert "CPU scheduling ensures fair allocation." in prompt
    assert "What is CPU scheduling?" in prompt


def test_fallback_summary_when_ollama_offline():
    """Unit test verifying graceful fallback summary when Ollama is offline."""
    service = OllamaGenerationService()
    chunks = [
        {"document": "test.pdf", "page": 1, "text": "Quicksort runs in O(n log n) average time. It partitions arrays."}
    ]
    summary = service._fallback_grounded_summary("What is Quicksort?", chunks, error_msg="Ollama offline")

    assert "Service Notice: Ollama offline" in summary
    assert "test.pdf (Page 1)" in summary
    assert "Quicksort runs in O(n log n) average time." in summary
