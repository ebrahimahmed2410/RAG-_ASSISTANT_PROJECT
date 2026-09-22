"""
Live Verification of FastAPI Backend and ChromaDB Vector Store
Executes real retrieval against the persisted vector store on disk.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from fastapi.testclient import TestClient
from app.main import app


def test_live_api():
    print("[*] Starting FastAPI test client with real persisted vector store...")
    with TestClient(app) as client:
        # 1. Health check
        health_resp = client.get("/health")
        print(f"[+] GET /health -> Status {health_resp.status_code}")
        print(f"    Payload: {health_resp.json()}")
        assert health_resp.status_code == 200
        assert health_resp.json()["components"]["vector_store_initialized"] is True
        assert health_resp.json()["details"]["indexed_chunk_count"] > 0

        # 2. Live Query with real retrieval (and graceful fallback if Ollama offline)
        query_payload = {"question": "What is the difference between TCP and UDP?"}
        print(f"\n[*] Sending POST /query with question: '{query_payload['question']}'...")
        query_resp = client.post("/query", json=query_payload)
        print(f"[+] POST /query -> Status {query_resp.status_code}")
        data = query_resp.json()
        print(f"    Answer Preview: {data['answer'][:200]}...")
        print(f"    Retrieved Sources Count: {len(data['sources'])}")
        for src in data['sources']:
            print(f"      - {src['document']} (Page {src['page']}) | Relevance: {src['relevance_score']:.1%}")
            print(f"        Snippet: {src['snippet'][:100]}...")

        assert query_resp.status_code == 200
        assert len(data["sources"]) > 0
        assert "cs104_computer_networks.pdf" in [s["document"] for s in data["sources"]]

        # 3. Validation rejection test (Empty query -> 422)
        print("\n[*] Sending invalid empty query (expecting 422)...")
        invalid_resp = client.post("/query", json={"question": "  "})
        print(f"[+] POST /query (empty) -> Status {invalid_resp.status_code}")
        assert invalid_resp.status_code == 422

        print("\n[OK] All live API and real vector store verification tests PASSED!")


if __name__ == "__main__":
    test_live_api()
