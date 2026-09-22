"""
Frontend API Client for communicating with the RAG Assistant Backend
"""

import os
from typing import Dict, Any, Optional
import httpx
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


class RAGApiClient:
    """Client wrapper for all RAG Assistant backend endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout_seconds: float = 65.0):
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.timeout = timeout_seconds

    def check_health(self) -> Dict[str, Any]:
        """
        Calls GET /health to verify backend service status.
        Returns dictionary with status and component details.
        """
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                if response.status_code == 200:
                    return response.json()
                return {
                    "status": "degraded",
                    "error": f"Backend returned HTTP {response.status_code}",
                    "components": {}
                }
        except httpx.ConnectError:
            return {
                "status": "unreachable",
                "error": f"Cannot connect to backend at {self.base_url}. Ensure FastAPI is running.",
                "components": {}
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "components": {}
            }

    def query(self, question: str) -> Dict[str, Any]:
        """
        Calls POST /query with user question.
        Returns parsed answer and source citations, or structured error details.
        """
        url = f"{self.base_url}/query"
        payload = {"question": question.strip()}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "answer": data.get("answer", ""),
                        "sources": data.get("sources", [])
                    }

                if response.status_code == 422:
                    error_data = response.json()
                    detail = error_data.get("detail", "Invalid question format.")
                    if isinstance(detail, list) and detail:
                        detail = detail[0].get("msg", "Validation error.")
                    return {
                        "success": False,
                        "error": f"Validation Error: {detail}",
                        "answer": "",
                        "sources": []
                    }

                if response.status_code == 503:
                    detail = response.json().get("detail", "Service unavailable.")
                    return {
                        "success": False,
                        "error": f"Service Unavailable: {detail}",
                        "answer": "",
                        "sources": []
                    }

                return {
                    "success": False,
                    "error": f"Backend Error (HTTP {response.status_code}): {response.text}",
                    "answer": "",
                    "sources": []
                }

        except httpx.ConnectError:
            return {
                "success": False,
                "error": f"Could not connect to FastAPI backend at {self.base_url}. Please check that the server is started ('uvicorn app.main:app --port 8000').",
                "answer": "",
                "sources": []
            }

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"Request timed out after {self.timeout}s. Generation or retrieval took too long.",
                "answer": "",
                "sources": []
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected client error: {str(e)}",
                "answer": "",
                "sources": []
            }


# Default client instance
api_client = RAGApiClient()
