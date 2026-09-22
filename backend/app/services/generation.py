"""
LLM Generation Service communicating with Ollama with strict grounded prompting
"""

from typing import List, Dict, Any, Optional
import httpx
from app.utils.logging_config import logger


GROUNDED_SYSTEM_PROMPT = """You are a precise, document-grounded academic assistant.

Your task is to answer the user's question using ONLY the factual information provided in the Context below.

CRITICAL RULES:
1. Base your answer strictly on the provided Context. Do NOT invent, assume, or extrapolate information.
2. If the provided Context does not contain sufficient facts to answer the question, state clearly and concisely:
   "I could not find this information in the provided documents."
3. Do NOT use unsupported external knowledge or prior training assumptions that conflict with or go beyond the text.
4. Always cite your sources explicitly in your response by referencing the document name and page number (e.g. [cs101_data_structures.pdf, Page 2]).
5. Structure your response clearly using bullet points or concise paragraphs where helpful."""


class OllamaGenerationService:
    """
    Manages communication with a local Ollama instance for grounded answer generation.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "llama3.2",
        timeout_seconds: float = 60.0
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks with document and page headers for prompt injection."""
        if not retrieved_chunks:
            return "No relevant context found."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            doc = chunk.get("document", "Unknown")
            page = chunk.get("page", "?")
            text = chunk.get("text", "").strip()
            context_blocks.append(
                f"--- [Source {i}: {doc} (Page {page})] ---\n{text}"
            )
        return "\n\n".join(context_blocks)

    def build_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Constructs the complete grounded RAG prompt."""
        context_str = self.format_context(retrieved_chunks)
        prompt = (
            f"{GROUNDED_SYSTEM_PROMPT}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt

    async def check_connection(self) -> Dict[str, Any]:
        """
        Checks if the Ollama service is reachable and queries available models.
        """
        url = f"{self.ollama_host}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name") for m in data.get("models", [])]
                    model_available = any(
                        self.model_name in m or m.startswith(self.model_name)
                        for m in models
                    )
                    return {
                        "reachable": True,
                        "model_available": model_available,
                        "available_models": models,
                        "configured_model": self.model_name
                    }
                return {
                    "reachable": False,
                    "error": f"Ollama returned HTTP status {response.status_code}"
                }
        except Exception as e:
            return {
                "reachable": False,
                "error": str(e)
            }

    async def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generates a grounded answer from Ollama using retrieved chunks.
        Gracefully handles service outages, empty context, and timeouts.
        """
        # 1. If no chunks were retrieved, do not hallucinate
        if not retrieved_chunks:
            return "I could not find this information in the provided documents."

        prompt = self.build_prompt(question, retrieved_chunks)
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for deterministic, factual extraction
                "top_p": 0.9,
            }
        }

        try:
            logger.info(f"Sending generation request to Ollama ({self.model_name}) at {url}...")
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("response", "").strip()
                    if answer:
                        return answer
                    return "I could not find this information in the provided documents."

                logger.error(
                    f"Ollama returned non-200 status code: {response.status_code}, body: {response.text}"
                )
                return self._fallback_grounded_summary(question, retrieved_chunks, status_code=response.status_code)

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            logger.warning(f"Ollama connection error at {self.ollama_host}: {str(e)}")
            return self._fallback_grounded_summary(
                question,
                retrieved_chunks,
                error_msg=f"Could not connect to Ollama at {self.ollama_host}. Please ensure Ollama is running ('ollama serve')."
            )

        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout_seconds} seconds.")
            return self._fallback_grounded_summary(
                question,
                retrieved_chunks,
                error_msg="Generation timed out waiting for the local LLM."
            )

        except Exception as e:
            logger.error(f"Unexpected generation error: {str(e)}", exc_info=True)
            return self._fallback_grounded_summary(
                question,
                retrieved_chunks,
                error_msg=f"Generation service error: {str(e)}"
            )

    def _fallback_grounded_summary(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        error_msg: Optional[str] = None,
        status_code: Optional[int] = None
    ) -> str:
        """
        Fallback generator that extracts verified sentences from the top retrieved chunks
        when the Ollama LLM process is offline or unreachable.
        """
        diagnostic = error_msg or (f"Ollama returned HTTP error {status_code}" if status_code else "Ollama offline")
        
        lines = [
            f"[Service Notice: {diagnostic}]",
            "",
            "Based directly on the retrieved document passages:",
            ""
        ]

        for i, chunk in enumerate(retrieved_chunks[:2], start=1):
            doc = chunk.get("document", "Document")
            page = chunk.get("page", 1)
            snippet = chunk.get("text", "").strip()
            first_sentence = snippet.split(". ")[0].strip()
            if not first_sentence.endswith("."):
                first_sentence += "."
            lines.append(f"- According to **{doc} (Page {page})**: {first_sentence}")

        return "\n".join(lines)
