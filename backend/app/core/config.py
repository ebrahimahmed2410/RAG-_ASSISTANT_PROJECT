"""
Application Configuration Module using Pydantic Settings
"""

import json
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    app_env: str = Field(default="development", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    
    # CORS Configuration
    cors_origins: Union[List[str], str] = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"],
        alias="CORS_ORIGINS"
    )

    # Vector Store & Embeddings Configuration
    chroma_persist_dir: str = Field(
        default="./data/vector_store",
        alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="cs_documents",
        alias="CHROMA_COLLECTION_NAME"
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME"
    )

    # Ollama LLM Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_HOST"
    )
    ollama_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_MODEL"
    )
    ollama_timeout_seconds: float = Field(
        default=60.0,
        alias="OLLAMA_TIMEOUT_SECONDS"
    )

    # RAG Retrieval Parameters
    top_k: int = Field(
        default=4,
        alias="TOP_K"
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton settings instance
settings = Settings()
