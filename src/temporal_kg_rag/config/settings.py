"""Configuration settings for the temporal knowledge graph RAG system."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")

    # OpenAI Configuration
    openai_api_key: str = Field(description="OpenAI API key")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    openai_embedding_dimensions: int = Field(
        default=1536, description="Embedding vector dimensions"
    )

    # LiteLLM Configuration
    litellm_api_base: str = Field(
        default="http://localhost:4000", description="LiteLLM proxy URL"
    )
    litellm_api_key: str = Field(default="sk-1234", description="LiteLLM API key")
    default_llm_model: str = Field(default="default", description="Default LLM model name")

    # Application Configuration
    chunk_size: int = Field(default=1000, description="Text chunk size in tokens")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks in tokens")
    max_retrieval_results: int = Field(default=10, description="Maximum retrieval results")
    temporal_window_days: int = Field(
        default=365, description="Default temporal window in days"
    )

    # Retrieval Configuration
    vector_similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity threshold for vector search"
    )
    hybrid_search_alpha: float = Field(
        default=0.5,
        description="Weight for vector vs graph search (0=all graph, 1=all vector)",
    )
    hybrid_search_k: int = Field(
        default=60, description="K parameter for Reciprocal Rank Fusion"
    )

    # Embedding Cache
    enable_embedding_cache: bool = Field(
        default=True, description="Enable embedding caching"
    )
    cache_dir: Path = Field(
        default=Path(".cache/embeddings"), description="Cache directory path"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format (json or text)")

    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, description="Streamlit server port")
    streamlit_server_address: str = Field(
        default="0.0.0.0", description="Streamlit server address"
    )

    # SpaCy Configuration
    spacy_model: str = Field(default="en_core_web_sm", description="SpaCy NER model")

    # Batch Processing
    embedding_batch_size: int = Field(
        default=100, description="Batch size for embedding generation"
    )
    max_concurrent_requests: int = Field(
        default=5, description="Max concurrent API requests"
    )

    # Graph Query Configuration
    max_traversal_depth: int = Field(
        default=2, description="Maximum depth for graph traversal"
    )
    query_timeout_seconds: int = Field(
        default=30, description="Timeout for graph queries in seconds"
    )

    def get_neo4j_config(self) -> dict:
        """Get Neo4j connection configuration as a dictionary."""
        return {
            "uri": self.neo4j_uri,
            "auth": (self.neo4j_user, self.neo4j_password),
            "database": self.neo4j_database,
        }

    def get_openai_config(self) -> dict:
        """Get OpenAI configuration as a dictionary."""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_embedding_model,
            "dimensions": self.openai_embedding_dimensions,
        }

    def get_litellm_config(self) -> dict:
        """Get LiteLLM configuration as a dictionary."""
        return {
            "api_base": self.litellm_api_base,
            "api_key": self.litellm_api_key,
            "model": self.default_llm_model,
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global _settings
    _settings = Settings()
    return _settings
