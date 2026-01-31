"""Configuration management for InsightRepo backend."""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    embedding_model: str = "nomic-embed-text"
    
    # Hugging Face (Optional)
    hf_api_token: str | None = None
    
    # RAG Configuration
    max_chunks: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Storage Configuration
    repos_dir: Path = Path("./repos")
    vector_store_dir: Path = Path("./vector_stores")
    
    # CORS
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.repos_dir.mkdir(exist_ok=True)
settings.vector_store_dir.mkdir(exist_ok=True)
