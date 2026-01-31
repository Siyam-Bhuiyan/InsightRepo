"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal


class RepositoryRequest(BaseModel):
    """Request model for repository ingestion."""
    source_type: Literal["github", "upload"] = Field(..., description="Type of repository source")
    github_url: HttpUrl | None = Field(None, description="GitHub repository URL")
    # For file uploads, handled separately via FastAPI UploadFile


class IndexingStatus(BaseModel):
    """Status of repository indexing process."""
    repo_id: str
    status: Literal["processing", "completed", "failed"]
    message: str
    total_files: int = 0
    processed_files: int = 0


class QueryRequest(BaseModel):
    """Request model for Q&A queries."""
    repo_id: str = Field(..., description="Repository identifier")
    question: str = Field(..., min_length=5, description="Natural language question")
    llm_mode: Literal["ollama", "huggingface"] = Field("ollama", description="LLM inference mode")


class CodeCitation(BaseModel):
    """Citation to source code used in answer generation."""
    file_path: str
    content: str
    similarity_score: float
    language: str | None = None
    start_line: int | None = None


class QueryResponse(BaseModel):
    """Response model for Q&A queries."""
    answer: str
    citations: list[CodeCitation]
    repo_id: str
    question: str
    llm_mode: str


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    models_available: list[str]
