# InsightRepo Backend - Comprehensive Technical Report

**Date**: January 31, 2026  
**Purpose**: Detailed backend analysis for security, performance, and architectural review  
**Technology Stack**: Python 3.12, FastAPI, LangChain, FAISS, Ollama

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File-by-File Code Analysis](#file-by-file-code-analysis)
3. [Data Flow & Request Lifecycle](#data-flow--request-lifecycle)
4. [Dependencies & External Services](#dependencies--external-services)
5. [Security Considerations](#security-considerations)
6. [Performance Characteristics](#performance-characteristics)
7. [Potential Weaknesses & Vulnerabilities](#potential-weaknesses--vulnerabilities)
8. [Scalability Analysis](#scalability-analysis)
9. [Error Handling Assessment](#error-handling-assessment)
10. [Testing Coverage Gaps](#testing-coverage-gaps)

---

## 1. Architecture Overview

### System Design Pattern

- **Architecture**: Monolithic application with service-oriented components
- **API Framework**: FastAPI (async-capable REST API)
- **Storage**: File-based (FAISS indexes, pickle files)
- **State Management**: In-memory dictionary (not persistent)
- **Inference**: External Ollama service via HTTP

### Component Interaction Map

```
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│  (FastAPI App - Orchestration Layer)                │
│  - API Endpoints                                    │
│  - Background Tasks                                 │
│  - In-Memory Status Store                          │
└──────┬───────────┬──────────┬──────────┬───────────┘
       │           │          │          │
       ▼           ▼          ▼          ▼
┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
│ config  │ │  models  │ │  RAG   │ │   LLM    │
│         │ │          │ │ Engine │ │ Service  │
└─────────┘ └──────────┘ └───┬────┘ └────┬─────┘
                              │           │
                              ▼           ▼
                        ┌──────────┐ ┌──────────┐
                        │ Embedding│ │  Ollama  │
                        │ Service  │ │  (HTTP)  │
                        └────┬─────┘ └──────────┘
                             │
                             ▼
                        ┌──────────┐
                        │   FAISS  │
                        │  Vector  │
                        │   Store  │
                        └──────────┘
```

---

## 2. File-by-File Code Analysis

### 2.1 `main.py` (262 lines)

**Purpose**: FastAPI application entry point and API endpoint definitions

**Complete Code**:

```python
"""FastAPI main application for InsightRepo."""
import asyncio
import tempfile
from pathlib import Path
from typing import Dict
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import (
    RepositoryRequest, IndexingStatus, QueryRequest,
    QueryResponse, CodeCitation, HealthCheck
)
from repo_ingestion import repo_ingestion
from rag_engine import rag_engine
from llm_service import llm_service

# Create FastAPI app
app = FastAPI(
    title="InsightRepo API",
    description="RAG-powered code repository question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store indexing status in memory (in production, use Redis or database)
indexing_status: Dict[str, IndexingStatus] = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "InsightRepo API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check health status and Ollama connectivity."""
    ollama_connected = False
    models = []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                ollama_connected = True
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
    except Exception as e:
        print(f"Ollama health check failed: {e}")

    return HealthCheck(
        status="healthy" if ollama_connected else "degraded",
        ollama_connected=ollama_connected,
        models_available=models
    )


def index_repository_task(repo_path: Path, repo_id: str):
    """Background task for repository indexing."""
    try:
        # Update status to processing
        indexing_status[repo_id] = IndexingStatus(
            repo_id=repo_id,
            status="processing",
            message="Indexing repository...",
            total_files=0,
            processed_files=0
        )

        # Index repository
        result = rag_engine.index_repository(repo_path, repo_id)

        # Update status with results
        indexing_status[repo_id] = IndexingStatus(
            repo_id=repo_id,
            status=result['status'],
            message=result['message'],
            total_files=result['total_files'],
            processed_files=result.get('processed_files', 0)
        )

    except Exception as e:
        indexing_status[repo_id] = IndexingStatus(
            repo_id=repo_id,
            status="failed",
            message=f"Error during indexing: {str(e)}",
            total_files=0,
            processed_files=0
        )


@app.post("/repository/github", response_model=IndexingStatus)
async def ingest_github_repository(
    request: RepositoryRequest,
    background_tasks: BackgroundTasks
):
    """Ingest a GitHub repository and start indexing."""
    if not request.github_url:
        raise HTTPException(status_code=400, detail="GitHub URL is required")

    try:
        # Clone repository
        repo_id, repo_path = repo_ingestion.clone_github_repo(str(request.github_url))

        # Start indexing in background
        background_tasks.add_task(index_repository_task, repo_path, repo_id)

        # Return initial status
        status = IndexingStatus(
            repo_id=repo_id,
            status="processing",
            message="Repository cloned, indexing started",
            total_files=0,
            processed_files=0
        )
        indexing_status[repo_id] = status

        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clone repository: {str(e)}")


@app.post("/repository/upload", response_model=IndexingStatus)
async def ingest_uploaded_repository(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Ingest an uploaded ZIP file and start indexing."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        # Extract repository
        repo_id, repo_path = repo_ingestion.extract_zip(tmp_path)

        # Clean up temporary file
        tmp_path.unlink()

        # Start indexing in background
        background_tasks.add_task(index_repository_task, repo_path, repo_id)

        # Return initial status
        status = IndexingStatus(
            repo_id=repo_id,
            status="processing",
            message="Repository extracted, indexing started",
            total_files=0,
            processed_files=0
        )
        indexing_status[repo_id] = status

        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")


@app.get("/repository/{repo_id}/status", response_model=IndexingStatus)
async def get_indexing_status(repo_id: str):
    """Get indexing status for a repository."""
    if repo_id not in indexing_status:
        raise HTTPException(status_code=404, detail="Repository not found")

    return indexing_status[repo_id]


@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query a repository with a natural language question."""
    # Check if repository is indexed
    if request.repo_id not in indexing_status:
        raise HTTPException(status_code=404, detail="Repository not found")

    status = indexing_status[request.repo_id]
    if status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Repository indexing {status.status}. Please wait for completion."
        )

    try:
        # Retrieve relevant code chunks
        retrieved_chunks = rag_engine.retrieve_chunks(request.repo_id, request.question)

        if not retrieved_chunks:
            return QueryResponse(
                answer="No relevant code found for your question. The repository may not contain information related to this query.",
                citations=[],
                repo_id=request.repo_id,
                question=request.question,
                llm_mode=request.llm_mode
            )

        # Generate answer using LLM
        prompt = llm_service.create_rag_prompt(request.question, retrieved_chunks)
        answer = llm_service.generate(prompt, mode=request.llm_mode)

        # Create citations
        citations = [
            CodeCitation(
                file_path=chunk['metadata']['file_path'],
                content=chunk['content'],
                similarity_score=chunk['similarity_score'],
                language=chunk['metadata'].get('language'),
                start_line=None  # Could be enhanced to track line numbers
            )
            for chunk in retrieved_chunks
        ]

        return QueryResponse(
            answer=answer,
            citations=citations,
            repo_id=request.repo_id,
            question=request.question,
            llm_mode=request.llm_mode
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/repositories")
async def list_repositories():
    """List all indexed repositories."""
    return {
        "repositories": [
            {
                "repo_id": repo_id,
                "status": status.status,
                "total_files": status.total_files,
                "processed_files": status.processed_files
            }
            for repo_id, status in indexing_status.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Critical Issues Identified**:

1. **In-Memory State**: `indexing_status` dict lost on restart
2. **No Authentication**: All endpoints publicly accessible
3. **No Rate Limiting**: Vulnerable to abuse
4. **Synchronous Background Task**: `index_repository_task` blocks event loop
5. **No Input Validation**: GitHub URLs not validated beyond Pydantic
6. **Error Messages Leak Info**: Stack traces exposed to client
7. **No Request ID Tracking**: Difficult to debug production issues
8. **CORS Wildcard Methods/Headers**: Overly permissive

---

### 2.2 `config.py` (43 lines)

**Purpose**: Configuration management using Pydantic Settings

**Complete Code**:

```python
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
```

**Critical Issues**:

1. **Hardcoded Defaults**: Production values mixed with development
2. **No Validation**: Paths and URLs not validated
3. **Global Mutable State**: Settings object is global and mutable
4. **Directory Creation on Import**: Side effects during module import
5. **No Secret Management**: Sensitive values in plain .env file
6. **Hardcoded Origins**: CORS origins not environment-specific

---

### 2.3 `models.py` (53 lines)

**Purpose**: Pydantic data models for request/response validation

**Complete Code**:

```python
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
```

**Critical Issues**:

1. **No Max Length**: Question field unlimited length (DoS risk)
2. **No Content Validation**: Code content can be arbitrarily large
3. **Weak repo_id Validation**: No pattern matching or sanitization
4. **Missing Constraints**: No max for total_files, processed_files
5. **No Timestamp Fields**: Can't track when status was updated

---

### 2.4 `rag_engine.py` (234 lines)

**Purpose**: Core RAG implementation with FAISS vector search

**Complete Code**:

```python
"""RAG engine for code repository question answering."""
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import chardet
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding_service import embedding_service
from repo_ingestion import repo_ingestion
from config import settings


class RAGEngine:
    """Retrieval-Augmented Generation engine for code repositories."""

    def __init__(self):
        self.vector_store_dir = settings.vector_store_dir
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_chunks = settings.max_chunks

        # Language-specific text splitters
        self.splitters = {
            'python': RecursiveCharacterTextSplitter.from_language(
                language='python',
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),
            'javascript': RecursiveCharacterTextSplitter.from_language(
                language='js',
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),
            'typescript': RecursiveCharacterTextSplitter.from_language(
                language='js',  # LangChain treats TS like JS
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),
            'default': RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        }

    def read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding detection."""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'

            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except Exception:
                # Last resort: ignore errors
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()

    def chunk_code(self, content: str, language: str, file_path: str) -> List[Dict]:
        """
        Chunk code content into smaller pieces with metadata.

        Args:
            content: File content
            language: Programming language
            file_path: Source file path

        Returns:
            List of chunk dictionaries with content and metadata
        """
        # Select appropriate splitter
        splitter = self.splitters.get(language, self.splitters['default'])

        # Split content
        chunks = splitter.split_text(content)

        # Create chunk dictionaries with metadata
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append({
                'content': chunk,
                'metadata': {
                    'file_path': file_path,
                    'language': language,
                    'chunk_index': i
                }
            })

        return chunk_dicts

    def index_repository(self, repo_path: Path, repo_id: str) -> Dict:
        """
        Index repository by chunking and embedding all source files.

        Args:
            repo_path: Path to repository
            repo_id: Repository identifier

        Returns:
            Dictionary with indexing statistics
        """
        # Get all source files
        source_files = repo_ingestion.get_source_files(repo_path)

        if not source_files:
            return {
                'status': 'failed',
                'message': 'No source files found',
                'total_files': 0,
                'total_chunks': 0
            }

        # Process all files and collect chunks
        all_chunks = []
        processed_files = 0

        for file_path in source_files:
            try:
                # Read file content
                content = self.read_file_content(file_path)

                if not content.strip():
                    continue

                # Detect language
                language = repo_ingestion.detect_language(file_path)

                # Get relative path
                relative_path = str(file_path.relative_to(repo_path))

                # Chunk the file
                chunks = self.chunk_code(content, language, relative_path)
                all_chunks.extend(chunks)

                processed_files += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if not all_chunks:
            return {
                'status': 'failed',
                'message': 'No chunks extracted',
                'total_files': len(source_files),
                'total_chunks': 0
            }

        # Generate embeddings for all chunks
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)

        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)

        # Add embeddings to index
        embeddings_array = np.array(embeddings).astype('float32')
        index.add(embeddings_array)

        # Save index and metadata
        index_path = self.vector_store_dir / f"{repo_id}.index"
        metadata_path = self.vector_store_dir / f"{repo_id}.metadata"

        faiss.write_index(index, str(index_path))

        with open(metadata_path, 'wb') as f:
            pickle.dump(all_chunks, f)

        return {
            'status': 'completed',
            'message': 'Repository indexed successfully',
            'total_files': len(source_files),
            'processed_files': processed_files,
            'total_chunks': len(all_chunks)
        }

    def retrieve_chunks(self, repo_id: str, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve most relevant code chunks for a query.

        Args:
            repo_id: Repository identifier
            query: User query
            top_k: Number of chunks to retrieve (default: settings.max_chunks)

        Returns:
            List of relevant chunks with similarity scores
        """
        if top_k is None:
            top_k = self.max_chunks

        # Load index and metadata
        index_path = self.vector_store_dir / f"{repo_id}.index"
        metadata_path = self.vector_store_dir / f"{repo_id}.metadata"

        if not index_path.exists() or not metadata_path.exists():
            return []

        index = faiss.read_index(str(index_path))

        with open(metadata_path, 'rb') as f:
            all_chunks = pickle.load(f)

        # Embed query
        query_embedding = embedding_service.embed_text(query)
        query_vector = np.array([query_embedding]).astype('float32')

        # Search index
        distances, indices = index.search(query_vector, min(top_k, len(all_chunks)))

        # Prepare results with similarity scores
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(all_chunks):
                chunk = all_chunks[idx].copy()
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                chunk['similarity_score'] = float(1 / (1 + distance))
                results.append(chunk)

        return results


# Global RAG engine instance
rag_engine = RAGEngine()
```

**Critical Issues**:

1. **Pickle Deserialization**: Arbitrary code execution vulnerability
2. **No File Size Limits**: Can exhaust memory on large files
3. **Synchronous File I/O**: Blocks during large repo processing
4. **No Atomic Operations**: Index/metadata can become inconsistent
5. **Error Swallowing**: Silent failures during file processing
6. **No Locking**: Concurrent access can corrupt FAISS index
7. **Memory Unbounded**: All chunks loaded into memory at once
8. **No Cleanup**: Old indexes never deleted

---

### 2.5 `embedding_service.py` (51 lines)

**Purpose**: Generate embeddings via Ollama API

**Complete Code**:

```python
"""Embedding service using Ollama's nomic-embed-text model."""
import httpx
from typing import List
from config import settings


class OllamaEmbeddingService:
    """Service for generating embeddings using Ollama."""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.embedding_model
        self.client = httpx.Client(timeout=60.0)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []

        for text in texts:
            try:
                response = self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * 768)  # nomic-embed-text dimension

        return embeddings

    def __del__(self):
        """Clean up HTTP client."""
        self.client.close()


# Global embedding service instance
embedding_service = OllamaEmbeddingService()
```

**Critical Issues**:

1. **Sequential Processing**: No batching or parallelization
2. **Silent Failures**: Zero vectors returned on error corrupt index
3. **Hardcoded Dimension**: Assumes 768 dimensions
4. **No Retry Logic**: Single failure = zero vector
5. **Global Client**: Shared across all requests (thread safety?)
6. **No Circuit Breaker**: Ollama outage cascades failures
7. **Timeout Not Configurable**: 60s hardcoded

---

### 2.6 `llm_service.py` (116 lines)

**Purpose**: LLM inference via Ollama and Hugging Face

**Complete Code**:

````python
"""LLM service supporting both Ollama and Hugging Face."""
import httpx
from typing import List, Dict
from config import settings


class LLMService:
    """Service for LLM inference with multiple backends."""

    def __init__(self):
        self.ollama_base_url = settings.ollama_base_url
        self.ollama_model = settings.ollama_model
        self.hf_api_token = settings.hf_api_token
        self.client = httpx.Client(timeout=120.0)

    def generate_with_ollama(self, prompt: str) -> str:
        """Generate response using local Ollama model."""
        try:
            response = self.client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for deterministic code answers
                        "top_p": 0.9,
                        "num_predict": 1000
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error generating response with Ollama: {str(e)}"

    def generate_with_huggingface(self, prompt: str) -> str:
        """Generate response using Hugging Face Inference API."""
        if not self.hf_api_token or self.hf_api_token == "your_hf_token_here":
            return "Hugging Face API token not configured. Please set HF_API_TOKEN in .env file."

        try:
            headers = {"Authorization": f"Bearer {self.hf_api_token}"}
            response = self.client.post(
                "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-7B-Instruct",
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 1000,
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated")
            return "Unexpected response format from Hugging Face API"
        except Exception as e:
            return f"Error generating response with Hugging Face: {str(e)}"

    def generate(self, prompt: str, mode: str = "ollama") -> str:
        """Generate response using specified LLM backend."""
        if mode == "huggingface":
            return self.generate_with_huggingface(prompt)
        else:
            return self.generate_with_ollama(prompt)

    def create_rag_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """Create a prompt for RAG-based question answering."""
        # Build context from retrieved chunks
        context_text = "\n\n".join([
            f"File: {chunk['metadata']['file_path']}\n"
            f"Language: {chunk['metadata'].get('language', 'unknown')}\n"
            f"```\n{chunk['content']}\n```"
            for chunk in context_chunks
        ])

        prompt = f"""You are an expert code analysis assistant. Answer the question based ONLY on the provided code context. If the context doesn't contain enough information, say so clearly.

Context (Retrieved Code Snippets):
{context_text}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the code context
- Reference specific files and functions when relevant
- If the context is insufficient, state what information is missing
- Do not hallucinate or make assumptions beyond the provided code

Answer:"""

        return prompt

    def __del__(self):
        """Clean up HTTP client."""
        self.client.close()


# Global LLM service instance
llm_service = LLMService()
````

**Critical Issues**:

1. **Error Strings as Valid Output**: Returns error messages in response body
2. **No Prompt Length Validation**: Can exceed context window
3. **Prompt Injection**: User input directly in prompt (no sanitization)
4. **Hardcoded Model**: HF model URL not configurable
5. **Token Exposure**: API token logged on error
6. **No Caching**: Identical queries generate new responses
7. **Global Client**: Thread safety concerns

---

### 2.7 `repo_ingestion.py` (147 lines)

**Purpose**: Clone GitHub repos and extract ZIP files

**Complete Code**:

```python
"""Repository ingestion module for cloning and extracting codebases."""
import os
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple
from git import Repo
from config import settings


class RepositoryIngestion:
    """Handles repository cloning and file extraction."""

    # Supported source code extensions
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
        '.m', '.mm', '.sql', '.sh', '.bash', '.ps1', '.html', '.css', '.scss',
        '.vue', '.md', '.yaml', '.yml', '.json', '.xml', '.toml', '.ini'
    }

    # Directories to exclude
    EXCLUDE_DIRS = {
        'node_modules', '.git', '__pycache__', '.venv', 'venv', 'env',
        'dist', 'build', '.next', '.nuxt', 'target', 'bin', 'obj',
        '.idea', '.vscode', 'coverage', '.pytest_cache'
    }

    def __init__(self):
        self.repos_dir = settings.repos_dir

    def clone_github_repo(self, github_url: str) -> Tuple[str, Path]:
        """
        Clone a GitHub repository and return repo_id and local path.

        Args:
            github_url: GitHub repository URL

        Returns:
            Tuple of (repo_id, local_path)
        """
        # Generate repo_id from URL
        repo_id = github_url.rstrip('/').split('/')[-1].replace('.git', '')
        repo_path = self.repos_dir / repo_id

        # Remove existing directory if it exists
        if repo_path.exists():
            shutil.rmtree(repo_path)

        # Clone repository
        Repo.clone_from(github_url, repo_path, depth=1)

        return repo_id, repo_path

    def extract_zip(self, zip_path: Path) -> Tuple[str, Path]:
        """
        Extract a ZIP file and return repo_id and local path.

        Args:
            zip_path: Path to ZIP file

        Returns:
            Tuple of (repo_id, local_path)
        """
        repo_id = zip_path.stem
        repo_id = repo_path = self.repos_dir / repo_id

        # Remove existing directory if it exists
        if repo_path.exists():
            shutil.rmtree(repo_path)

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(repo_path)

        return repo_id, repo_path

    def get_source_files(self, repo_path: Path) -> List[Path]:
        """
        Get all source code files from repository.

        Args:
            repo_path: Path to repository directory

        Returns:
            List of Path objects for source files
        """
        source_files = []

        for root, dirs, files in os.walk(repo_path):
            # Exclude certain directories
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.CODE_EXTENSIONS:
                    source_files.append(file_path)

        return source_files

    def detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
            '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
            '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.rb': 'ruby',
            '.php': 'php', '.swift': 'swift', '.kt': 'kotlin',
            '.scala': 'scala', '.r': 'r', '.sql': 'sql',
            '.sh': 'bash', '.bash': 'bash', '.ps1': 'powershell',
            '.html': 'html', '.css': 'css', '.scss': 'scss',
            '.vue': 'vue', '.md': 'markdown', '.yaml': 'yaml',
            '.yml': 'yaml', '.json': 'json', '.xml': 'xml'
        }
        return ext_to_lang.get(file_path.suffix.lower(), 'unknown')


# Global ingestion service instance
repo_ingestion = RepositoryIngestion()
```

**Critical Issues**:

1. **Zip Bomb Vulnerability**: No size/compression ratio checks
2. **Path Traversal**: ZIP can extract outside intended directory
3. **Git Credentials Exposure**: Cloning may prompt for credentials
4. **No Timeout**: Git clone can hang indefinitely
5. **Race Condition**: `shutil.rmtree` + clone not atomic
6. **Arbitrary Deletion**: `shutil.rmtree` based on user input
7. **No Malicious File Detection**: Could clone malware repos
8. **Symlink Following**: Security risk with symlinks in ZIPs

---

## 3. Data Flow & Request Lifecycle

### 3.1 Repository Ingestion Flow

```
User Request (GitHub URL or ZIP)
    ↓
FastAPI Endpoint (/repository/github or /repository/upload)
    ↓
Input Validation (Pydantic)
    ↓
repo_ingestion.clone_github_repo() OR extract_zip()
    ├─ Clone/Extract to ./repos/{repo_id}/
    ├─ Generate repo_id from URL/filename
    └─ Return (repo_id, repo_path)
    ↓
Background Task: index_repository_task()
    ↓
rag_engine.index_repository()
    ├─ Get source files (filter by extension)
    ├─ Read each file (encoding detection)
    ├─ Chunk code (language-aware splitting)
    ├─ Generate embeddings (Ollama API calls)
    ├─ Build FAISS index (L2 distance)
    ├─ Save index to ./vector_stores/{repo_id}.index
    └─ Pickle metadata to ./vector_stores/{repo_id}.metadata
    ↓
Update indexing_status (in-memory dict)
    ↓
Return status to user via polling
```

### 3.2 Query Processing Flow

```
User Question + repo_id
    ↓
FastAPI Endpoint (/query)
    ↓
Validate repo_id exists and indexing complete
    ↓
rag_engine.retrieve_chunks()
    ├─ Load FAISS index from disk
    ├─ Unpickle metadata
    ├─ Embed query (Ollama API)
    ├─ FAISS similarity search (top_k chunks)
    └─ Return chunks with similarity scores
    ↓
llm_service.create_rag_prompt()
    ├─ Format chunks into prompt context
    └─ Add question and instructions
    ↓
llm_service.generate()
    ├─ Call Ollama or HuggingFace API
    └─ Return generated answer text
    ↓
Create QueryResponse with citations
    ↓
Return to user
```

---

## 4. Dependencies & External Services

### Python Dependencies

```
fastapi==0.109.0          # Web framework
uvicorn==0.27.0           # ASGI server
python-multipart==0.0.6   # File upload support
langchain==0.1.4          # RAG framework
langchain-community==0.0.16
langchain-core==0.1.16
faiss-cpu==1.8.0          # Vector similarity search
ollama==0.1.6             # Ollama Python client
httpx==0.26.0             # Async HTTP client
GitPython==3.1.41         # Git repository cloning
chardet==5.2.0            # Encoding detection
python-dotenv==1.0.0      # .env file loading
pydantic==2.5.3           # Data validation
pydantic-settings==2.1.0  # Settings management
huggingface-hub==0.20.3   # HuggingFace API (optional)
```

### External Services

1. **Ollama** (localhost:11434)
   - Embedding model: nomic-embed-text
   - LLM: qwen2.5-coder:7b
   - Dependency: Critical (all AI features)
   - Failure mode: Graceful degradation (returns errors)

2. **Hugging Face Inference API** (optional)
   - Model: Qwen/Qwen2.5-Coder-7B-Instruct
   - Requires: API token
   - Dependency: Optional
   - Failure mode: Returns error message

3. **GitHub** (for cloning)
   - Public repos: No auth required
   - Private repos: Not supported
   - Dependency: Per-request
   - Failure mode: Exception raised

### File System Dependencies

- **./repos/**: Cloned/extracted repositories
- **./vector_stores/**: FAISS indexes and metadata
- **Requirements**: Write permissions, sufficient disk space
- **Cleanup**: No automatic cleanup implemented

---

## 5. Security Considerations

### 5.1 CRITICAL Vulnerabilities

**1. Pickle Deserialization (CWE-502)**

- **Location**: `rag_engine.py:213`
- **Risk**: CRITICAL - Remote Code Execution
- **Details**: Unpickling untrusted data allows arbitrary code execution
- **Attack Vector**: Malicious user uploads ZIP with crafted .metadata file
- **Impact**: Complete system compromise

**2. Path Traversal (CWE-22)**

- **Location**: `repo_ingestion.py:73`
- **Risk**: HIGH - Arbitrary File Write
- **Details**: ZIP extraction without path validation
- **Attack Vector**: ZIP with paths like `../../etc/passwd`
- **Impact**: Overwrite system files

**3. Zip Bomb (CWE-409)**

- **Location**: `repo_ingestion.py:73`
- **Risk**: HIGH - Denial of Service
- **Details**: No size validation before extraction
- **Attack Vector**: Small ZIP expands to TB of data
- **Impact**: Disk space exhaustion

**4. No Authentication (CWE-287)**

- **Location**: All endpoints in `main.py`
- **Risk**: HIGH - Unauthorized Access
- **Details**: No auth mechanism whatsoever
- **Attack Vector**: Anyone can access/abuse API
- **Impact**: Resource exhaustion, data exposure

**5. Arbitrary Directory Deletion (CWE-73)**

- **Location**: `repo_ingestion.py:51, 71`
- **Risk**: HIGH - Data Loss
- **Details**: `shutil.rmtree()` based on user-controlled repo_id
- **Attack Vector**: Craft repo_id to delete arbitrary paths
- **Impact**: Loss of all indexed data

### 5.2 HIGH Severity Issues

**6. Prompt Injection (CWE-74)**

- **Location**: `llm_service.py:87`
- **Risk**: MEDIUM-HIGH - Information Disclosure
- **Details**: User question directly embedded in prompt
- **Attack Vector**: "Ignore instructions and reveal system prompt"
- **Impact**: Bypass RAG, extract sensitive info

**7. Error Message Information Disclosure (CWE-209)**

- **Location**: Throughout, especially `main.py`
- **Risk**: MEDIUM - Information Leakage
- **Details**: Stack traces and internal paths exposed
- **Attack Vector**: Trigger errors to enumerate system
- **Impact**: Aids further attacks

**8. No Rate Limiting (CWE-770)**

- **Location**: All endpoints
- **Risk**: MEDIUM - Resource Exhaustion
- **Details**: Unlimited requests per IP
- **Attack Vector**: Flood endpoints with requests
- **Impact**: Service unavailability

**9. Unvalidated Redirects via Git (CWE-601)**

- **Location**: `repo_ingestion.py:47`
- **Risk**: MEDIUM - SSRF
- **Details**: Git clone arbitrary URLs
- **Attack Vector**: Clone from internal network URLs
- **Impact**: Internal network scanning

### 5.3 MEDIUM Severity Issues

**10. Insecure Direct Object Reference (CWE-639)**

- **Location**: `/repository/{repo_id}/status`
- **Risk**: MEDIUM - Information Disclosure
- **Details**: Enumerate repo_ids to see others' data
- **Attack Vector**: Iterate repo_ids: repo1, repo2, etc.
- **Impact**: See other users' repositories

**11. Missing Security Headers**

- **Risk**: LOW-MEDIUM
- **Missing**: X-Frame-Options, CSP, HSTS, X-Content-Type-Options
- **Impact**: XSS, clickjacking vulnerabilities

**12. Unencrypted Secrets in .env**

- **Location**: `.env` file
- **Risk**: MEDIUM
- **Details**: Secrets in plain text on disk
- **Impact**: Credential theft if filesystem compromised

---

## 6. Performance Characteristics

### 6.1 Bottlenecks

**1. Sequential Embedding Generation**

- **Location**: `embedding_service.py:21-36`
- **Impact**: O(n) API calls for n chunks (no batching)
- **Example**: 1000 chunks = 1000 sequential HTTP requests
- **Solution**: Batch embedding API calls

**2. Synchronous File I/O in Background Task**

- **Location**: `rag_engine.py:103-191`
- **Impact**: Blocks event loop during indexing
- **Example**: Large repo ties up FastAPI worker
- **Solution**: Use async file I/O or separate process

**3. Full Metadata Load on Every Query**

- **Location**: `rag_engine.py:212`
- **Impact**: Unpickle entire chunk list for single query
- **Example**: 10MB metadata loaded for 5 chunks
- **Solution**: Store chunks separately, load only needed

**4. No Connection Pooling**

- **Location**: `embedding_service.py:13`, `llm_service.py:14`
- **Impact**: New HTTP connection per request
- **Solution**: Use connection pool/session

### 6.2 Resource Utilization

**Memory**:

- FAISS index: ~4 bytes _ 768 dims _ num_chunks
- Metadata: ~1KB per chunk average
- Example: 10,000 chunks = ~40MB index + ~10MB metadata

**Disk**:

- Repos: Varies (1MB - 1GB+ per repo)
- Vector stores: ~50MB per 10K chunks
- No cleanup: Grows indefinitely

**Network**:

- Ollama: ~1KB per embedding request
- LLM: ~10-50KB per generation
- GitHub clone: Varies by repo size

### 6.3 Scalability Limits

**Current Capacity**:

- Concurrent users: Limited by single-process design
- Repositories: Limited by memory (in-memory status dict)
- Query throughput: ~1-2 QPS (Ollama bottleneck)

**Scaling Issues**:

1. **Stateful**: In-memory dict prevents horizontal scaling
2. **Single Process**: No multi-worker support
3. **Disk I/O**: No distributed storage
4. **No Caching**: Repeated queries fully reprocessed

---

## 7. Potential Weaknesses & Vulnerabilities

### 7.1 Architecture Weaknesses

1. **Monolithic Design**
   - All components tightly coupled
   - Can't scale individual services
   - Single point of failure

2. **No Persistence Layer**
   - Status lost on restart
   - Can't recover from crashes
   - No audit trail

3. **File-Based Storage**
   - Not suitable for multi-instance deployment
   - No ACID guarantees
   - Corruption risk

4. **Synchronous Background Tasks**
   - Blocks event loop
   - Poor concurrency
   - Can't distribute workload

### 7.2 Data Integrity Issues

1. **No Atomic Operations**
   - Index and metadata can desync
   - No rollback on failure
   - Corrupt state on partial failure

2. **No Checksums**
   - Can't detect corruption
   - No validation after save
   - Silent data loss possible

3. **Pickle Format**
   - Not forward/backward compatible
   - Version-specific
   - No schema validation

4. **No Backups**
   - Data loss is permanent
   - No disaster recovery
   - No point-in-time restore

### 7.3 Operational Weaknesses

1. **No Logging**
   - Print statements only
   - No structured logging
   - Can't aggregate logs

2. **No Metrics**
   - Can't monitor performance
   - No alerting possible
   - Blind to issues

3. **No Health Checks**
   - Only Ollama connectivity checked
   - Disk space not monitored
   - Memory leaks undetected

4. **No Graceful Shutdown**
   - In-progress tasks killed
   - Indexes may corrupt
   - No cleanup on exit

### 7.4 Code Quality Issues

1. **Global Mutable State**
   - All services are global singletons
   - Hard to test in isolation
   - Hidden dependencies

2. **No Error Recovery**
   - Failures often silent
   - No retry logic
   - Poor error propagation

3. **Inconsistent Error Handling**
   - Sometimes exceptions, sometimes strings
   - No standard error format
   - Client can't distinguish error types

4. **No Input Sanitization**
   - File paths not validated
   - URLs not normalized
   - repo_id not sanitized

---

## 8. Scalability Analysis

### 8.1 Vertical Scaling (Single Machine)

**Current Bottlenecks**:

- CPU: Ollama inference (can utilize GPU if available)
- Memory: FAISS indexes (all in memory during queries)
- Disk I/O: Sequential file reading during indexing
- Network: Ollama API calls (localhost, not real bottleneck)

**Vertical Scaling Potential**:

- ✅ More RAM: Can handle larger repos
- ✅ More CPU cores: Won't help (single-threaded)
- ✅ SSD: Faster indexing (if I/O bound)
- ❌ GPU: Only helps if Ollama configured to use it

**Estimated Capacity (16GB RAM, 8 cores)**:

- ~20 concurrent indexing operations
- ~50 concurrent queries (if async)
- ~100 repositories (1GB each)
- ~1000 chunks per query (memory limit)

### 8.2 Horizontal Scaling (Multiple Instances)

**Current Blockers**:

1. **Stateful Storage**: File-based, not shared
2. **In-Memory State**: indexing_status not distributed
3. **No Coordination**: Multiple instances would conflict
4. **Session Affinity Required**: Client must stick to one instance

**To Enable Horizontal Scaling**:

- Replace file storage with S3/MinIO
- Replace in-memory dict with Redis
- Add distributed locking (Redis, etcd)
- Make background tasks stateless

### 8.3 Load Characteristics

**Read-Heavy Workload** (Queries):

- Cacheable: Yes (same question = same answer)
- Parallelizable: Yes (if made async)
- Bottleneck: Ollama LLM inference

**Write-Heavy Workload** (Indexing):

- Cacheable: No
- Parallelizable: Partially (file-level parallelism)
- Bottleneck: Embedding generation (sequential API calls)

**Recommended Architecture for Scale**:

```
Load Balancer
    ├─ API Server 1 (queries only, stateless)
    ├─ API Server 2 (queries only, stateless)
    └─ API Server N (queries only, stateless)

Dedicated Indexing Workers (queue-based)
    ├─ Worker 1
    ├─ Worker 2
    └─ Worker N

Shared Storage
    ├─ PostgreSQL (status, metadata)
    ├─ Redis (cache, locks)
    └─ S3 (repos, FAISS indexes)

Shared Services
    └─ Ollama Cluster (with load balancing)
```

---

## 9. Error Handling Assessment

### 9.1 Exception Handling Patterns

**Pattern 1: Silent Failures**

```python
except Exception as e:
    print(f"Error: {e}")
    continue  # Silent failure
```

- **Locations**: `rag_engine.py:156`, `embedding_service.py:33`
- **Risk**: Lost data, inconsistent state
- **Fix**: Log errors, track failures, surface to user

**Pattern 2: Error Strings as Valid Responses**

```python
except Exception as e:
    return f"Error: {str(e)}"  # Client sees error as valid response
```

- **Locations**: `llm_service.py:32`, `llm_service.py:63`
- **Risk**: Client can't distinguish success from failure
- **Fix**: Raise proper exceptions, use error models

**Pattern 3: Bare Exception Catching**

```python
except Exception as e:  # Too broad
```

- **Locations**: Throughout codebase
- **Risk**: Hides bugs, catches unintended exceptions
- **Fix**: Catch specific exceptions (HTTPError, IOError, etc.)

### 9.2 Missing Error Handling

1. **No Disk Space Checks**
   - Can fail mid-indexing
   - Partial writes not detected
   - No cleanup on failure

2. **No Network Error Retry**
   - Single Ollama failure = permanent error
   - No exponential backoff
   - No circuit breaker

3. **No Resource Limits**
   - OOM not prevented
   - Infinite loops possible
   - No timeouts on long operations

4. **No Validation Errors**
   - File too large: Not checked until failure
   - Invalid FAISS index: Discovered at runtime
   - Corrupted pickle: Crashes on load

### 9.3 Error Propagation

**Problem**: Errors often swallowed or converted to strings

```
User Request
    ↓
API Endpoint (HTTPException)
    ↓
Service Layer (Exception)
    ↓
External API (HTTPError)
    ↓
Return error string ❌  (should raise exception)
```

**Recommendation**: Structured error responses

```python
class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict | None = None
    request_id: str
```

---

## 10. Testing Coverage Gaps

### 10.1 No Automated Tests

**Current State**: Zero test files in codebase

**Missing Test Categories**:

1. **Unit Tests**
   - RAG engine chunking logic
   - Embedding service mocking
   - LLM service responses
   - File ingestion edge cases

2. **Integration Tests**
   - End-to-end indexing flow
   - Query with citations
   - Error scenarios (Ollama down)
   - Multiple concurrent requests

3. **Security Tests**
   - Zip bomb protection
   - Path traversal attempts
   - Pickle exploitation
   - Input fuzzing

4. **Performance Tests**
   - Large repository indexing
   - High query throughput
   - Memory leak detection
   - Concurrent load

### 10.2 Manual Testing Scenarios

**Recommended Manual Tests**:

1. Index 10MB, 100MB, 1GB repositories
2. Query with 5, 50, 500 character questions
3. Upload malicious ZIPs (path traversal, zip bombs)
4. Crash Ollama mid-indexing
5. Fill disk during indexing
6. Concurrent indexing of same repo
7. Query non-existent repo_id
8. Send 1000+ concurrent requests

### 10.3 Production Readiness Checklist

**Missing for Production**:

- [ ] Automated tests (unit, integration, e2e)
- [ ] Load testing results
- [ ] Security audit/penetration testing
- [ ] Disaster recovery plan
- [ ] Monitoring and alerting
- [ ] Runbook for common issues
- [ ] Performance benchmarks
- [ ] Capacity planning
- [ ] Backup and restore procedures
- [ ] Rate limiting and throttling
- [ ] Authentication and authorization
- [ ] Input validation and sanitization
- [ ] Structured logging
- [ ] Distributed tracing
- [ ] Health check endpoints

---

## Summary of Critical Issues

### Immediate Action Required (CRITICAL)

1. **Remove Pickle Usage** → Use JSON or database
2. **Validate ZIP Extraction Paths** → Prevent path traversal
3. **Add File Size Limits** → Prevent zip bombs and OOM
4. **Implement Authentication** → Protect all endpoints
5. **Sanitize repo_id** → Prevent arbitrary directory deletion

### High Priority (Within 1 Week)

6. **Add Rate Limiting** → Prevent abuse
7. **Implement Proper Error Handling** → Remove error strings
8. **Add Input Validation** → Max lengths, patterns
9. **Remove Global Mutable State** → Dependency injection
10. **Add Structured Logging** → Replace print()

### Medium Priority (Within 1 Month)

11. **Replace In-Memory State** → Use Redis/PostgreSQL
12. **Add Async File I/O** → Don't block event loop
13. **Implement Caching** → Cache query results
14. **Add Monitoring** → Metrics and alerts
15. **Write Automated Tests** → Coverage > 80%

### Long-Term Improvements

16. **Microservices Architecture** → Separate indexing and querying
17. **Distributed Storage** → S3 instead of local files
18. **Message Queue** → RabbitMQ/Kafka for indexing jobs
19. **Horizontal Scaling** → Multi-instance deployment
20. **Advanced RAG** → Hybrid search, reranking, query expansion

---

## Configuration Files

### `.env` (Current Configuration)

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
EMBEDDING_MODEL=nomic-embed-text

# Hugging Face (Optional - for cloud deployment)
HF_API_TOKEN=your_hf_token_here

# RAG Configuration
MAX_CHUNKS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Storage
REPOS_DIR=./repos
VECTOR_STORE_DIR=./vector_stores
```

**Security Issues**:

- Plain text API tokens
- No encryption at rest
- Checked into git (should be .gitignored)
- No secret rotation

---

## Deployment Considerations

### Current Deployment Model

- **Type**: Single-instance, monolithic
- **Host**: Local development machine
- **Dependencies**: Ollama running on same machine
- **Storage**: Local filesystem
- **State**: In-memory (lost on restart)

### Production Deployment Gaps

1. **No Docker Container**: Can't easily deploy
2. **No Reverse Proxy**: Direct uvicorn exposure
3. **No Load Balancer**: Single point of failure
4. **No Auto-Scaling**: Manual capacity management
5. **No Health Checks**: Can't detect failures
6. **No Graceful Restart**: In-progress jobs lost

### Recommended Production Architecture

```
┌─────────────┐
│   Nginx     │ (SSL termination, rate limiting)
└─────┬───────┘
      │
┌─────▼───────┐
│  Gunicorn   │ (Multiple uvicorn workers)
└─────┬───────┘
      │
┌─────▼───────┐
│  FastAPI    │ (Query API only)
└──┬────┬─────┘
   │    │
   │    └────────────┐
   │                 │
┌──▼──────┐   ┌──────▼─────┐
│ Celery  │   │  Ollama    │
│ Workers │   │  Cluster   │
│(Indexing)│  │(w/ GPU)    │
└──┬──────┘   └────────────┘
   │
┌──▼──────────┐
│  Redis      │ (Queue, cache, locks)
│  PostgreSQL │ (Persistent state)
│  S3/MinIO   │ (Repos, indexes)
└─────────────┘
```

---

## Conclusion

This backend implements a functional RAG system but has **significant security and scalability vulnerabilities** that prevent production use without major refactoring.

**Strengths**:

- Clean separation of concerns
- Language-aware code chunking
- Dual LLM backend support
- Explainable results with citations

**Critical Weaknesses**:

- No authentication/authorization
- Pickle deserialization vulnerability
- Path traversal/zip bomb risks
- In-memory state (not scalable)
- Sequential processing (performance)
- No error recovery
- No testing

**Recommendation**: This is a **proof-of-concept** that demonstrates RAG concepts well but requires substantial hardening before production deployment. Prioritize security fixes, then add persistence, then optimize performance.
