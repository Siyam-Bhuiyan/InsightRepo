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
from status_store import status_store

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

# Note: Status is now persisted to disk via status_store
# No longer using in-memory dictionary


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
        status_store.set(repo_id, IndexingStatus(
            repo_id=repo_id,
            status="processing",
            message="Indexing repository...",
            total_files=0,
            processed_files=0
        ))
        
        # Index repository
        result = rag_engine.index_repository(repo_path, repo_id)
        
        # Update status with results
        status_store.set(repo_id, IndexingStatus(
            repo_id=repo_id,
            status=result['status'],
            message=result['message'],
            total_files=result['total_files'],
            processed_files=result.get('processed_files', 0)
        ))
        
    except Exception as e:
        status_store.set(repo_id, IndexingStatus(
            repo_id=repo_id,
            status="failed",
            message=f"Error during indexing: {str(e)}",
            total_files=0,
            processed_files=0
        ))


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
        status_store.set(repo_id, status)
        
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
        status_store.set(repo_id, status)
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")


@app.get("/repository/{repo_id}/status", response_model=IndexingStatus)
async def get_indexing_status(repo_id: str):
    """Get indexing status for a repository."""
    status = status_store.get(repo_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    return status


@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query a repository with a natural language question."""
    # Check if repository is indexed
    status = status_store.get(request.repo_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Repository not found")
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
    all_statuses = status_store.list_all()
    return {
        "repositories": [
            {
                "repo_id": repo_id,
                "status": status.status,
                "total_files": status.total_files,
                "processed_files": status.processed_files
            }
            for repo_id, status in all_statuses.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
