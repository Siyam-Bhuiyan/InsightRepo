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
