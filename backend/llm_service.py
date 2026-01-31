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
