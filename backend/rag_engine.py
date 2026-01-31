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
