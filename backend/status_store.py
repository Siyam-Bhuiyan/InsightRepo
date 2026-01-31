"""Simple persistence layer for indexing status."""
import json
from pathlib import Path
from typing import Dict, Optional
from models import IndexingStatus


class StatusStore:
    """File-based persistence for indexing status."""
    
    def __init__(self, storage_path: Path = Path("./status_store.json")):
        self.storage_path = storage_path
        self._cache: Dict[str, IndexingStatus] = {}
        self._load()
    
    def _load(self):
        """Load status from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for repo_id, status_dict in data.items():
                        self._cache[repo_id] = IndexingStatus(**status_dict)
            except Exception as e:
                print(f"Failed to load status store: {e}")
    
    def _save(self):
        """Save status to disk."""
        try:
            data = {
                repo_id: status.model_dump() 
                for repo_id, status in self._cache.items()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save status store: {e}")
    
    def set(self, repo_id: str, status: IndexingStatus):
        """Store status for a repository."""
        self._cache[repo_id] = status
        self._save()
    
    def get(self, repo_id: str) -> Optional[IndexingStatus]:
        """Get status for a repository."""
        return self._cache.get(repo_id)
    
    def exists(self, repo_id: str) -> bool:
        """Check if repository exists."""
        return repo_id in self._cache
    
    def list_all(self) -> Dict[str, IndexingStatus]:
        """Get all stored statuses."""
        return self._cache.copy()


# Global status store instance
status_store = StatusStore()
