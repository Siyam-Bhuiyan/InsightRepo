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
        repo_path = self.repos_dir / repo_id
        
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
