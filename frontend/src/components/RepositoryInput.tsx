import { useState } from 'react';
import { Upload, Github, Loader } from 'lucide-react';
import { apiService, IndexingStatus } from '../services/api';

interface RepositoryInputProps {
  onRepositoryIndexed: (status: IndexingStatus) => void;
}

export default function RepositoryInput({ onRepositoryIndexed }: RepositoryInputProps) {
  const [inputType, setInputType] = useState<'github' | 'upload'>('github');
  const [githubUrl, setGithubUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGithubSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const status = await apiService.ingestGithubRepo(githubUrl);
      onRepositoryIndexed(status);
      
      // Poll for status updates
      pollIndexingStatus(status.repo_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to clone repository');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);
    setLoading(true);

    try {
      const status = await apiService.uploadRepository(file);
      onRepositoryIndexed(status);
      
      // Poll for status updates
      pollIndexingStatus(status.repo_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload repository');
    } finally {
      setLoading(false);
    }
  };

  const pollIndexingStatus = (repoId: string) => {
    const interval = setInterval(async () => {
      try {
        const status = await apiService.getIndexingStatus(repoId);
        onRepositoryIndexed(status);
        
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        clearInterval(interval);
      }
    }, 2000);
  };

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 shadow-xl">
      <div className="flex items-center gap-2 mb-4">
        <Github size={24} className="text-blue-400" />
        <h2 className="text-xl font-bold text-white">Import from GitHub</h2>
      </div>
      
      {/* Tab Selector */}
      <div className="flex gap-2 mb-6 bg-slate-900/50 p-1 rounded-lg">
        <button
          onClick={() => setInputType('github')}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition font-medium ${
            inputType === 'github'
              ? 'bg-blue-500 text-white shadow-lg'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <Github size={18} />
          <span className="text-sm">GitHub URL</span>
        </button>
        <button
          onClick={() => setInputType('upload')}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition font-medium ${
            inputType === 'upload'
              ? 'bg-blue-500 text-white shadow-lg'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <Upload size={18} />
          <span className="text-sm">Upload Source Archive</span>
        </button>
      </div>

      {/* GitHub Input */}
      {inputType === 'github' && (
        <form onSubmit={handleGithubSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              GitHub Repository URL
            </label>
            <input
              type="url"
              value={githubUrl}
              onChange={(e) => setGithubUrl(e.target.value)}
              placeholder="https://github.com/username/repository"
              className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-primary hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Cloning & Indexing...' : 'Clone Repository'}
          </button>
        </form>
      )}

      {/* Upload Input */}
      {inputType === 'upload' && (
        <div className="space-y-4">
          <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-blue-500 transition cursor-pointer">
            <Upload size={48} className="mx-auto text-slate-500 mb-3" />
            <p className="text-white font-medium mb-1">Drag and drop your .zip or .tar file here</p>
            <p className="text-sm text-slate-400 mb-4">Max size: 550MB</p>
            <input
              type="file"
              accept=".zip"
              onChange={handleFileUpload}
              disabled={loading}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="inline-flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white font-medium py-2 px-6 rounded-lg cursor-pointer transition"
            >
              <Upload size={18} />
              Select Local File
            </label>
          </div>
          <p className="text-xs text-slate-500 text-center">
            Upload a ZIP file containing your source code repository
          </p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
          {error}
        </div>
      )}
    </div>
  );
}
