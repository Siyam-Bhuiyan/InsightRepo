import { useState } from 'react';
import { Send } from 'lucide-react';
import { apiService, QueryResponse } from '../services/api';

interface QueryInterfaceProps {
  repoId: string;
  onQueryResult: (result: QueryResponse) => void;
}

export default function QueryInterface({ repoId, onQueryResult }: QueryInterfaceProps) {
  const [question, setQuestion] = useState('');
  const [llmMode, setLlmMode] = useState<'ollama' | 'huggingface'>('ollama');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setError(null);
    setLoading(true);

    try {
      const result = await apiService.queryRepository(repoId, question, llmMode);
      onQueryResult(result);
      setQuestion(''); // Clear input after successful query
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to process query');
    } finally {
      setLoading(false);
    }
  };

  const exampleQuestions = [
    'Where is authentication handled?',
    'How does data flow from the frontend to the database?',
    'Which files are responsible for API request validation?',
    'What are the main components of this application?',
    'How is error handling implemented?',
  ];

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 shadow-xl">
      <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
        <span className="text-lg">💬</span>
        Ask a Question
      </h3>

      {/* Query Form */}
      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-2">
            Ask about this repository...
          </label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g., How is authentication handled?"
            rows={4}
            className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none transition"
            required
          />
        </div>

        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm shadow-lg shadow-blue-500/30"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
              Generating...
            </>
          ) : (
            <>
              <Send size={16} />
              Ask →
            </>
          )}
        </button>
      </form>

      {/* Quick Actions */}
      <div className="mt-4 pt-4 border-t border-slate-700">
        <p className="text-xs font-medium text-slate-500 mb-2">Quick actions:</p>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setQuestion('Summarize this repo')}
            className="text-xs text-slate-400 hover:text-white bg-slate-700/30 hover:bg-slate-700/50 px-3 py-2 rounded transition text-left"
          >
            📊 Summarize
          </button>
          <button
            onClick={() => setQuestion('Explain file structure')}
            className="text-xs text-slate-400 hover:text-white bg-slate-700/30 hover:bg-slate-700/50 px-3 py-2 rounded transition text-left"
          >
            📁 Structure
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
          {error}
        </div>
      )}
    </div>
  );
}
