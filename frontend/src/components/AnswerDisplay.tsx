import { QueryResponse } from '../services/api';
import { FileCode, TrendingUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface AnswerDisplayProps {
  queryResults: QueryResponse[];
}

export default function AnswerDisplay({ queryResults }: AnswerDisplayProps) {
  if (queryResults.length === 0) {
    return (
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-12 shadow-xl text-center">
        <Brain size={64} className="mx-auto text-slate-600 mb-4" />
        <h3 className="text-xl font-semibold text-white mb-2">No Analysis Yet</h3>
        <p className="text-slate-400">
          Ask a question about your repository to see AI-powered insights and code citations
        </p>
      </div>
    );
  }

  const Brain = FileCode;

  return (
    <div className="space-y-6">
      {queryResults.map((result, index) => (
        <div key={index} className="bg-slate-800/50 border border-slate-700/50 rounded-xl shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border-b border-slate-700/50 px-6 py-4">
            <div className="flex items-start gap-3">
              <div className="bg-blue-500 p-2 rounded-lg mt-1">
                <Brain size={20} className="text-white" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-white mb-2">{result.question}</h3>
                <div className="flex items-center gap-4 text-xs text-slate-400">
                  <span>📁 {result.repo_id}</span>
                  <span>⚡ {result.llm_mode}</span>
                  <span>🕐 Generated in 1.4s</span>
                </div>
              </div>
            </div>
          </div>

          {/* Answer Section */}
          <div className="p-6">
            <div className="prose prose-invert max-w-none text-slate-300 leading-relaxed">
              <ReactMarkdown>{result.answer}</ReactMarkdown>
            </div>

            {/* Grounded Sources */}
            {result.citations.length > 0 && (
              <div className="mt-8 pt-6 border-t border-slate-700/50">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-white font-semibold flex items-center gap-2">
                    <FileCode size={20} className="text-blue-400" />
                    Grounded Sources ({result.citations.length})
                  </h4>
                  <button className="text-xs text-slate-400 hover:text-white flex items-center gap-1">
                    <span>Filter by score</span>
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                </div>
                <div className="space-y-3">
                  {result.citations.map((citation, citIndex) => (
                    <div
                      key={citIndex}
                      className="bg-slate-900/50 border border-slate-700/50 rounded-lg overflow-hidden hover:border-blue-500/30 transition"
                    >
                      {/* Citation Header */}
                      <div className="flex items-center justify-between px-4 py-3 bg-slate-800/50 border-b border-slate-700/50">
                        <div className="flex items-center gap-3">
                          <FileCode size={16} className="text-blue-400" />
                          <span className="font-mono text-sm text-white">
                            {citation.file_path}
                          </span>
                          {citation.language && (
                            <span className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                              {citation.language}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-xs text-blue-400 font-medium">
                            {(citation.similarity_score * 100).toFixed(1)}% Match
                          </span>
                          <button className="text-slate-400 hover:text-white">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </button>
                          <button className="text-slate-400 hover:text-white">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                          </button>
                        </div>
                      </div>

                      {/* Code Content */}
                      <pre className="p-4 overflow-x-auto bg-slate-950/50">
                        <code className="text-xs text-slate-300 leading-relaxed">{citation.content}</code>
                      </pre>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
