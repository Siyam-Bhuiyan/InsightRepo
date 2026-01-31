import { useState, useEffect } from 'react';
import { Brain, Settings, BarChart, FolderGit2, User } from 'lucide-react';
import RepositoryInput from './components/RepositoryInput';
import IndexingProgress from './components/IndexingProgress';
import QueryInterface from './components/QueryInterface';
import AnswerDisplay from './components/AnswerDisplay';
import { IndexingStatus, QueryResponse, apiService } from './services/api';

type TabType = 'dashboard' | 'explorer' | 'analysis' | 'settings';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [indexingStatus, setIndexingStatus] = useState<IndexingStatus | null>(null);
  const [queryResults, setQueryResults] = useState<QueryResponse[]>([]);
  const [currentQuery, setCurrentQuery] = useState<QueryResponse | null>(null);
  const [health, setHealth] = useState<{ connected: boolean; models: string[] }>({
    connected: false,
    models: [],
  });

  useEffect(() => {
    // Check backend health on mount
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const healthData = await apiService.healthCheck();
      setHealth({
        connected: healthData.ollama_connected,
        models: healthData.models_available,
      });
    } catch (err) {
      console.error('Health check failed:', err);
    }
  };

  const handleRepositoryIndexed = (status: IndexingStatus) => {
    setIndexingStatus(status);
    if (status.status === 'completed') {
      setActiveTab('explorer');
    }
  };

  const handleQueryResult = (result: QueryResponse) => {
    setQueryResults([result, ...queryResults]);
    setCurrentQuery(result);
    setActiveTab('analysis');
  };

  const tabs = [
    { id: 'dashboard' as TabType, label: 'Dashboard', icon: FolderGit2 },
    { id: 'explorer' as TabType, label: 'Explorer', icon: Brain },
    { id: 'analysis' as TabType, label: 'Analysis', icon: BarChart },
    { id: 'settings' as TabType, label: 'Settings', icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-lg sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-blue-500 to-cyan-500 p-2 rounded-lg">
                <Brain size={28} className="text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">InsightRepo</h1>
                <p className="text-slate-400 text-xs">Repository Intelligence Platform</p>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav className="flex items-center gap-2">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                      activeTab === tab.id
                        ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                        : 'text-slate-400 hover:text-white hover:bg-slate-800'
                    }`}
                  >
                    <Icon size={18} />
                    <span className="font-medium">{tab.label}</span>
                  </button>
                );
              })}
            </nav>

            {/* User Menu */}
            <div className="flex items-center gap-4">
              {health.connected ? (
                <div className="flex items-center gap-2 text-green-400 text-sm">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span className="font-medium">Synced</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-red-400 text-sm">
                  <div className="w-2 h-2 bg-red-400 rounded-full" />
                  <span className="font-medium">Offline</span>
                </div>
              )}
              <button className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
                <Settings size={20} className="text-slate-400" />
              </button>
              <button className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
                <User size={20} className="text-slate-400" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1800px] mx-auto px-6 py-6">
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            <div className="text-center py-12">
              <h2 className="text-4xl font-bold text-white mb-3">Index Your Knowledge</h2>
              <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                Connect your codebase to InsightRepo. We'll parse your files and build a local
                RAG index for deep, repository-aware AI conversations.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-5xl mx-auto">
              <RepositoryInput onRepositoryIndexed={handleRepositoryIndexed} />
            </div>

            {indexingStatus && (
              <div className="max-w-5xl mx-auto">
                <IndexingProgress status={indexingStatus} />
              </div>
            )}

            {/* Features Grid */}
            <div className="grid grid-cols-3 gap-6 max-w-6xl mx-auto mt-12">
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
                <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
                  <FolderGit2 className="text-blue-400" size={24} />
                </div>
                <h3 className="text-white font-semibold mb-2">🌐 Local Indexing</h3>
                <p className="text-slate-400 text-sm">All code stays on your machine</p>
              </div>
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
                <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
                  <span className="text-2xl">🔒</span>
                </div>
                <h3 className="text-white font-semibold mb-2">🔐 Encrypted Storage</h3>
                <p className="text-slate-400 text-sm">Your keys are stored locally</p>
              </div>
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
                <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4">
                  <span className="text-2xl">⚡</span>
                </div>
                <h3 className="text-white font-semibold mb-2">⚡ GPU Accelerated</h3>
                <p className="text-slate-400 text-sm">Fast semantic search</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'explorer' && (
          <div className="grid grid-cols-12 gap-6 h-[calc(100vh-140px)]">
            {/* Left Sidebar - File Tree */}
            <div className="col-span-3 space-y-4">
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                  <FolderGit2 size={18} className="text-blue-400" />
                  EXPLORER
                </h3>
                {indexingStatus ? (
                  <div className="space-y-2">
                    <div className="text-sm text-slate-300 bg-slate-700/50 px-3 py-2 rounded">
                      📁 {indexingStatus.repo_id}
                    </div>
                    <div className="text-xs text-slate-400 px-3">
                      {indexingStatus.total_files} files indexed
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-400 text-sm">No repository loaded</p>
                )}
              </div>

              {indexingStatus?.status === 'completed' && (
                <QueryInterface
                  repoId={indexingStatus.repo_id}
                  onQueryResult={handleQueryResult}
                />
              )}
            </div>

            {/* Center - Code View */}
            <div className="col-span-6 bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
              <div className="bg-slate-800/80 px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-green-400 text-xs">●</span>
                  <span className="text-white font-medium text-sm">
                    {currentQuery ? 'Answer Context' : 'insight-repo-main'}
                  </span>
                </div>
              </div>
              <div className="p-6 overflow-auto h-full">
                {currentQuery ? (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-blue-400 font-semibold mb-2">Question:</h3>
                      <p className="text-white">{currentQuery.question}</p>
                    </div>
                    {currentQuery.citations.length > 0 && (
                      <div>
                        <h3 className="text-blue-400 font-semibold mb-3">Code Context:</h3>
                        <div className="space-y-4">
                          {currentQuery.citations.map((citation, idx) => (
                            <div key={idx} className="bg-slate-950 border border-slate-700 rounded-lg overflow-hidden">
                              <div className="bg-slate-800 px-4 py-2 border-b border-slate-700 flex items-center justify-between">
                                <span className="text-xs text-slate-300 font-mono">{citation.file_path}</span>
                                <span className="text-xs text-blue-400">{(citation.similarity_score * 100).toFixed(1)}% match</span>
                              </div>
                              <pre className="p-4 overflow-x-auto text-xs">
                                <code className="text-slate-300">{citation.content}</code>
                              </pre>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <Brain size={48} className="text-slate-600 mx-auto mb-4" />
                      <p className="text-slate-400">Ask a question to see code context</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Panel - System Status */}
            <div className="col-span-3 space-y-4">
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                <h3 className="text-white font-semibold mb-3">SYSTEM STATUS</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-slate-400">Vector DB Health</span>
                      <span className="text-xs text-green-400 font-medium">Healthy</span>
                    </div>
                    <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
                      <div className="h-full bg-green-400 w-full"></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-slate-400">Embeddings</span>
                      <span className="text-xs text-blue-400 font-medium">{indexingStatus?.processed_files || 0}</span>
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-slate-400">Chunks</span>
                      <span className="text-xs text-blue-400 font-medium">12.8k</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                <h3 className="text-white font-semibold mb-3">CONTEXT FOCUS</h3>
                <div className="space-y-2">
                  {currentQuery?.citations.slice(0, 3).map((citation, idx) => (
                    <div key={idx} className="text-xs bg-slate-700/50 px-3 py-2 rounded">
                      <div className="text-blue-400 font-mono truncate">{citation.file_path.split('/').pop()}</div>
                    </div>
                  )) || <p className="text-slate-400 text-xs">No context loaded</p>}
                </div>
              </div>

              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                <h3 className="text-white font-semibold mb-3">RECENT SEARCHES</h3>
                <div className="space-y-2">
                  {queryResults.slice(0, 3).map((result, idx) => (
                    <button
                      key={idx}
                      onClick={() => setCurrentQuery(result)}
                      className="text-xs text-slate-300 hover:text-white bg-slate-700/30 hover:bg-slate-700/50 px-3 py-2 rounded w-full text-left truncate transition"
                    >
                      {result.question}
                    </button>
                  )) || <p className="text-slate-400 text-xs">No searches yet</p>}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="space-y-6">
            <AnswerDisplay queryResults={queryResults} />
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="max-w-5xl mx-auto space-y-6">
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h2 className="text-2xl font-bold text-white mb-6">System Configuration</h2>
              <p className="text-slate-400 mb-6">
                Configure your RAG engine settings, switch between local and cloud inference, and
                manage your vector database health.
              </p>

              {/* Inference Source */}
              <div className="mb-8">
                <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                  <Brain size={20} className="text-blue-400" />
                  Inference Source
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <button className="bg-slate-700 hover:bg-slate-600 border-2 border-blue-500 text-white px-6 py-4 rounded-lg transition text-left">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-2xl">💻</span>
                      <span className="font-semibold">Local Mode (Ollama)</span>
                    </div>
                    <p className="text-xs text-slate-300">Running on http://localhost:11434 (v0.1.48)</p>
                  </button>
                  <button className="bg-slate-700/50 hover:bg-slate-700 text-slate-300 px-6 py-4 rounded-lg transition text-left">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-2xl">☁️</span>
                      <span className="font-semibold">Cloud Mode (Hugging Face)</span>
                    </div>
                    <p className="text-xs text-slate-400">Requires API token</p>
                  </button>
                </div>
              </div>

              {/* Model Selection */}
              <div className="mb-8">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-white font-semibold">Model Selection</h3>
                  <span className="text-blue-400 text-sm">Ollama Instance Connected</span>
                </div>
                <select className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:border-blue-500">
                  <option>Llama 3 (88 instruct)</option>
                  <option selected>qwen2.5-coder:7b (Code Optimized)</option>
                </select>
                <div className="mt-4 flex items-center gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Temperature</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      defaultValue="0.7"
                      className="w-32"
                    />
                    <span className="text-white text-sm ml-2">0.7</span>
                  </div>
                </div>
              </div>

              {/* Vector Database */}
              <div className="mb-8">
                <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                  <span className="text-xl">🗄️</span>
                  Vector Database
                </h3>
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <div className="grid grid-cols-3 gap-6 mb-4">
                    <div>
                      <div className="text-xs text-slate-400 mb-1">INDEX HEALTH</div>
                      <div className="text-lg font-semibold text-green-400">Healthy</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-400 mb-1">CHUNKS</div>
                      <div className="text-lg font-semibold text-white">12,842</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-400 mb-1">LAST RE-INDEX</div>
                      <div className="text-lg font-semibold text-white">4h ago</div>
                    </div>
                  </div>
                  <button className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg transition">
                    🔄 Re-index Repository
                  </button>
                </div>
              </div>

              {/* Performance */}
              <div>
                <h3 className="text-white font-semibold mb-4">Local Performance</h3>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <div className="text-xs text-slate-400 mb-1">VRAM USAGE</div>
                      <div className="text-sm text-white mb-2">4.2 GB / 8 GB</div>
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div className="h-full bg-blue-400" style={{ width: '52%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-400 mb-1">TOKEN GENERATION</div>
                      <div className="text-sm text-white mb-2">24 T/S</div>
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div className="h-full bg-green-400" style={{ width: '75%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
