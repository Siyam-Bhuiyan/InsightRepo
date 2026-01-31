import { IndexingStatus } from '../services/api';
import { CheckCircle, Circle, Loader } from 'lucide-react';

interface IndexingProgressProps {
  status: IndexingStatus | null;
}

export default function IndexingProgress({ status }: IndexingProgressProps) {
  if (!status) return null;

  const progress = status.total_files > 0 ? (status.processed_files / status.total_files) * 100 : 0;
  const isProcessing = status.status === 'processing';
  const isCompleted = status.status === 'completed';

  const steps = [
    {
      id: 1,
      title: 'Fetching remote repository',
      description: 'Successfully cloned insight-repo-test.git from GitHub',
      status: 'completed',
      time: '0.4s',
    },
    {
      id: 2,
      title: 'Parsing Abstract Syntax Trees (AST)',
      description: `Analyzing ${status.processed_files} files in ${status.repo_id}/components...`,
      status: isProcessing ? 'running' : isCompleted ? 'completed' : 'pending',
      time: isProcessing ? 'Running' : '1.2s',
    },
    {
      id: 3,
      title: 'Generating Vector Embeddings',
      description: 'Creating semantic mappings for code blocks...',
      status: progress > 50 ? (isCompleted ? 'completed' : 'running') : 'pending',
      time: progress > 50 ? (isCompleted ? '2.1s' : 'Running') : '',
    },
  ];

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          {isProcessing && <span className="inline-block w-2 h-2 bg-blue-400 rounded-full animate-pulse" />}
          <h3 className="text-white font-semibold">INDEXING IN PROGRESS</h3>
        </div>
        <p className="text-slate-400 text-sm">
          ID: <span className="font-mono text-blue-400">{status.repo_id}</span>
        </p>
      </div>

      {/* Progress Steps */}
      <div className="space-y-4 mb-6">
        {steps.map((step) => (
          <div key={step.id} className="flex gap-4">
            <div className="flex-shrink-0 mt-1">
              {step.status === 'completed' ? (
                <CheckCircle size={20} className="text-green-400" />
              ) : step.status === 'running' ? (
                <Loader size={20} className="text-blue-400 animate-spin" />
              ) : (
                <Circle size={20} className="text-slate-600" />
              )}
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <h4 className={`font-medium ${
                  step.status === 'completed' ? 'text-green-400' :
                  step.status === 'running' ? 'text-blue-400' :
                  'text-slate-500'
                }`}>
                  {step.title}
                </h4>
                {step.time && (
                  <span className="text-xs text-slate-500">{step.time}</span>
                )}
              </div>
              <p className="text-sm text-slate-400">{step.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Progress Bar */}
      {status.total_files > 0 && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-slate-400">
            <span>{status.processed_files} / {status.total_files} files</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Log Output */}
      <div className="mt-6 bg-slate-950 border border-slate-700 rounded-lg p-4 font-mono text-xs">
        <div className="space-y-1 text-slate-400">
          <div>[14:22:01] <span className="text-green-400">[INFO]</span> Initializing worker threads...</div>
          <div>[14:22:01] <span className="text-blue-400">[INFO]</span> Tokenizing {status.repo_id}...</div>
          <div>[14:22:03] <span className="text-yellow-400">[PARSE]</span> Building tree for {status.repo_id}/{status.message.split(' ').pop() || 'files'}</div>
          <div>[14:22:05] <span className="text-cyan-400">[EMBED]</span> Populating knowledge store...</div>
        </div>
      </div>
    </div>
  );
}
