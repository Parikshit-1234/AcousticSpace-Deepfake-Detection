import React from 'react';
import { Cpu, CheckCircle, AlertTriangle, ShieldAlert } from 'lucide-react';

interface SubModel {
  name: string;
  description: string;
  spoof_probability: number;
  status: string; // "SAFE", "WARNING", "DANGER"
}

interface ModelEnsembleCardProps {
  models: SubModel[];
}

export const ModelEnsembleCard: React.FC<ModelEnsembleCardProps> = ({
  models,
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'SAFE':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'WARNING':
        return <AlertTriangle className="w-4 h-4 text-amber-400" />;
      case 'DANGER':
        return <ShieldAlert className="w-4 h-4 text-red-400" />;
      default:
        return <Cpu className="w-4 h-4 text-slate-400" />;
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'SAFE':
        return 'badge-success';
      case 'WARNING':
        return 'badge-warning';
      case 'DANGER':
        return 'badge-danger';
      default:
        return 'bg-slate-900 border border-slate-800 text-slate-400';
    }
  };

  return (
    <div className="glass-panel p-6 flex flex-col gap-6">
      <div className="flex items-center justify-between border-b border-slate-800/40 pb-4">
        <div>
          <h3 className="title-font text-lg font-bold text-slate-100 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            Ensemble Consensus Matrix
          </h3>
          <p className="text-xs text-slate-400">Decision fusion matrix from fine-tuned Wav2Vec 2.0 / XLS-R & WavLM speech transformer architectures</p>
        </div>
        <div className="text-right">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Decision Weights</span>
          <span className="text-xs font-semibold text-indigo-400">Transformer Fusion</span>
        </div>
      </div>

      {/* Model Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map((model, idx) => {
          const probPercent = Math.round(model.spoof_probability * 100);
          
          return (
            <div
              key={idx}
              className="bg-slate-900/30 p-4 rounded-xl border border-slate-800/60 flex flex-col justify-between gap-3 hover:border-slate-700/60 transition-colors"
            >
              <div className="flex justify-between items-start gap-2">
                <div>
                  <h4 className="title-font text-sm font-semibold text-slate-200">{model.name}</h4>
                  <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">{model.description}</p>
                </div>
                <span className={`badge ${getStatusBadgeClass(model.status)} text-[9px] shrink-0`}>
                  {model.status}
                </span>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-400 flex items-center gap-1">
                    {getStatusIcon(model.status)}
                    Spoof Likelihood
                  </span>
                  <span className="mono-font text-slate-200">{probPercent}%</span>
                </div>
                {/* Progress bar */}
                <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      model.status === 'DANGER'
                        ? 'bg-red-500'
                        : model.status === 'WARNING'
                        ? 'bg-amber-500'
                        : 'bg-emerald-500'
                    }`}
                    style={{ width: `${probPercent}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary Note */}
      <div className="bg-indigo-950/20 p-3 rounded-lg border border-indigo-950/40 text-xs text-slate-400 leading-relaxed">
        <span className="font-semibold text-indigo-300">Fine-Tuned Transformer Fusion: </span>
        Uses fine-tuned Wav2Vec 2.0 / XLS-R and WavLM pre-trained speech representation models for state-of-the-art deepfake voice detection. All custom CNNs have been purged.
      </div>
    </div>
  );
};
