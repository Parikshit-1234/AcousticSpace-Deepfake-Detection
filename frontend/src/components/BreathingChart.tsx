import React from 'react';
import { Wind, ShieldAlert, CheckCircle2 } from 'lucide-react';

interface BreathingEvent {
  start: number;
  end: number;
  duration: number;
  intensity: number;
}

interface BreathingMismatch {
  timestamp: number;
  reason: string;
  severity: number;
}

interface BreathingChartProps {
  events: BreathingEvent[];
  mismatches: BreathingMismatch[];
  coherenceScore: number;
  syllablesCount: number;
  duration: number;
  currentTime: number;
}

export const BreathingChart: React.FC<BreathingChartProps> = ({
  events,
  mismatches,
  coherenceScore,
  syllablesCount,
  duration,
  currentTime,
}) => {
  const coherencePercent = Math.round(coherenceScore * 100);

  const getCoherenceStatus = (score: number) => {
    if (score >= 0.85) return { text: 'Optimal Sync', class: 'text-emerald-400 border-emerald-950 bg-emerald-950/20' };
    if (score >= 0.6) return { text: 'Minor Cadence Drift', class: 'text-amber-400 border-amber-950 bg-amber-950/20' };
    return { text: 'Severe Mismatch Flag', class: 'text-red-400 border-red-950 bg-red-950/20' };
  };

  const status = getCoherenceStatus(coherenceScore);

  return (
    <div className="glass-panel p-4 flex flex-col gap-4">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="title-font text-md font-semibold text-slate-100 flex items-center gap-2">
            <Wind className="w-4.5 h-4.5 text-indigo-400" />
            Vocal Respiration & Cadence Alignment
          </h3>
          <p className="text-xs text-slate-400">Verifies in-breath breaks occur logically at syllable junctions</p>
        </div>
        <div className={`px-2.5 py-1 rounded-lg border text-xs font-semibold ${status.class}`}>
          {status.text} ({coherencePercent}%)
        </div>
      </div>

      {/* Cadence Stats */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-slate-900/40 p-2 rounded-lg border border-slate-800/40 flex justify-between">
          <span className="text-slate-400">Syllable Onsets Detected</span>
          <span className="font-semibold text-slate-200 mono-font">{syllablesCount}</span>
        </div>
        <div className="bg-slate-900/40 p-2 rounded-lg border border-slate-800/40 flex justify-between">
          <span className="text-slate-400">Breathing Pauses Isolated</span>
          <span className="font-semibold text-slate-200 mono-font">{events.length}</span>
        </div>
      </div>

      {/* Graphical Respiration Track */}
      <div className="relative bg-slate-950/50 p-4 rounded-lg border border-slate-900/80 flex flex-col gap-4">
        {/* Playhead line overlay */}
        {duration > 0 && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 z-10 pointer-events-none transition-all duration-75"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          />
        )}

        {/* Row 1: Breathing Inhalations */}
        <div className="relative">
          <div className="text-[10px] uppercase text-slate-500 mb-1.5 font-semibold">Respiration Events</div>
          <div className="h-6 bg-slate-900/60 rounded border border-slate-800/40 relative overflow-hidden">
            {duration > 0 && events.map((e, idx) => {
              const left = (e.start / duration) * 100;
              const width = ((e.end - e.start) / duration) * 100;
              return (
                <div
                  key={idx}
                  className="absolute h-full bg-emerald-500/25 border-x border-emerald-400/50 flex items-center justify-center text-[8px] text-emerald-300 font-semibold"
                  style={{ left: `${left}%`, width: `${width}%` }}
                  title={`Breath Pause: ${e.duration.toFixed(2)}s`}
                >
                  Inhale
                </div>
              );
            })}
            {events.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-[10px] text-slate-600">
                No respiration signatures found
              </div>
            )}
          </div>
        </div>

        {/* Row 2: Cadence Mismatch Anomalies */}
        <div className="relative">
          <div className="text-[10px] uppercase text-slate-500 mb-1.5 font-semibold">Alignment Anomalies</div>
          <div className="h-6 bg-slate-900/60 rounded border border-slate-800/40 relative">
            {duration > 0 && mismatches.map((m, idx) => {
              const left = (m.timestamp / duration) * 100;
              return (
                <div
                  key={idx}
                  className="absolute w-3 h-3 bg-red-500 rounded-full border border-white -translate-x-1.5 translate-y-1.5 animate-pulse cursor-pointer flex items-center justify-center"
                  style={{ left: `${left}%` }}
                  title={`Anomaly: ${m.reason}`}
                />
              );
            })}
            {mismatches.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-[10px] text-emerald-500/80 gap-1">
                <CheckCircle2 className="w-3 h-3" />
                No respiration overlays detected
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Forensic Findings */}
      <div className="flex flex-col gap-2">
        {mismatches.map((m, idx) => (
          <div
            key={idx}
            className="flex items-start gap-2 bg-red-950/20 p-2.5 rounded-lg border border-red-950 text-xs text-red-300"
          >
            <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold">Anomaly detected at {m.timestamp.toFixed(2)}s</div>
              <div className="text-slate-400 mt-0.5">{m.reason}</div>
            </div>
          </div>
        ))}
        {mismatches.length === 0 && (
          <div className="text-xs text-slate-400 italic text-center py-1">
            Respiration cycle aligns correctly with vocal stop-consonants and pauses.
          </div>
        )}
      </div>
    </div>
  );
};
