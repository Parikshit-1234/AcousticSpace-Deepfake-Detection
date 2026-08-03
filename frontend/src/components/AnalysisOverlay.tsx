import React, { useState, useEffect } from 'react';
import { Activity, Disc, Sparkles } from 'lucide-react';

interface AnalysisOverlayProps {
  filename?: string;
}

export const AnalysisOverlay: React.FC<AnalysisOverlayProps> = ({ filename }) => {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { title: "Decoding PCM Audio Stream", detail: "Checking sample rates & 16-bit PCM channel arrays..." },
    { title: "Extracting Room Impulse Response (RIR)", detail: "Measuring early reflections, RT60 decay & C50 clarity..." },
    { title: "Scanning Respiration & Cadence", detail: "Detecting friction noise spectrum & vocal tract pauses..." },
    { title: "Evaluating Neural Ensemble", detail: "Running Wav2Vec2, WavLM & Spatial Acoustic Classifiers..." },
    { title: "Fusing Forensic Decision Matrix", detail: "Calculating overall spoof probability & biometrics..." }
  ];

  useEffect(() => {
    // Smooth progress simulation
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev < 95) {
          const next = prev + Math.floor(Math.random() * 6) + 2;
          return Math.min(next, 95);
        }
        return prev;
      });
    }, 180);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Update step text based on progress percentage
    if (progress < 20) setCurrentStep(0);
    else if (progress < 45) setCurrentStep(1);
    else if (progress < 70) setCurrentStep(2);
    else if (progress < 88) setCurrentStep(3);
    else setCurrentStep(4);
  }, [progress]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md transition-opacity duration-300">
      <div className="w-full max-w-lg glass-panel p-8 flex flex-col items-center gap-6 relative overflow-hidden border border-indigo-500/30 shadow-2xl shadow-indigo-950/50 rounded-2xl">
        
        {/* Top Scan Status Header */}
        <div className="flex items-center justify-between w-full border-b border-indigo-900/40 pb-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-indigo-500 animate-ping" />
            <span className="text-xs font-bold tracking-wider uppercase text-indigo-400 title-font">
              Forensic Acoustic Scan Active
            </span>
          </div>
          <span className="text-xs font-mono font-bold text-cyan-400 bg-indigo-950/60 px-2.5 py-1 rounded-md border border-cyan-500/30">
            {progress}% COMPLETE
          </span>
        </div>

        {/* Central Futuristic Radar / Sonar Visualizer */}
        <div className="relative w-44 h-44 flex items-center justify-center my-2">
          {/* Outer Rotating Radar Ring */}
          <div className="absolute inset-0 rounded-full border-2 border-dashed border-indigo-500/30 animate-spin-slow" />
          
          {/* Inner Pulsing Echo Rings */}
          <div className="absolute w-36 h-36 rounded-full border border-cyan-500/20 animate-ping opacity-30" style={{ animationDuration: '3s' }} />
          <div className="absolute w-28 h-28 rounded-full border border-indigo-400/30" />
          <div className="absolute w-20 h-20 rounded-full border border-indigo-500/50 bg-indigo-950/30" />

          {/* Central Pulsing Icon */}
          <div className="relative z-10 bg-indigo-600/20 p-4 rounded-full border border-indigo-400/50 shadow-lg shadow-indigo-500/20">
            <Activity className="w-10 h-10 text-indigo-400 animate-pulse" />
          </div>

          {/* Radar Scanning Sweep Needle */}
          <div 
            className="absolute inset-0 rounded-full pointer-events-none opacity-40"
            style={{
              background: 'conic-gradient(from 0deg at 50% 50%, rgba(99, 102, 241, 0.4) 0deg, rgba(6, 182, 212, 0) 60deg)',
              animation: 'spin-slow 3s linear infinite'
            }}
          />
        </div>

        {/* Live Audio Equalizer Frequency Bars */}
        <div className="flex items-end justify-center gap-1.5 h-10 w-full px-8">
          {[40, 75, 30, 90, 60, 100, 45, 80, 50, 95, 35, 70, 85, 40, 65, 90].map((height, i) => (
            <div
              key={i}
              className="flex-1 bg-gradient-to-t from-indigo-600 to-cyan-400 rounded-t-sm"
              style={{
                height: `${Math.max(15, (height * (progress + 10)) / 110)}%`,
                transition: 'height 0.15s ease-in-out',
                animation: `equalizer-bar 1.${(i % 5) + 2}s ease-in-out infinite alternate`,
                animationDelay: `${i * 0.08}s`
              }}
            />
          ))}
        </div>

        {/* File Details & Step Progress Information */}
        <div className="w-full bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Disc className="w-3.5 h-3.5 text-indigo-400 shrink-0" /> Target File:
            </span>
            <span className="font-semibold text-slate-200 truncate max-w-[200px]">
              {filename || "audio_trace.wav"}
            </span>
          </div>

          {/* Current Step Description */}
          <div className="border-t border-slate-800/80 pt-2.5 flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400 animate-spin" style={{ animationDuration: '4s' }} />
              <span className="text-xs font-bold text-slate-100 title-font">
                {steps[currentStep].title}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 pl-6 leading-relaxed">
              {steps[currentStep].detail}
            </p>
          </div>
        </div>

        {/* Progress Bar Container */}
        <div className="w-full flex flex-col gap-1.5">
          <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800 p-0.5">
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-cyan-400 to-indigo-400 transition-all duration-300 shadow-md shadow-cyan-500/30"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex justify-between items-center text-[10px] text-slate-500">
            <span>Neural RIR Scanner v2.4</span>
            <span className="font-mono">{progress < 100 ? "Processing..." : "Finalizing..."}</span>
          </div>
        </div>

      </div>
    </div>
  );
};
