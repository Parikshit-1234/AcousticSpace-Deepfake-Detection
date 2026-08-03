import React, { useState, useEffect, useRef } from 'react';
import { Activity, Disc, Sparkles, Radio, Zap } from 'lucide-react';

interface AnalysisOverlayProps {
  filename?: string;
}

export const AnalysisOverlay: React.FC<AnalysisOverlayProps> = ({ filename }) => {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [sinePhase, setSinePhase] = useState(0);
  
  const animFrameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(performance.now());

  const steps = [
    { title: "Decoding PCM Audio Stream", detail: "Analyzing sample rate, bit depth & multi-channel PCM arrays...", code: "PCM_24BIT_16KHZ" },
    { title: "Extracting Room Impulse Response (RIR)", detail: "Evaluating early reflections, RT60 reverberation decay & C50 clarity...", code: "RIR_ESTIMATION" },
    { title: "Scanning Respiration & Cadence", detail: "Detecting friction noise spectrum, chest expansion & pause markers...", code: "RESP_COHERENCE" },
    { title: "Evaluating Fine-Tuned Transformer Ensemble", detail: "Running Wav2Vec 2.0 / XLS-R & WavLM Speech Representation Models...", code: "TRANSFORMER_ENSEMBLE" },
    { title: "Fusing Forensic Decision Matrix", detail: "Calculating overall spoof probability & synthetic vocal biometrics...", code: "FUSION_CALC" }
  ];

  // 60 FPS Ultra-Smooth RequestAnimationFrame Progress Loop
  useEffect(() => {
    startTimeRef.current = performance.now();

    const animateProgress = (now: number) => {
      const elapsed = now - startTimeRef.current;
      // Target 100% completion in ~800ms with smooth cubic ease-out curve
      const duration = 850;
      const linearRatio = Math.min(elapsed / duration, 1.0);
      
      // Cubic ease-out: 1 - (1 - t)^3 for silky liquid motion
      const easedRatio = 1 - Math.pow(1 - linearRatio, 3);
      const currentVal = Math.min(Math.floor(easedRatio * 99), 99);
      
      setProgress(currentVal);
      setSinePhase((prev) => (prev + 0.15) % (Math.PI * 2));

      if (linearRatio < 1.0) {
        animFrameRef.current = requestAnimationFrame(animateProgress);
      }
    };

    animFrameRef.current = requestAnimationFrame(animateProgress);

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, []);

  // Update current step smoothly based on progress
  useEffect(() => {
    if (progress < 20) setCurrentStep(0);
    else if (progress < 45) setCurrentStep(1);
    else if (progress < 68) setCurrentStep(2);
    else if (progress < 88) setCurrentStep(3);
    else setCurrentStep(4);
  }, [progress]);

  // Generate smooth cubic bezier SVG path for 60 FPS oscilloscope waveform
  const generateSmoothOscilloscopePath = () => {
    const width = 360;
    const height = 40;
    const midY = height / 2;
    let d = `M 0 ${midY}`;
    
    const points = [];
    for (let x = 0; x <= width; x += 12) {
      const freq1 = Math.sin(x * 0.04 + sinePhase) * 11;
      const freq2 = Math.cos(x * 0.09 - sinePhase * 1.3) * 5;
      const y = midY + freq1 + freq2;
      points.push({ x, y });
    }

    for (let i = 0; i < points.length - 1; i++) {
      const curr = points[i];
      const next = points[i + 1];
      const controlX = (curr.x + next.x) / 2;
      d += ` C ${controlX} ${curr.y}, ${controlX} ${next.y}, ${next.x} ${next.y}`;
    }

    return d;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/85 backdrop-blur-xl transition-opacity duration-500 ease-in-out">
      {/* Background Cybernetic Grid Lines */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-15" 
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(99, 102, 241, 0.2) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(99, 102, 241, 0.2) 1px, transparent 1px)
          `,
          backgroundSize: '32px 32px'
        }}
      />
      
      {/* Ambient Pulsing Glow Background Orbs */}
      <div className="absolute w-[500px] h-[500px] bg-indigo-600/15 rounded-full blur-3xl animate-pulse pointer-events-none transition-all duration-700" />
      <div className="absolute w-[350px] h-[350px] bg-cyan-500/10 rounded-full blur-2xl animate-pulse pointer-events-none transition-all duration-700" style={{ animationDelay: '1s' }} />

      {/* Main Glassmorphism Cybernetic HUD Container */}
      <div className="w-full max-w-xl glass-panel p-6 md:p-8 flex flex-col items-center gap-6 relative overflow-hidden border border-indigo-500/40 shadow-2xl shadow-indigo-950/80 rounded-2xl z-10 glow-indigo transform transition-all duration-300">
        
        {/* Holographic Top Status Bar */}
        <div className="flex items-center justify-between w-full border-b border-indigo-900/60 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="relative flex items-center justify-center">
              <span className="w-3 h-3 rounded-full bg-cyan-400 animate-ping absolute" />
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 relative" />
            </div>
            <div>
              <span className="text-xs font-extrabold tracking-widest uppercase text-indigo-300 title-font block">
                AcousticSpace Forensics HUD
              </span>
              <span className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
                <Radio className="w-3 h-3 text-cyan-400 animate-pulse" /> LIVE DEEPFAKE SCANNER v2.4
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-cyan-300 bg-indigo-950/80 px-3 py-1 rounded-lg border border-cyan-500/40 shadow-sm shadow-cyan-500/20 transition-all duration-200">
              {progress}% COMPLETED
            </span>
          </div>
        </div>

        {/* Central Futuristic Radar & Sonar HUD Scanner */}
        <div className="relative w-52 h-52 flex items-center justify-center my-1">
          {/* Outer Ring with Cardinal Ticks */}
          <div className="absolute inset-0 rounded-full border border-indigo-500/30 flex items-center justify-center">
            <div className="absolute -top-2 text-[9px] font-mono text-indigo-400 font-bold">000°</div>
            <div className="absolute -right-3 text-[9px] font-mono text-indigo-400 font-bold">090°</div>
            <div className="absolute -bottom-2 text-[9px] font-mono text-indigo-400 font-bold">180°</div>
            <div className="absolute -left-3 text-[9px] font-mono text-indigo-400 font-bold">270°</div>
          </div>

          {/* Rotating Dashed Radar Target Circle */}
          <div className="absolute inset-2 rounded-full border-2 border-dashed border-cyan-500/40 animate-spin-slow" />
          
          {/* Pulsing Concentric Sonar Echo Wave Circles */}
          <div className="absolute w-44 h-44 rounded-full border border-indigo-500/20 animate-ping opacity-40" style={{ animationDuration: '2.5s' }} />
          <div className="absolute w-36 h-36 rounded-full border border-cyan-400/30" />
          <div className="absolute w-24 h-24 rounded-full border border-indigo-400/50 bg-indigo-950/40" />

          {/* Central Pulsing Icon */}
          <div className="relative z-10 bg-gradient-to-br from-indigo-600/30 to-cyan-500/20 p-4 rounded-full border border-indigo-400/60 shadow-xl shadow-indigo-500/30">
            <Activity className="w-10 h-10 text-cyan-300 animate-pulse" />
          </div>

          {/* Laser Scanning Sweep Radar Needle */}
          <div 
            className="absolute inset-0 rounded-full pointer-events-none opacity-60"
            style={{
              background: 'conic-gradient(from 0deg at 50% 50%, rgba(6, 182, 212, 0.5) 0deg, rgba(99, 102, 241, 0.1) 45deg, transparent 90deg)',
              animation: 'spin-slow 2.5s linear infinite'
            }}
          />
        </div>

        {/* Realtime Smooth Oscilloscope Waveform Display */}
        <div className="w-full bg-slate-950/70 p-3 rounded-xl border border-indigo-900/50 flex flex-col gap-1.5 relative overflow-hidden">
          <div className="flex justify-between items-center text-[10px] text-slate-400 font-mono px-1">
            <span className="flex items-center gap-1 text-cyan-400 font-semibold">
              <Zap className="w-3 h-3" /> AUDIO SIGNAL OSCILLOSCOPE
            </span>
            <span className="text-indigo-400">16,000 Hz • PCM MONO</span>
          </div>

          <div className="h-10 w-full flex items-center justify-center relative">
            <svg className="w-full h-full" viewBox="0 0 360 40">
              <path
                d={generateSmoothOscilloscopePath()}
                fill="none"
                stroke="url(#cyan-indigo-gradient)"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ transition: 'd 0.08s linear' }}
              />
              <defs>
                <linearGradient id="cyan-indigo-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#22d3ee" />
                  <stop offset="50%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#d946ef" />
                </linearGradient>
              </defs>
            </svg>
          </div>
        </div>

        {/* 24-Band Smooth Dynamic Audio Spectrum Equalizer */}
        <div className="flex items-end justify-center gap-1 h-9 w-full px-4">
          {[45, 80, 35, 95, 60, 100, 50, 85, 40, 90, 70, 30, 85, 60, 95, 40, 75, 85, 50, 90, 65, 40, 80, 55].map((height, i) => (
            <div
              key={i}
              className="flex-1 bg-gradient-to-t from-indigo-600 via-cyan-400 to-fuchsia-400 rounded-t-xs"
              style={{
                height: `${Math.max(15, (height * (progress + 20)) / 120)}%`,
                transition: 'height 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                animation: `equalizer-bar 1.${(i % 5) + 2}s ease-in-out infinite alternate`,
                animationDelay: `${i * 0.04}s`
              }}
            />
          ))}
        </div>

        {/* Realtime Telemetry Grid */}
        <div className="grid grid-cols-4 gap-2 w-full">
          <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800 text-center">
            <span className="text-[9px] text-slate-400 block uppercase font-mono">RT60 Decay</span>
            <span className="text-xs font-bold font-mono text-cyan-300">0.38s</span>
          </div>
          <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800 text-center">
            <span className="text-[9px] text-slate-400 block uppercase font-mono">C50 Clarity</span>
            <span className="text-xs font-bold font-mono text-indigo-300">13.5 dB</span>
          </div>
          <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800 text-center">
            <span className="text-[9px] text-slate-400 block uppercase font-mono">Respiration</span>
            <span className="text-xs font-bold font-mono text-emerald-300">98.2%</span>
          </div>
          <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800 text-center">
            <span className="text-[9px] text-slate-400 block uppercase font-mono">Ensemble</span>
            <span className="text-xs font-bold font-mono text-fuchsia-300">Transformers</span>
          </div>
        </div>

        {/* Live Forensic Scanning Step Information */}
        <div className="w-full bg-slate-900/80 p-4 rounded-xl border border-indigo-900/40 flex flex-col gap-3 transition-all duration-300">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 flex items-center gap-1.5 font-medium">
              <Disc className="w-3.5 h-3.5 text-indigo-400 shrink-0" /> Target Evidence File:
            </span>
            <span className="font-mono font-bold text-slate-200 truncate max-w-[220px]">
              {filename || "audio_trace.wav"}
            </span>
          </div>

          {/* Current Step Description */}
          <div className="border-t border-slate-800/80 pt-2.5 flex flex-col gap-1.5 transition-opacity duration-300">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400 animate-spin" style={{ animationDuration: '3s' }} />
                <span className="text-xs font-bold text-slate-100 title-font">
                  Phase {currentStep + 1}: {steps[currentStep].title}
                </span>
              </div>
              <span className="text-[9px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/60">
                {steps[currentStep].code}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 pl-6 leading-relaxed">
              {steps[currentStep].detail}
            </p>
          </div>
        </div>

        {/* Silky Liquid Progress Bar */}
        <div className="w-full flex flex-col gap-1.5">
          <div className="w-full h-2.5 bg-slate-950 rounded-full overflow-hidden border border-indigo-900/60 p-0.5 shadow-inner">
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-cyan-400 to-fuchsia-500 relative"
              style={{
                width: `${progress}%`,
                transition: 'width 0.12s linear',
                boxShadow: '0 0 12px rgba(6, 182, 212, 0.5)'
              }}
            >
              <div className="absolute inset-0 bg-white/20 animate-pulse rounded-full" />
            </div>
          </div>
          <div className="flex justify-between items-center text-[10px] text-slate-400 font-mono">
            <span>AcousticSpace Fine-Tuned Transformer Matrix</span>
            <span className="text-cyan-400 font-semibold">{progress < 99 ? "Analyzing Vocal Biometrics..." : "Finalizing Matrix..."}</span>
          </div>
        </div>

      </div>
    </div>
  );
};
