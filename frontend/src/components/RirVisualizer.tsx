import React, { useRef, useEffect } from 'react';
import { Activity, Landmark } from 'lucide-react';

interface RirVisualizerProps {
  rirWaveform: number[];
  rt60: number;
  c50: number;
}

export const RirVisualizer: React.FC<RirVisualizerProps> = ({
  rirWaveform,
  rt60,
  c50,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear background
    ctx.clearRect(0, 0, width, height);

    // Draw background grid lines (horizontal and vertical)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    
    // Vertical time grid lines
    const numVLines = 10;
    for (let i = 0; i <= numVLines; i++) {
      const x = (width / numVLines) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal amplitude lines
    const numHLines = 4;
    for (let i = 1; i < numHLines; i++) {
      const y = (height / numHLines) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (rirWaveform.length === 0) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    // Draw RIR waveform
    // RIR amplitude decays exponentially. We plot it symmetrically around center.
    const center = height / 2;
    ctx.lineWidth = 1.5;
    
    // Create cyan to violet gradient for RIR impulses
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, '#06b6d4'); // Cyan for direct arrival
    gradient.addColorStop(0.1, '#6366f1'); // Indigo for early reflections
    gradient.addColorStop(0.5, 'rgba(99, 102, 241, 0.4)'); // Late decay
    gradient.addColorStop(1, 'rgba(217, 70, 239, 0.1)'); // End
    
    ctx.strokeStyle = gradient;
    ctx.beginPath();

    const step = rirWaveform.length / width;
    for (let x = 0; x < width; x++) {
      const dataIdx = Math.floor(x * step);
      const val = rirWaveform[dataIdx] || 0;
      const amp = val * (height * 0.45);
      
      if (x === 0) {
        ctx.moveTo(x, center - amp);
      } else {
        ctx.lineTo(x, center - amp);
      }
    }
    ctx.stroke();

    // 50ms marker (Early reflections limit)
    // Assuming the waveform displays 500ms of RIR, 50ms is at 10% width.
    const earlyLimitX = width * 0.1;
    ctx.strokeStyle = 'rgba(217, 70, 239, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(earlyLimitX, 0);
    ctx.lineTo(earlyLimitX, height);
    ctx.stroke();

    // Label the marker
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#d946ef';
    ctx.fillText('50ms (Early Reflections Limit)', earlyLimitX + 6, 15);

    // Direct path label at x = 0
    ctx.fillStyle = '#06b6d4';
    ctx.fillText('Direct Path', 5, height - 10);
  }, [rirWaveform]);

  // Forensic classification of the reverb profile
  const getReverbClassification = (rt: number) => {
    if (rt > 1.8) return { text: 'Church / Large Hall (High Mismatch Risk)', class: 'badge-danger', tip: 'Vocal reflections match a cathedral/hall, which is highly inconsistent with close-proximity speech.' };
    if (rt > 1.2) return { text: 'Reverberant Auditorium (Suspicious)', class: 'badge-warning', tip: 'Voice exhibits medium echo. Check if background matches auditorium noise floor.' };
    if (rt > 0.6) return { text: 'Standard Living Room / Office', class: 'badge-success', tip: 'Typical domestic acoustic reflections present.' };
    return { text: 'Anechoic Chamber / Dry Studio', class: 'badge-success', tip: 'Extremely clean recording. Reverb matches professional vocal booth.' };
  };

  const reverbRating = getReverbClassification(rt60);

  return (
    <div className="glass-panel p-4 flex flex-col gap-4">
      <div>
        <h3 className="title-font text-md font-semibold text-slate-100 flex items-center gap-2">
          <Landmark className="w-4.5 h-4.5 text-indigo-400" />
          Extracted Room Impulse Response (RIR)
        </h3>
        <p className="text-xs text-slate-400">Isolated acoustics profile reflecting early reflections and decay</p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/40 text-center">
          <div className="text-[10px] uppercase tracking-wider text-slate-500">Decay Time (RT60)</div>
          <div className="text-lg font-bold text-slate-200 mono-font mt-0.5">{rt60.toFixed(2)}s</div>
        </div>
        <div className="bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/40 text-center">
          <div className="text-[10px] uppercase tracking-wider text-slate-500">Clarity (C50)</div>
          <div className="text-lg font-bold text-slate-200 mono-font mt-0.5">{c50.toFixed(1)} dB</div>
        </div>
        <div className="bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/40 text-center flex flex-col justify-center items-center">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Acoustic Space</div>
          <span className={`badge ${reverbRating.class} text-[9px]`}>{rt60 > 1.2 ? 'Echoic' : 'Dry'}</span>
        </div>
      </div>

      <div className="relative h-28 bg-slate-950/40 rounded-lg overflow-hidden border border-slate-800/40">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
        />
      </div>

      <div className="bg-slate-950/30 p-3 rounded-lg border border-slate-900 flex items-start gap-2.5 text-xs">
        <Activity className="w-4.5 h-4.5 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">Space Classification: </span>
          <span className="text-slate-400">{reverbRating.tip}</span>
        </div>
      </div>
    </div>
  );
};
