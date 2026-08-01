import React, { useRef, useEffect } from 'react';
import { Play, Pause, AlertTriangle } from 'lucide-react';

interface Mismatch {
  timestamp: number;
  reason: string;
  severity: number;
}

interface AudioVisualizerProps {
  waveform: number[];
  currentTime: number;
  duration: number;
  mismatches: Mismatch[];
  isPlaying: boolean;
  onPlayPause: () => void;
  onScrub: (time: number) => void;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  waveform,
  currentTime,
  duration,
  mismatches,
  isPlaying,
  onPlayPause,
  onScrub,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI screens
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear background
    ctx.clearRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    const numGridLines = 6;
    for (let i = 1; i < numGridLines; i++) {
      const y = (height / numGridLines) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (waveform.length === 0) {
      // Draw a flat line if no waveform data
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    const padding = 2;
    const barWidth = (width / waveform.length) - padding;
    const center = height / 2;

    // Draw normal bars
    waveform.forEach((val, idx) => {
      const x = idx * (barWidth + padding);
      const magnitude = Math.abs(val) * (height * 0.85);
      const yStart = center - magnitude / 2;

      // Determine progress status
      const timeAtBar = (idx / waveform.length) * duration;
      const hasPassed = timeAtBar <= currentTime;

      // Check if this bar falls within any mismatch anomaly zone (within +-0.3s of timestamp)
      const isAnomaly = mismatches.some(
        (m) => Math.abs(timeAtBar - m.timestamp) < 0.3
      );

      if (isAnomaly) {
        ctx.fillStyle = hasPassed ? '#ef4444' : 'rgba(239, 68, 68, 0.4)';
      } else {
        ctx.fillStyle = hasPassed ? '#6366f1' : 'rgba(255, 255, 255, 0.15)';
      }

      // Draw rounded rectangle for bar
      ctx.beginPath();
      ctx.roundRect(x, yStart, Math.max(1.5, barWidth), Math.max(2, magnitude), 2);
      ctx.fill();
    });

    // Highlight mismatch areas on the timeline
    mismatches.forEach((m) => {
      const xPos = (m.timestamp / duration) * width;
      // Draw subtle background highlight
      const gradient = ctx.createLinearGradient(xPos - 15, 0, xPos + 15, 0);
      gradient.addColorStop(0, 'rgba(239, 68, 68, 0.0)');
      gradient.addColorStop(0.5, 'rgba(239, 68, 68, 0.08)');
      gradient.addColorStop(1, 'rgba(239, 68, 68, 0.0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(xPos - 20, 0, 40, height);

      // Draw vertical anomaly marker line
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(xPos, 0);
      ctx.lineTo(xPos, height);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Draw playhead vertical line
    const playheadX = (currentTime / duration) * width;
    if (playheadX > 0 && playheadX < width) {
      // Glow background for playhead
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 8;
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
      ctx.shadowBlur = 0; // reset

      // Playhead handle
      ctx.fillStyle = '#06b6d4';
      ctx.beginPath();
      ctx.arc(playheadX, 0, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(playheadX, height, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [waveform, currentTime, duration, mismatches]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || duration === 0) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percent = clickX / rect.width;
    const targetTime = percent * duration;
    onScrub(Math.max(0, Math.min(duration, targetTime)));
  };

  const formatTime = (time: number) => {
    const m = Math.floor(time / 60);
    const s = Math.floor(time % 60);
    const ms = Math.floor((time % 1) * 100);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  return (
    <div className="glass-panel p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="title-font text-md font-semibold text-slate-100 flex items-center gap-2">
            Spectral Amplitude & Anomalies
          </h3>
          <p className="text-xs text-slate-400">Click waveform timeline to analyze specific coordinates</p>
        </div>
        <div className="text-right">
          <div className="text-xs text-indigo-400 font-medium mono-font">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>
      </div>

      <div className="relative h-32 bg-slate-950/40 rounded-lg overflow-hidden border border-slate-800/40">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="w-full h-full cursor-ew-resize block"
        />
      </div>

      <div className="flex justify-between items-center">
        <button
          onClick={onPlayPause}
          disabled={waveform.length === 0}
          className="btn-secondary"
          style={{ padding: '8px 16px', borderRadius: '8px' }}
        >
          {isPlaying ? (
            <>
              <Pause className="w-4 h-4 text-indigo-400" />
              <span>Pause Scan</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 text-emerald-400" />
              <span>Initiate Scan</span>
            </>
          )}
        </button>

        {mismatches.length > 0 && (
          <div className="flex items-center gap-2 text-red-400 text-xs font-medium bg-red-950/30 px-3 py-1.5 rounded-full border border-red-900/30">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>{mismatches.length} Spatial-Vocal Anomaly Coordinates Identified</span>
          </div>
        )}
      </div>
    </div>
  );
};
