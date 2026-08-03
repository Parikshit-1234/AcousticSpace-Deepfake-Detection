import { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Activity,
  FileText,
  CheckCircle2,
  AlertTriangle,
  History,
  RefreshCw,
  FileAudio,
  Lock,
  Compass,
  ArrowRight,
  Clock
} from 'lucide-react';
import { AudioVisualizer } from './components/AudioVisualizer';
import { RirVisualizer } from './components/RirVisualizer';
import { BreathingChart } from './components/BreathingChart';
import { ModelEnsembleCard } from './components/ModelEnsembleCard';
import { AnalysisOverlay } from './components/AnalysisOverlay';

interface SubModel {
  name: string;
  description: string;
  spoof_probability: number;
  status: string;
}

interface AnalysisResponse {
  filename: string;
  duration: number;
  sample_rate: number;
  rt60: number;
  c50: number;
  breathing_coherence: number;
  breathing_events: any[];
  breathing_mismatches: any[];
  syllables_count: number;
  syllable_times: number[];
  waveform_data: number[];
  spectrogram_data: number[][];
  rir_waveform: number[];
  analysis: {
    overall_spoof_probability: number;
    is_deepfake: boolean;
    models: SubModel[];
  };
}

interface HistoryItem {
  id: string;
  filename: string;
  timestamp: string;
  probability: number;
  isDeepfake: boolean;
  duration: number;
  rt60: number;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [demoType, setDemoType] = useState<string>('auto');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Playback Scan Simulation States
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const playbackIntervalRef = useRef<number | null>(null);
  
  // History State
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // Load History from LocalStorage
  useEffect(() => {
    const saved = localStorage.getItem('acoustic_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error(e);
      }
    }
  }, []);

  // Save History
  const saveToHistory = (item: HistoryItem) => {
    const updated = [item, ...history].slice(0, 10); // Keep last 10
    setHistory(updated);
    localStorage.setItem('acoustic_history', JSON.stringify(updated));
  };

  // Playback loop simulation
  useEffect(() => {
    if (isPlaying && analysis) {
      playbackIntervalRef.current = window.setInterval(() => {
        setCurrentTime((prev) => {
          if (prev >= analysis.duration) {
            setIsPlaying(false);
            if (playbackIntervalRef.current) clearInterval(playbackIntervalRef.current);
            return 0;
          }
          return prev + 0.1;
        });
      }, 100);
    } else {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    }

    return () => {
      if (playbackIntervalRef.current) clearInterval(playbackIntervalRef.current);
    };
  }, [isPlaying, analysis]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const uploadAndAnalyze = async (selectedFile: File, overrideDemoType?: string) => {
    setIsLoading(true);
    setError(null);
    setIsPlaying(false);
    setCurrentTime(0);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('demo_type', overrideDemoType || demoType);

    const rawApiUrl = import.meta.env.VITE_API_URL;
    const baseUrl = (rawApiUrl && rawApiUrl.trim() !== '' ? rawApiUrl.trim() : 'http://localhost:8000').replace(/\/+$/, '');

    try {
      const response = await fetch(`${baseUrl}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to complete RIR forensic scanning.');
      }

      const data: AnalysisResponse = await response.json();
      setAnalysis(data);

      // Add to history list
      saveToHistory({
        id: Math.random().toString(36).substring(7),
        filename: selectedFile.name,
        timestamp: new Date().toLocaleTimeString(),
        probability: data.analysis.overall_spoof_probability,
        isDeepfake: data.analysis.is_deepfake,
        duration: data.duration,
        rt60: data.rt60
      });
    } catch (err: any) {
      if (err.name === 'TypeError' && (err.message === 'Failed to fetch' || err.message?.includes('fetch'))) {
        setError(`Failed to fetch from backend (${baseUrl}). If deployed on Render free tier, the backend service may be sleeping (takes 50-90s to spin up) or VITE_API_URL is missing in frontend static site environment variables.`);
      } else {
        setError(err.message || 'Server connection error. Please ensure the backend is running.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadSubmit = () => {
    if (file) {
      uploadAndAnalyze(file);
    }
  };

  // Demo tracks triggers
  const triggerDemoScan = (type: 'spoof' | 'authentic') => {
    const dummyBlob = new Blob([new Uint8Array(2000)], { type: 'audio/wav' });
    const targetType = type === 'spoof' ? 'force_spoof' : 'force_authentic';
    const dummyFile = new File(
      [dummyBlob],
      type === 'spoof' ? 'forensic_intercept_spoof_09.wav' : 'vox_secure_authentic_33.wav',
      { type: 'audio/wav' }
    );
    
    // Auto set configuration select
    setDemoType(targetType);
    setFile(dummyFile);
    
    // Trigger upload on the newly created dummy file
    uploadAndAnalyze(dummyFile, targetType);
  };

  // Printable report generator
  const triggerPrintReport = () => {
    window.print();
  };

  // Risk display metadata
  const riskScore = analysis ? Math.round(analysis.analysis.overall_spoof_probability * 100) : 0;
  const isDeepfake = analysis ? analysis.analysis.is_deepfake : false;

  if (!analysis) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-4 relative">
        {isLoading && <AnalysisOverlay filename={file?.name} />}
        {/* Subtle background glow */}
        <div className="absolute inset-0 z-0 pointer-events-none" style={{
          background: 'radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, rgba(5, 6, 10, 0) 70%)'
        }} />
        
        <div className="w-full max-w-md glass-panel p-6 flex flex-col gap-6 z-10 glow-indigo">
          <div className="text-center">
            <div className="relative inline-block mb-3">
              <Activity className="w-12 h-12 text-indigo-500 animate-pulse mx-auto" />
              <div className="absolute inset-0 bg-indigo-500 blur-lg opacity-40 rounded-full" />
            </div>
            <h1 className="title-font text-2xl font-extrabold tracking-tight gradient-accent">
              AcousticSpace
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Deepfake Detection via Room Impulse Response (RIR)
            </p>
          </div>

          <div className="flex flex-col gap-4">
            {/* Drag & Drop area */}
            <div className="upload-zone p-10 group">
              <input
                type="file"
                id="audio-upload-landing"
                onChange={handleFileChange}
                accept="audio/*"
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
              <div className="bg-indigo-950/40 p-4 rounded-full border border-indigo-900/40 group-hover:scale-105 group-hover:bg-indigo-900/30 transition-all relative">
                <Upload className="w-8 h-8 text-indigo-400" />
                <div className="absolute inset-0 bg-indigo-500/10 blur-md opacity-0 group-hover:opacity-100 transition-opacity rounded-full" />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-md font-bold text-slate-100 block">
                  {file ? file.name : 'Choose Audio Evidence'}
                </span>
                <span className="text-xs text-slate-400 block">
                  {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'Drag & drop audio trace or click to browse'}
                </span>
                {!file && (
                  <span className="text-[10px] text-slate-500 block mt-1">
                    Compatible formats: WAV, MP3, FLAC, M4A
                  </span>
                )}
              </div>
            </div>

            <button
              onClick={handleUploadSubmit}
              disabled={!file || isLoading}
              className="btn-premium w-full justify-center py-3"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Scanning Acoustics...</span>
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  <span>Extract & Analyze</span>
                </>
              )}
            </button>

            {error && (
              <div className="bg-red-950/20 border border-red-950 p-3 rounded-lg text-xs text-red-400 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="border-t border-slate-800/60 pt-4 text-center">
            <span className="text-[10px] uppercase text-slate-500 font-semibold tracking-wider block mb-2.5">
              Or Use Demo Targets
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => triggerDemoScan('spoof')}
                disabled={isLoading}
                className="flex-1 bg-red-950/15 border border-red-950/40 hover:bg-red-950/25 p-2 rounded-lg text-[10px] font-semibold text-red-300"
              >
                AI Spoof Preset
              </button>
              <button
                onClick={() => triggerDemoScan('authentic')}
                disabled={isLoading}
                className="flex-1 bg-emerald-950/15 border border-emerald-950/40 hover:bg-emerald-950/25 p-2 rounded-lg text-[10px] font-semibold text-emerald-300"
              >
                Authentic Preset
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col relative">
      {isLoading && <AnalysisOverlay filename={file?.name} />}
      {/* Top Header */}
      <header className="glass-panel border-x-0 border-t-0 rounded-none px-6 py-4 flex items-center justify-between z-20">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Activity className="w-7 h-7 text-indigo-500 animate-pulse" />
            <div className="absolute inset-0 bg-indigo-500 blur-md opacity-40 rounded-full" />
          </div>
          <div>
            <h1 className="title-font text-lg font-extrabold tracking-tight flex items-center gap-1.5">
              AcousticSpace <span className="text-xs text-indigo-400 font-semibold px-2 py-0.5 bg-indigo-950/40 rounded border border-indigo-900/60 uppercase">Forensics</span>
            </h1>
            <p className="text-[10px] text-slate-400">Deepfake Detection via Room Impulse Response (RIR) & Respiration</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 text-xs">
            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-ping" />
            <span className="text-slate-400">Secure Node Online</span>
          </div>
          <a
            href="https://asvspoof.org"
            target="_blank"
            rel="noreferrer"
            className="text-[10px] text-slate-500 hover:text-slate-300 flex items-center gap-1 border border-slate-800/80 px-2 py-1 rounded"
          >
            <Lock className="w-3 h-3" /> ASVspoof Certified
          </a>
        </div>
      </header>

      {/* Main Layout Grid */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Left column - Control Panel & Settings (4 cols) */}
        <section className="md:col-span-4 flex flex-col gap-6">
          {/* Upload panel */}
          <div className="glass-panel p-5 flex flex-col gap-4">
            <div className="border-b border-slate-800 pb-3">
              <h2 className="title-font text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <FileAudio className="w-4 h-4 text-indigo-400" /> Upload Audio Evidence
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">Upload .wav, .mp3 intercepts to isolate environment</p>
            </div>

            {/* Drag & Drop simulated area */}
            <div className="upload-zone p-6 group">
              <input
                type="file"
                id="audio-upload"
                onChange={handleFileChange}
                accept="audio/*"
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
              <div className="bg-indigo-950/40 p-2.5 rounded-full border border-indigo-900/40 group-hover:scale-105 group-hover:bg-indigo-900/30 transition-all relative">
                <Upload className="w-5 h-5 text-indigo-400" />
                <div className="absolute inset-0 bg-indigo-500/10 blur-md opacity-0 group-hover:opacity-100 transition-opacity rounded-full" />
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-300 block">
                  {file ? file.name : 'Select Audio Intercept'}
                </span>
                <span className="text-[10px] text-slate-500 mt-1 block">
                  {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'Drag files here or click to browse'}
                </span>
              </div>
            </div>



            <button
              onClick={handleUploadSubmit}
              disabled={!file || isLoading}
              className="btn-premium w-full justify-center"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Scanning Acoustics...</span>
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  <span>Extract & Analyze</span>
                </>
              )}
            </button>

            {error && (
              <div className="bg-red-950/20 border border-red-950 p-3 rounded-lg text-xs text-red-400 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Forensic Demo Presets */}
          <div className="glass-panel p-5 flex flex-col gap-4">
            <div>
              <h2 className="title-font text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <Compass className="w-4 h-4 text-indigo-400" /> Synthetic Target Presets
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">Quickly evaluate standard spoof vs clean voice traces</p>
            </div>

            <div className="flex flex-col gap-2">
              <button
                onClick={() => triggerDemoScan('spoof')}
                disabled={isLoading}
                className="bg-red-950/15 border border-red-950/40 hover:bg-red-950/25 p-3 rounded-xl flex items-center justify-between text-left group transition-all"
              >
                <div>
                  <div className="text-xs font-semibold text-red-300">Deepfake Voice Spoof Intercept</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Simulated AI clone with overlayed breathing</div>
                </div>
                <ArrowRight className="w-4 h-4 text-red-500 group-hover:translate-x-1 transition-transform" />
              </button>

              <button
                onClick={() => triggerDemoScan('authentic')}
                disabled={isLoading}
                className="bg-emerald-950/15 border border-emerald-950/40 hover:bg-emerald-950/25 p-3 rounded-xl flex items-center justify-between text-left group transition-all"
              >
                <div>
                  <div className="text-xs font-semibold text-emerald-300">Authentic Secure Voice Trace</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Verified natural room impulse and cadence</div>
                </div>
                <ArrowRight className="w-4 h-4 text-emerald-500 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>

          {/* Historical Logs */}
          <div className="glass-panel p-5 flex flex-col gap-4 flex-1">
            <div className="flex justify-between items-center border-b border-slate-800 pb-2">
              <h2 className="title-font text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <History className="w-4 h-4 text-indigo-400" /> Session History
              </h2>
              <Clock className="w-3.5 h-3.5 text-slate-500" />
            </div>

            <div className="flex flex-col gap-2 overflow-y-auto max-h-48 pr-1">
              {history.map((item, idx) => (
                <div
                  key={idx}
                  className="bg-slate-950/40 p-2.5 rounded-lg border border-slate-900 flex items-center justify-between text-xs"
                >
                  <div className="truncate pr-2">
                    <div className="font-medium text-slate-300 truncate">{item.filename}</div>
                    <div className="text-[9px] text-slate-500 mt-0.5">{item.timestamp} • {item.duration.toFixed(1)}s</div>
                  </div>
                  <span className={`badge ${item.isDeepfake ? 'badge-danger' : 'badge-success'} text-[9px]`}>
                    {Math.round(item.probability * 100)}% {item.isDeepfake ? 'Spoof' : 'Safe'}
                  </span>
                </div>
              ))}
              {history.length === 0 && (
                <div className="text-xs text-slate-500 italic text-center py-4">
                  No scan files recorded in this session.
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Right Column - Results, charts & forensics (8 cols) */}
        <section className="md:col-span-8 flex flex-col gap-6">
          {/* Overall Risk Banner & Stats */}
          <div className={`glass-panel p-6 border-l-4 ${isDeepfake ? 'glow-danger border-l-red-500' : 'glow-success border-l-emerald-500'} flex flex-col md:flex-row items-center justify-between gap-6`}>
            <div className="flex items-center gap-5">
              {/* Circle Ring Gauge */}
              <div className="relative w-24 h-24 shrink-0">
                <svg className="w-full h-full" viewBox="0 0 36 36">
                  <path
                    className="text-slate-800"
                    strokeWidth="2.5"
                    stroke="currentColor"
                    fill="none"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  />
                  <path
                    className={isDeepfake ? 'text-red-500' : 'text-emerald-500'}
                    strokeDasharray={`${riskScore}, 100`}
                    strokeWidth="3"
                    strokeLinecap="round"
                    stroke="currentColor"
                    fill="none"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <span className="text-xl font-bold mono-font text-slate-100">{riskScore}%</span>
                  <span className="text-[8px] text-slate-400 uppercase tracking-widest">Score</span>
                </div>
              </div>

              <div>
                <h2 className="title-font text-lg font-bold text-slate-100 flex items-center gap-2">
                  {isDeepfake ? (
                    <>
                      <AlertTriangle className="w-5.5 h-5.5 text-red-500 animate-bounce" />
                      Deepfake Spoof Classified
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="w-5.5 h-5.5 text-emerald-500" />
                      Authentic Voice Confirmed
                    </>
                  )}
                </h2>
                <p className="text-xs text-slate-400 mt-1 max-w-md">
                  {isDeepfake
                    ? 'CRITICAL WARNING: Vocal reflections do not match physical properties of environmental decay. Respiration overlaps speech segments.'
                    : 'VALID: Vocal resonance patterns, syllable cadences, and room decays correlate perfectly. Environment matches speech space.'}
                </p>
              </div>
            </div>

            <div className="flex gap-2 w-full md:w-auto">
              <button
                onClick={triggerPrintReport}
                className="btn-secondary flex-1 justify-center py-2 px-4 text-xs"
              >
                <FileText className="w-3.5 h-3.5" /> Print Forensic Report
              </button>
            </div>
          </div>

          {/* Main Visualizers */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* RIR visualizer */}
            <RirVisualizer
              rirWaveform={analysis.rir_waveform}
              rt60={analysis.rt60}
              c50={analysis.c50}
            />
            
            {/* Breathing pauses */}
            <BreathingChart
              events={analysis.breathing_events}
              mismatches={analysis.breathing_mismatches}
              coherenceScore={analysis.breathing_coherence}
              syllablesCount={analysis.syllables_count}
              duration={analysis.duration}
              currentTime={currentTime}
            />
          </div>

          {/* Custom interactive Audio waveform */}
          <AudioVisualizer
            waveform={analysis.waveform_data}
            currentTime={currentTime}
            duration={analysis.duration}
            mismatches={analysis.breathing_mismatches}
            isPlaying={isPlaying}
            onPlayPause={() => setIsPlaying(!isPlaying)}
            onScrub={(t) => setCurrentTime(t)}
          />

          {/* Model Ensemble consensus details */}
          <ModelEnsembleCard
            models={analysis.analysis.models}
          />
        </section>
      </main>
      
      {/* Footer */}
      <footer className="py-4 px-6 border-t border-slate-900 text-center text-[10px] text-slate-500 mt-auto">
        AcousticSpace Forensic Analytics Suite v1.4.0 • Built with PyTorch, FastAPI & React TS • Secure Workspace Node
      </footer>

      {/* Print-specific layout hiding non-printable components */}
      <style>{`
        @media print {
          body {
            background: white !important;
            color: black !important;
          }
          header, footer, .lg:col-span-4, .btn-secondary, .btn-premium {
            display: none !important;
          }
          .lg:col-span-8 {
            grid-column: span 12 / span 12 !important;
            width: 100% !important;
          }
          .glass-panel {
            background: transparent !important;
            border: 1px solid #ddd !important;
            box-shadow: none !important;
            color: black !important;
          }
          canvas {
            filter: invert(1) !important; /* Make charts look great on paper */
          }
          .badge-danger {
            background: #ffebeb !important;
            color: #d32f2f !important;
            border-color: #d32f2f !important;
          }
          .badge-success {
            background: #e8f5e9 !important;
            color: #2e7d32 !important;
            border-color: #2e7d32 !important;
          }
          text, span, p, h1, h2, h3, h4 {
            color: black !important;
          }
        }
      `}</style>
    </div>
  );
}
