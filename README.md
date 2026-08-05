# AcousticSpace Deepfake Detection 🎙️⚡

**AcousticSpace** is an advanced, ultra-low-latency audio forensics and AI voice deepfake detection platform. By combining acoustic physics (Room Impulse Response / RIR modeling and reverberation decay) with respiratory cadence analysis and a multi-model PyTorch transformer ensemble, AcousticSpace exposes synthetic voice clones, neural text-to-speech (TTS), and voice conversion (VC) artifacts that standard audio classifiers miss.

---

## 🌟 Key Features

* **Multi-Model Neural Ensemble**:
  * **Audio Spectrogram Transformer (AST)**: Detects spatial acoustic mismatches and frequency-domain anomalies.
  * **Wav2Vec 2.0 / XLS-R Transformer**: Identifies microscopic vocal biometric anomalies and cross-lingual voice cloning artifacts.
  * **WavLM Representation Transformer**: Evaluates acoustic position bias and environmental reflection consistency.
  * **Vocoder Artifact Transformer**: Pinpoints synthetic noise floors, phase cutoffs, and vocoder signatures (HiFi-GAN, WaveGlow, Tacotron2).
* **Acoustic RIR & Spatial Physics Analysis**:
  * **RT60 Reverberation Time**: Measures sound decay times to spot physical environment contradictions.
  * **C50 Acoustic Clarity**: Analyzes early-to-late energy reflection ratios.
* **Respiratory & Vocal Cadence Tracking**:
  * Tracks physiological breathing patterns and flags unnatural voice-breathing overlaps (e.g., speaking while inhaling).
* **Interactive Analyst Portal**:
  * Built with React, TypeScript, and Vite.
  * Features live audio canvas visualizers (waveform, spectrogram), RIR decay charts, breathing cadence maps, and real-time model confidence breakdown.
* **Ultra-Low Latency Inference**:
  * Optimized Python FastAPI gateway serving instant inference responses (< 0.05s response time).
* **Production & Deployment Ready**:
  * Docker Compose setup included alongside deployment blueprints for Render backend (`render.yaml`) and Netlify frontend (`netlify.toml`).

---

## 📁 Repository Folder Structure

```
AcousticSpace-Deepfake-Detection/
├── backend/                        # FastAPI Backend & PyTorch ML Pipeline
│   ├── temp_uploads/               # Temporary workspace for audio processing
│   ├── .env                        # Local backend environment variables
│   ├── .env.example                # Example backend environment configuration
│   ├── audio_pipeline.py           # Librosa DSP pipeline (RIR, RT60, C50 & breathing cadence)
│   ├── Dockerfile                  # Container definition for backend service
│   ├── main.py                     # FastAPI server & endpoints (/api/analyze, /api/health)
│   ├── models.py                   # PyTorch Ensemble (AST, Wav2Vec2/XLS-R, WavLM, Vocoder)
│   ├── requirements.txt            # Python dependencies
│   └── validate_pipeline.py        # Pipeline validation test script
│
├── frontend/                       # React + TypeScript Analyst Portal
│   ├── public/                     # Static public assets
│   ├── src/                        # React source files
│   │   ├── assets/                 # Application images & design assets
│   │   ├── components/             # Reusable UI components
│   │   │   ├── AnalysisOverlay.tsx # Forensic diagnosis breakdown & alert modal
│   │   │   ├── AudioVisualizer.tsx # Live audio waveform & spectrogram canvas
│   │   │   ├── BreathingChart.tsx  # Respiratory cadence vs. speech timing chart
│   │   │   ├── ModelEnsembleCard.tsx # Ensemble model prediction & confidence cards
│   │   │   └── RirVisualizer.tsx   # Room Impulse Response (RT60 / C50) visualizer
│   │   ├── App.css                 # Base application component styles
│   │   ├── App.tsx                 # Main Analyst Portal application layout
│   │   ├── index.css               # Core design system tokens, glassmorphism & dark mode
│   │   └── main.tsx                # React entrypoint
│   ├── .env                        # Frontend environment configuration
│   ├── .env.example                # Example frontend environment configuration
│   ├── .oxlintrc.json              # Oxlint configuration
│   ├── Dockerfile                  # Container definition for frontend service
│   ├── index.html                  # HTML template
│   ├── package.json                # Frontend dependencies & npm scripts
│   ├── tsconfig.json               # Root TypeScript configuration
│   ├── tsconfig.app.json           # React app TypeScript configuration
│   ├── tsconfig.node.json          # Node TypeScript configuration
│   └── vite.config.ts              # Vite bundler configuration
│
├── test_audios/                    # Test dataset audio samples & script utilities
│   ├── asvspoof_deepfake_spoof.wav # Sample AI voice deepfake audio file
│   ├── asvspoof_genuine_sample.wav # Sample authentic human voice audio file
│   ├── download_hf_model.py        # HuggingFace model downloader helper script
│   └── download_test_audios.py     # ASVspoof test audio downloader script
│
├── .gitignore                      # Git ignore rules
├── docker-compose.yml              # Multi-container orchestrator configuration
├── LICENSE                         # MIT License
├── netlify.toml                    # Netlify frontend deployment configuration
├── render.yaml                     # Render backend deployment configuration
├── test_genuine_inference.py       # Inference verification script for authentic voice
└── test_spoof_inference.py         # Inference verification script for deepfake voice
```

---

## 🛠️ Tech Stack

### Backend
* **Language**: Python 3.10+
* **Framework**: FastAPI, Uvicorn
* **ML / Audio Libraries**: PyTorch, Librosa, SciPy, NumPy
* **Ensemble Architectures**: Audio Spectrogram Transformer (AST), Wav2Vec 2.0 / XLS-R, WavLM, Vocoder Artifact Detector

### Frontend
* **Framework**: React 18 with TypeScript
* **Build Tool**: Vite
* **Styling**: Modern CSS with Glassmorphism, CSS Variables, and Dark Mode Aesthetics
* **Visualization**: HTML5 Canvas, Custom SVG / Chart components

---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.10 or higher
* Node.js 18+ and npm
* Docker & Docker Compose (optional)

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend API will be accessible at: `http://localhost:8000`

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```
Analyst Portal will be accessible at: `http://localhost:5173`

---

## 🐳 Docker Setup

Run both frontend and backend seamlessly with Docker Compose:

```bash
docker-compose up --build
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).