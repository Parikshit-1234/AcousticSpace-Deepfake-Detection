# validate_pipeline.py
import sys
import os

print("Starting pipeline validation...")

try:
    import numpy as np
    print("[SUCCESS] numpy loaded.")
except ImportError:
    print("[ERROR] numpy not installed.")
    sys.exit(1)

try:
    import scipy
    print("[SUCCESS] scipy loaded.")
except ImportError:
    print("[ERROR] scipy not installed.")
    sys.exit(1)

try:
    import librosa
    print("[SUCCESS] librosa loaded.")
except ImportError:
    print("[ERROR] librosa not installed.")
    sys.exit(1)

try:
    import torch
    print("[SUCCESS] torch loaded.")
except ImportError:
    print("[ERROR] torch not installed.")
    sys.exit(1)

try:
    import fastapi
    print("[SUCCESS] fastapi loaded.")
except ImportError:
    print("[ERROR] fastapi not installed.")
    sys.exit(1)

try:
    from audio_pipeline import AudioProcessingPipeline
    from models import EnsembleForensicClassifier
    print("[SUCCESS] AcousticSpace modules imported correctly.")
except Exception as e:
    print(f"[ERROR] Import failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nRunning self-check on model ensemble classification...")
classifier = EnsembleForensicClassifier()
dummy_dsp = {
    "rt60": 0.45,
    "c50": 12.0,
    "rir_waveform": [0.0] * 8000,
    "syllables_count": 8,
    "syllable_times": [0.3, 0.6, 1.0, 1.3, 1.6, 2.0, 2.3, 2.6],
    "breathing_events": [{"duration": 0.35, "intensity": 0.1}],
    "breathing_mismatches": [],
    "breathing_coherence": 1.0,
    "spectrogram_data": [[0.0] * 80] * 64
}

res = classifier.analyze_audio(dummy_dsp)
print("Spoof probability for clean audio:", res["overall_spoof_probability"])
print("Is deepfake classification:", res["is_deepfake"])
print("Model breakdown counts:", len(res["models"]), "models present.")

# Check spoofed logic
dummy_dsp["breathing_mismatches"] = [{"timestamp": 1.0, "reason": "vocal during breath", "severity": 0.9}]
dummy_dsp["breathing_coherence"] = 0.2
res_spoof = classifier.analyze_audio(dummy_dsp)
print("Spoof probability for fake audio:", res_spoof["overall_spoof_probability"])
print("Is deepfake classification for fake audio:", res_spoof["is_deepfake"])

if res_spoof["overall_spoof_probability"] > res["overall_spoof_probability"] and res_spoof["is_deepfake"] == True:
    print("\n[SUCCESS] Pipeline and model ensemble work as expected!")
else:
    print("\n[ERROR] Pipeline behavior mismatch.")
    sys.exit(1)
