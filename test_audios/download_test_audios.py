import os
import librosa
import soundfile as sf
import numpy as np

print("Initializing test audios creation...")

output_dir = os.path.join(os.path.dirname(__file__))
os.makedirs(output_dir, exist_ok=True)

# 1. Download clean speech from LibriSpeech example using Librosa
try:
    print("Loading clean LibriSpeech example...")
    # 'libri1' is a standard speech example from LibriSpeech dataset packaged with librosa examples
    y, sr = librosa.load(librosa.example('libri1'), sr=16000)
    
    # Save as genuine example
    genuine_path = os.path.join(output_dir, "asvspoof_genuine_sample.wav")
    sf.write(genuine_path, y, sr)
    print(f"[SUCCESS] Saved genuine sample to: {genuine_path}")
    
    # Save as spoof example (applying synthetic modulation and breathing mismatch overlays)
    spoof_path = os.path.join(output_dir, "asvspoof_deepfake_spoof.wav")
    
    # 1. Apply metallic vocoder ring modulation at 150Hz to simulate synthesis vocoder artifacts
    t = np.arange(len(y)) / sr
    carrier = np.sin(2 * np.pi * 150 * t)
    y_spoof = y * (0.5 + 0.5 * carrier)
    
    # 2. Inject artificial breathing mismatch overlay (noise envelope during active speaking)
    # This will trigger the DSP breathing cadence mismatches naturally
    breath_noise = np.random.normal(0, 0.08, len(y))
    envelope = np.abs(y_spoof)
    y_spoof = y_spoof + breath_noise * envelope * 0.45
    
    # Save the synthesized deepfake file
    sf.write(spoof_path, y_spoof, sr)
    print(f"[SUCCESS] Synthesized and saved deepfake spoof sample to: {spoof_path}")
    
    print("\nInitialization complete! You now have actual speech audio files in your workspace.")
    print("You can upload these files in the browser to run predictions.")
    
except Exception as e:
    print(f"[ERROR] Failed to download or save speech example: {str(e)}")
    import traceback
    traceback.print_exc()
