import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import librosa

class AudioProcessingPipeline:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def load_audio(self, file_path):
        """Loads and normalizes audio file to target sample rate."""
        y, sr = librosa.load(file_path, sr=self.target_sr)
        # Normalize
        if len(y) > 0:
            y = y / (np.max(np.abs(y)) + 1e-8)
        return y, sr

    def extract_rir_features(self, y, sr):
        """
        Blind estimation of Room Impulse Response (RIR) characteristics.
        Calculates RT60 (reverberation decay), Clarity (C50), and simulates
        the RIR envelope.
        """
        # 1. Calculate energy envelope (RMS)
        frame_length = int(0.02 * sr)  # 20ms frame
        hop_length = int(0.01 * sr)   # 10ms hop
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = 20 * np.log10(rms + 1e-5)
        
        # 2. Estimate RT60 by analyzing energy decay during pauses
        # Find local maxima followed by steep decays in energy
        decays = []
        for i in range(1, len(rms_db) - 50):
            # Check if there is an energy drop starting at i
            if rms_db[i] > -20 and rms_db[i] > rms_db[i-1]:  # local peak
                start_val = rms_db[i]
                # Look ahead for a decay
                decay_len = 0
                for j in range(i+1, min(i+50, len(rms_db))):
                    if rms_db[j] < start_val:
                        decay_len = j - i
                    else:
                        break
                if decay_len > 10:  # Valid decay period
                    drop = start_val - rms_db[i+decay_len]
                    if drop > 10:  # Significant drop (at least 10dB)
                        # Estimate seconds per dB drop
                        sec_per_db = (decay_len * 0.01) / drop
                        decays.append(sec_per_db * 60)  # Extrapolate to 60dB decay (RT60)
        
        rt60 = float(np.median(decays)) if len(decays) > 0 else 0.35  # default dry room
        # Keep RT60 within realistic bounds [0.1s, 3.0s]
        rt60 = max(0.1, min(3.0, rt60))
        
        # 3. Calculate C50 (Clarity): ratio of energy in early 50ms to late (>50ms) reflections.
        # Since we don't have the true RIR, we approximate it by correlation analysis of the signal.
        # We can analyze the autocorrelation to find echoes/reflections.
        autocorr = librosa.autocorrelate(y, max_size=int(0.2 * sr))
        autocorr = autocorr / (np.max(autocorr) + 1e-8)
        
        # 50ms in samples
        samples_50ms = int(0.05 * sr)
        early_energy = np.sum(autocorr[:samples_50ms]**2)
        late_energy = np.sum(autocorr[samples_50ms:]**2)
        
        c50 = float(10 * np.log10(early_energy / (late_energy + 1e-8)))
        # Bound C50 between -15dB and +30dB
        c50 = max(-15.0, min(30.0, c50))

        # 4. Generate a simulated RIR curve matching the estimated RT60 and C50
        # RIR consists of a direct path impulse, followed by sparse early reflections,
        # followed by exponential noise decay (late reverberation).
        rir_len = int(rt60 * sr)
        t = np.arange(rir_len) / sr
        
        # Exponential decay factor based on RT60 (energy decays by 60dB, i.e., amplitude decays by 10^-3)
        decay_constant = 3 * np.log(10) / rt60
        envelope = np.exp(-decay_constant * t)
        
        # Add white noise shaped by the decay envelope
        noise = np.random.randn(rir_len)
        rir_waveform = noise * envelope
        
        # Inject early reflections (spikes in the first 50ms)
        num_reflections = 8
        for _ in range(num_reflections):
            ref_idx = int(np.random.uniform(0.005, 0.05) * sr)
            ref_amp = np.random.uniform(0.1, 0.5) * envelope[ref_idx]
            if ref_idx < rir_len:
                rir_waveform[ref_idx] += ref_amp * np.sign(np.random.randn())
                
        # Inject direct path at t=0
        rir_waveform[0] = 1.0
        
        # Normalize RIR
        rir_waveform = rir_waveform / (np.max(np.abs(rir_waveform)) + 1e-8)
        
        return {
            "rt60": rt60,
            "c50": c50,
            "rir_waveform": rir_waveform.tolist()[:int(0.5 * sr)]  # send first 500ms
        }

    def detect_breathing_patterns(self, y, sr):
        """
        Detects breathing pauses.
        Breathing is characterized by low energy in speech frequencies,
        relative high frequency energy (in-breath friction), and duration between 200ms-800ms.
        """
        # Bandpass filter the audio in the breathing friction band (500 Hz - 3000 Hz)
        nyq = 0.5 * sr
        b, a = signal.butter(4, [500/nyq, 3000/nyq], btype='band')
        y_filt = signal.filtfilt(b, a, y)
        
        # Calculate amplitude envelopes
        frame_len = int(0.02 * sr)
        hop_len = int(0.01 * sr)
        
        # High frequency energy envelope
        hf_energy = librosa.feature.rms(y=y_filt, frame_length=frame_len, hop_length=hop_len)[0]
        # Total energy envelope
        total_energy = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        
        # Normalization
        hf_energy = hf_energy / (np.max(hf_energy) + 1e-8)
        total_energy = total_energy / (np.max(total_energy) + 1e-8)
        
        # Speech onset detection (to compare breathing pauses with syllable boundaries)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_len)
        peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_len)
        syllable_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_len).tolist()
        
        # Find candidates for breathing: pauses where total energy is low but high-frequency band remains active
        breathing_events = []
        is_breathing = False
        start_frame = 0
        
        # Thresholds
        silence_threshold = 0.15 # Max total energy to count as silent speech break
        hf_friction_threshold = 0.08 # Min filtered high-freq energy to count as breathing friction
        
        for f in range(len(total_energy)):
            if total_energy[f] < silence_threshold and hf_energy[f] > hf_friction_threshold:
                if not is_breathing:
                    is_breathing = True
                    start_frame = f
            else:
                if is_breathing:
                    is_breathing = False
                    end_frame = f
                    duration = (end_frame - start_frame) * 0.01  # hop size is 10ms
                    # Breathing usually lasts between 150ms and 1000ms
                    if 0.15 <= duration <= 1.0:
                        start_time = start_frame * 0.01
                        end_time = end_frame * 0.01
                        breathing_events.append({
                            "start": start_time,
                            "end": end_time,
                            "duration": duration,
                            "intensity": float(np.mean(hf_energy[start_frame:end_frame]))
                        })
                        
        # Evaluate alignment coherence between breathing and spoken syllables
        # Natural speech has breathing occurring strictly in major pauses (i.e. not in the middle of onsets)
        # Deepfakes often overlay speech directly over breathing or miss breathing entirely
        breathing_mismatches = []
        coherence_score = 1.0
        
        for breath in breathing_events:
            # Check if any syllable peak occurs inside the breathing window
            in_breath_syllables = [t for t in syllable_times if breath["start"] < t < breath["end"]]
            if len(in_breath_syllables) > 0:
                breathing_mismatches.append({
                    "timestamp": breath["start"],
                    "reason": "Vocalization during inhalation (breathing-speech overlap)",
                    "severity": 0.95
                })
                coherence_score -= 0.15

        # Check for natural frequency: typical speech has 4-12 breath events per minute
        clip_duration = len(y) / sr
        breaths_per_minute = len(breathing_events) * (60.0 / clip_duration) if clip_duration > 0 else 0
        
        if breaths_per_minute > 25:
            breathing_mismatches.append({
                "timestamp": clip_duration / 2.0,
                "reason": "Hyper-frequent respiration pattern (unnatural cadence)",
                "severity": 0.65
            })
            coherence_score -= 0.2
        elif breaths_per_minute < 2 and clip_duration > 15:
            breathing_mismatches.append({
                "timestamp": clip_duration / 2.0,
                "reason": "Unnaturally long speech duration without breathing",
                "severity": 0.75
            })
            coherence_score -= 0.25

        coherence_score = max(0.0, min(1.0, coherence_score))

        return {
            "events": breathing_events,
            "mismatches": breathing_mismatches,
            "coherence_score": coherence_score,
            "syllables_count": len(syllable_times),
            "syllable_times": syllable_times
        }

    def process_audio(self, file_path):
        """Runs the entire extraction pipeline on an audio file."""
        y, sr = self.load_audio(file_path)
        
        # 1. Basic properties
        duration = len(y) / sr
        
        # 2. Extract RIR
        rir_info = self.extract_rir_features(y, sr)
        
        # 3. Detect breathing
        breathing_info = self.detect_breathing_patterns(y, sr)
        
        # 4. Generate downsampled waveform for visualization (e.g. 200 points)
        step = max(1, len(y) // 200)
        waveform_data = y[::step].tolist()
        
        # 5. Extract mel spectrogram for neural networks
        # We can extract a 64-band mel spectrogram and downsample it for display
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=int(0.05*sr))
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize to [0, 1] range for visual presentation
        mel_spec_normalized = ((mel_spec_db + 80) / 80).clip(0, 1)
        
        # Downsample time steps for rendering
        t_steps = 80
        step_time = max(1, mel_spec_normalized.shape[1] // t_steps)
        spectrogram_data = mel_spec_normalized[:, ::step_time].tolist()

        return {
            "duration": duration,
            "sample_rate": sr,
            "rt60": rir_info["rt60"],
            "c50": rir_info["c50"],
            "rir_waveform": rir_info["rir_waveform"],
            "breathing_events": breathing_info["events"],
            "breathing_mismatches": breathing_info["mismatches"],
            "breathing_coherence": breathing_info["coherence_score"],
            "syllables_count": breathing_info["syllables_count"],
            "syllable_times": breathing_info["syllable_times"],
            "waveform_data": waveform_data,
            "spectrogram_data": spectrogram_data
        }

if __name__ == "__main__":
    # Test pipeline on dummy noise
    pipeline = AudioProcessingPipeline()
    dummy_signal = np.random.randn(16000 * 5)  # 5 seconds of noise
    # Save test file
    wav.write("dummy_test.wav", 16000, (dummy_signal * 32767).astype(np.int16))
    res = pipeline.process_audio("dummy_test.wav")
    print("Test successful! Duration:", res["duration"], "RT60:", res["rt60"])
    os.remove("dummy_test.wav")
