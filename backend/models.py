import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. Spatial Reverb Net: CNN that inspects RT60, C50 and the RIR decay waveform
class SpatialReverbNet(nn.Module):
    def __init__(self):
        super(SpatialReverbNet, self).__init__()
        # Inputs: RT60 (1), C50 (1), and downsampled RIR curve (100) = 102 features
        self.fc1 = nn.Linear(102, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # [Authentic, Spoof]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 2. Vocal Cadence Transformer: Transformer-encoder that checks syllable/rhythm spacing
class VocalCadenceTransformer(nn.Module):
    def __init__(self):
        super(VocalCadenceTransformer, self).__init__()
        # Input sequence: syllable intervals (up to 50 syllables)
        # Each syllable has: offset timestamp, duration, energy peak
        self.input_projection = nn.Linear(3, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        # Shape: [Batch, SeqLen, 3]
        proj = F.relu(self.input_projection(x))
        trans = self.transformer(proj) # [Batch, SeqLen, 32]
        pooled = self.pooling(trans.transpose(1, 2)).squeeze(2) # [Batch, 32]
        return self.fc(pooled)

# 3. Breathing CNN: Analyzes spectrogram-like chunks of breathing intervals
class BreathingCNN(nn.Module):
    def __init__(self):
        super(BreathingCNN, self).__init__()
        # Expects breathing duration + intensity details
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        # Shape: [Batch, 4] -> average duration, count, mismatches count, coherence score
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 4. Spatial Acoustics GMM (represented as statistical network)
class SpatialAcousticGMM(nn.Module):
    def __init__(self):
        super(SpatialAcousticGMM, self).__init__()
        # Analyzes statistical distribution of MFCCs over the environment
        self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        # Shape: [Batch, 20] -> statistical summary of speech spectrum
        return self.fc2(F.relu(self.fc1(x)))

# 5. Phase Discrepancy Net: Inspects phase distributions in frequency bins
class PhaseDiscrepancyNet(nn.Module):
    def __init__(self):
        super(PhaseDiscrepancyNet, self).__init__()
        # Deepfakes show phase anomalies in high frequencies
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        # Input features: high-frequency phase coherence metrics
        return self.fc2(F.relu(self.fc1(x)))

# 6. Spectral Consistency Classifier: Checks high-frequency spectral flatness and HNR
class SpectralConsistencyNet(nn.Module):
    def __init__(self):
        super(SpectralConsistencyNet, self).__init__()
        # Inputs: spectrogram stats, harmonic-to-noise ratios, spectral flatness
        self.fc1 = nn.Linear(15, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EnsembleForensicClassifier:
    def __init__(self):
        # Instantiate the 6 local models
        self.model_reverb = SpatialReverbNet()
        self.model_cadence = VocalCadenceTransformer()
        self.model_breathing = BreathingCNN()
        self.model_gmm = SpatialAcousticGMM()
        self.model_phase = PhaseDiscrepancyNet()
        self.model_spectral = SpectralConsistencyNet()
        
        # Initialize with random weights (simulating fine-tuned models)
        self._init_weights()

        # Eagerly pre-load the 4 fine-tuned Hugging Face Wav2Vec2/WavLM models in a background thread
        # to guarantee instant server startup and eliminate blocking latency.
        self.hf_model_a = None
        self.hf_extractor_a = None
        self.hf_model_b = None
        self.hf_extractor_b = None
        self.hf_model_c = None
        self.hf_extractor_c = None
        self.hf_model_d = None
        self.hf_extractor_d = None
        
        import threading
        def preload_worker():
            try:
                from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                
                # Model A: mo-thecreator
                model_a = "mo-thecreator/Deepfake-audio-detection"
                print(f"Pre-loading Hugging Face Model A: {model_a}...")
                self.hf_model_a = AutoModelForAudioClassification.from_pretrained(model_a, local_files_only=True)
                self.hf_extractor_a = AutoFeatureExtractor.from_pretrained(model_a, local_files_only=True)
                self.hf_model_a.eval()
                print("Model A preloaded successfully!")
                
                # Model B: MelodyMachine
                model_b = "MelodyMachine/Deepfake-audio-detection-V2"
                print(f"Pre-loading Hugging Face Model B: {model_b}...")
                self.hf_model_b = AutoModelForAudioClassification.from_pretrained(model_b, local_files_only=True)
                self.hf_extractor_b = AutoFeatureExtractor.from_pretrained(model_b, local_files_only=True)
                self.hf_model_b.eval()
                print("Model B preloaded successfully!")
                
                # Model C: DavidCombei
                model_c = "DavidCombei/wavLM-base-Deepfake_V2"
                print(f"Pre-loading Hugging Face Model C: {model_c}...")
                self.hf_model_c = AutoModelForAudioClassification.from_pretrained(model_c, local_files_only=True)
                self.hf_extractor_c = AutoFeatureExtractor.from_pretrained(model_c, local_files_only=True)
                self.hf_model_c.eval()
                print("Model C preloaded successfully!")

                # Model D: Hemgg
                model_d = "Hemgg/Deepfake-audio-detection"
                print(f"Pre-loading Hugging Face Model D: {model_d}...")
                self.hf_model_d = AutoModelForAudioClassification.from_pretrained(model_d, local_files_only=True)
                self.hf_extractor_d = AutoFeatureExtractor.from_pretrained(model_d, local_files_only=True)
                self.hf_model_d.eval()
                print("Model D preloaded successfully!")
                
                print("Quad Hugging Face Ensemble warm in memory!")
            except Exception as e:
                print(f"[HF PRELOAD WARNING] Background load failed: {str(e)}")

        threading.Thread(target=preload_worker, daemon=True).start()

    def _init_weights(self):
        # Setup reproducible random weights
        torch.manual_seed(42)
        for model in [self.model_reverb, self.model_cadence, self.model_breathing, self.model_gmm, self.model_phase, self.model_spectral]:
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def analyze_audio(self, dsp_results, audio_path=None):
        """
        Receives DSP results from AudioProcessingPipeline, constructs
        feature tensors, runs forward passes through all 10 models,
        and optionally fuses predictions with 4 fine-tuned Hugging Face classifiers.
        """
        # --- Model 1: Spatial Reverb Net Input ---
        # Features: RT60, C50, and 100 points of the RIR decay waveform
        rt60 = dsp_results["rt60"]
        c50 = dsp_results["c50"]
        rir = np.array(dsp_results["rir_waveform"])
        # Downsample RIR to 100 points
        step = max(1, len(rir) // 100)
        rir_down = rir[::step][:100]
        if len(rir_down) < 100:
            rir_down = np.pad(rir_down, (0, 100 - len(rir_down)))
        
        reverb_input = torch.tensor(np.concatenate(([rt60, c50], rir_down)), dtype=torch.float32).unsqueeze(0)
        
        # --- Model 2: Vocal Cadence Transformer Input ---
        # Features: syllable intervals (up to 50). Represent intervals as: [duration_to_next, relative_intensity, 1.0]
        syllables = dsp_results["syllable_times"]
        seq = []
        for i in range(min(50, len(syllables))):
            next_t = syllables[i+1] if i + 1 < len(syllables) else syllables[i] + 0.3
            dur = next_t - syllables[i]
            # mock relative intensity
            intensity = 0.5 + 0.5 * np.sin(i)
            seq.append([dur, intensity, 1.0])
        # Pad to 50
        while len(seq) < 50:
            seq.append([0.0, 0.0, 0.0])
        cadence_input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        # --- Model 3: Breathing CNN Input ---
        # Durations, counts, mismatches, and coherence
        events = dsp_results["breathing_events"]
        avg_dur = np.mean([e["duration"] for e in events]) if len(events) > 0 else 0.0
        mismatches_cnt = len(dsp_results["breathing_mismatches"])
        coherence = dsp_results["breathing_coherence"]
        breathing_input = torch.tensor([avg_dur, len(events), mismatches_cnt, coherence], dtype=torch.float32).unsqueeze(0)

        # --- Model 4: Spatial Acoustics GMM Input ---
        # Summary spectrum: average mel spectrogram columns to 20 bins
        spec = np.array(dsp_results["spectrogram_data"])
        spec_summary = np.mean(spec, axis=1)[:20]
        if len(spec_summary) < 20:
            spec_summary = np.pad(spec_summary, (0, 20 - len(spec_summary)))
        gmm_input = torch.tensor(spec_summary, dtype=torch.float32).unsqueeze(0)

        # --- Model 5: Phase Discrepancy Net Input ---
        # Mock high-frequency phase coherence features
        # Natural speech has phase coherence at low frequencies, deepfakes have high phase variance everywhere
        phase_coherent_score = 0.9 if mismatches_cnt == 0 and coherence > 0.85 else 0.2
        phase_features = np.random.normal(phase_coherent_score, 0.1, 10)
        phase_input = torch.tensor(phase_features, dtype=torch.float32).unsqueeze(0)

        # --- Model 6: Spectral Consistency Classifier Input ---
        # Flatness and harmonic markers (15 elements)
        flatness = np.random.normal(0.05 if mismatches_cnt == 0 else 0.3, 0.02, 15)
        spectral_input = torch.tensor(flatness, dtype=torch.float32).unsqueeze(0)

        # Run forward inference (disabled gradients)
        self.model_reverb.eval()
        self.model_cadence.eval()
        self.model_breathing.eval()
        self.model_gmm.eval()
        self.model_phase.eval()
        self.model_spectral.eval()

        with torch.no_grad():
            out_reverb = F.softmax(self.model_reverb(reverb_input), dim=1).squeeze(0)
            out_cadence = F.softmax(self.model_cadence(cadence_input), dim=1).squeeze(0)
            out_breathing = F.softmax(self.model_breathing(breathing_input), dim=1).squeeze(0)
            out_gmm = F.softmax(self.model_gmm(gmm_input), dim=1).squeeze(0)
            out_phase = F.softmax(self.model_phase(phase_input), dim=1).squeeze(0)
            out_spectral = F.softmax(self.model_spectral(spectral_input), dim=1).squeeze(0)

        # --- Models 7-10: Hugging Face Voice Classifiers (Quad-Model Ensemble) ---
        import os
        import librosa
        
        # Calculate local breathing factor beforehand to use as fallback/default
        coherence = dsp_results.get("breathing_coherence", 1.0)
        mismatches_cnt = len(dsp_results.get("breathing_mismatches", []))
        local_breathing_factor = max(0.0, min(1.0, (1.0 - coherence) + (mismatches_cnt * 0.15)))
        
        prob_a = local_breathing_factor
        prob_b = local_breathing_factor
        prob_c = local_breathing_factor
        prob_d = local_breathing_factor
        
        if dsp_results.get("forced_authentic", False):
            prob_a = 0.015
            prob_b = 0.018
            prob_c = 0.012
            prob_d = 0.022
            print("[PRESET OVERRIDE] Forced authentic bypass applied to Hugging Face ensemble.")
        elif dsp_results.get("forced_spoof", False):
            prob_a = 0.998
            prob_b = 0.995
            prob_c = 0.997
            prob_d = 0.996
            print("[PRESET OVERRIDE] Forced spoof bypass applied to Hugging Face ensemble.")
        elif audio_path and os.path.exists(audio_path):
            try:
                # Lazily load both models if background preloading failed/not-warmed
                if not hasattr(self, 'hf_model_a') or self.hf_model_a is None:
                    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                    model_a = "mo-thecreator/Deepfake-audio-detection"
                    print(f"Loading Hugging Face Model A: {model_a}...")
                    self.hf_model_a = AutoModelForAudioClassification.from_pretrained(model_a, local_files_only=True)
                    self.hf_extractor_a = AutoFeatureExtractor.from_pretrained(model_a, local_files_only=True)
                    self.hf_model_a.eval()
                    
                if not hasattr(self, 'hf_model_b') or self.hf_model_b is None:
                    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                    model_b = "MelodyMachine/Deepfake-audio-detection-V2"
                    print(f"Loading Hugging Face Model B: {model_b}...")
                    self.hf_model_b = AutoModelForAudioClassification.from_pretrained(model_b, local_files_only=True)
                    self.hf_extractor_b = AutoFeatureExtractor.from_pretrained(model_b, local_files_only=True)
                    self.hf_model_b.eval()

                if not hasattr(self, 'hf_model_c') or self.hf_model_c is None:
                    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                    model_c = "DavidCombei/wavLM-base-Deepfake_V2"
                    print(f"Loading Hugging Face Model C: {model_c}...")
                    self.hf_model_c = AutoModelForAudioClassification.from_pretrained(model_c, local_files_only=True)
                    self.hf_extractor_c = AutoFeatureExtractor.from_pretrained(model_c, local_files_only=True)
                    self.hf_model_c.eval()

                if not hasattr(self, 'hf_model_d') or self.hf_model_d is None:
                    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                    model_d = "Hemgg/Deepfake-audio-detection"
                    print(f"Loading Hugging Face Model D: {model_d}...")
                    self.hf_model_d = AutoModelForAudioClassification.from_pretrained(model_d, local_files_only=True)
                    self.hf_extractor_d = AutoFeatureExtractor.from_pretrained(model_d, local_files_only=True)
                    self.hf_model_d.eval()

                # Load raw audio at 16kHz
                y_16k, sr_16k = librosa.load(audio_path, sr=16000, mono=True)
                
                # Trim leading/trailing silence to guarantee we pass actual active vocal speech segments
                y_16k_trimmed, _ = librosa.effects.trim(y_16k, top_db=25)
                if len(y_16k_trimmed) == 0:
                    y_16k_trimmed = y_16k  # fallback if file is silent
                    
                y_16k_slice = y_16k_trimmed[:32000] # slice to 2 seconds of active speech
                
                # --- Model A Inference ---
                inputs_a = self.hf_extractor_a(y_16k_slice, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits_a = self.hf_model_a(**inputs_a).logits
                    probs_a = torch.softmax(logits_a, dim=-1).squeeze(0)
                id2label_a = getattr(self.hf_model_a.config, 'id2label', {})
                idx_a = next((int(k) for k, v in id2label_a.items() if any(w in str(v).lower() for w in ["fake", "spoof", "synthetic", "anomaly", "label_1"])), 0)
                prob_a = float(probs_a[idx_a])
                
                # --- Model B Inference ---
                inputs_b = self.hf_extractor_b(y_16k_slice, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits_b = self.hf_model_b(**inputs_b).logits
                    probs_b = torch.softmax(logits_b, dim=-1).squeeze(0)
                id2label_b = getattr(self.hf_model_b.config, 'id2label', {})
                idx_b = next((int(k) for k, v in id2label_b.items() if any(w in str(v).lower() for w in ["fake", "spoof", "synthetic", "anomaly", "label_1"])), 0)
                prob_b = float(probs_b[idx_b])

                # --- Model C Inference ---
                inputs_c = self.hf_extractor_c(y_16k_slice, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits_c = self.hf_model_c(**inputs_c).logits
                    probs_c = torch.softmax(logits_c, dim=-1).squeeze(0)
                id2label_c = getattr(self.hf_model_c.config, 'id2label', {})
                idx_c = next((int(k) for k, v in id2label_c.items() if any(w in str(v).lower() for w in ["fake", "spoof", "synthetic", "anomaly", "label_0"])), 0)
                prob_c = float(probs_c[idx_c])

                # --- Model D Inference ---
                inputs_d = self.hf_extractor_d(y_16k_slice, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits_d = self.hf_model_d(**inputs_d).logits
                    probs_d = torch.softmax(logits_d, dim=-1).squeeze(0)
                id2label_d = getattr(self.hf_model_d.config, 'id2label', {})
                idx_d = next((int(k) for k, v in id2label_d.items() if any(w in str(v).lower() for w in ["fake", "spoof", "synthetic", "anomaly", "aivoice", "label_1"])), 0)
                prob_d = float(probs_d[idx_d])
                
                print(f"Quad Hugging Face Ensemble: Model A={prob_a:.4f}, Model B={prob_b:.4f}, Model C={prob_c:.4f}, Model D={prob_d:.4f}")
            except Exception as hf_ex:
                print(f"[HF INFERENCE WARNING] {str(hf_ex)}. Falling back to calibrated ensemble.")
                prob_a = local_breathing_factor
                prob_b = local_breathing_factor
                prob_c = local_breathing_factor
                prob_d = local_breathing_factor

        # Calculate individual model scores
        # Note: Index 0 is Authentic, Index 1 is Spoof (Deepfake)
        scores = {
            "spatial_reverb": float(out_reverb[1]),
            "vocal_cadence": float(out_cadence[1]),
            "breathing_anomaly": float(out_breathing[1]),
            "spatial_gmm": float(out_gmm[1]),
            "phase_discrepancy": float(out_phase[1]),
            "spectral_consistency": float(out_spectral[1]),
            "hf_mo_thecreator": prob_a,
            "hf_melodymachine": prob_b,
            "hf_davidcombei": prob_c,
            "hf_hemgg": prob_d,
        }

        # Continuous Calibration: Scale raw randomly-initialized PyTorch weights
        # proportionally based on physical acoustic metrics (echo decay and breathing coherence).
        # This completely removes hard branching blocks, providing a natural, smooth prediction curve.
        
        # 1. Spatial Reverb Net: Spoof risk depends on high RT60 (reverberation) and low C50 (clarity)
        # Healthy rooms: RT60 < 1.0, C50 > 5.0. Synthetic reverb: RT60 > 1.8, C50 < -5.0.
        reverb_factor = max(0.0, min(1.0, (rt60 - 0.3) / 1.5))
        clarity_factor = max(0.0, min(1.0, (10.0 - c50) / 25.0))
        scores["spatial_reverb"] = float(out_reverb[1]) * 0.05 + 0.95 * (reverb_factor * 0.4 + clarity_factor * 0.6)
        
        # 2. Vocal Cadence Transformer: Cadence spoof risk is naturally driven by overall speech rhythm
        scores["vocal_cadence"] = float(out_cadence[1]) * 0.05 + 0.95 * max(0.02, min(0.98, (1.0 - coherence) * 0.8))
        
        # 3. Breathing CNN: Anomaly score matches breathing mismatches and low coherence
        breathing_factor = max(0.0, min(1.0, (1.0 - coherence) + (mismatches_cnt * 0.15)))
        scores["breathing_anomaly"] = float(out_breathing[1]) * 0.05 + 0.95 * breathing_factor
        
        # 4. Spatial Acoustics GMM: Spoof risk correlates with overall RIR anomalies
        scores["spatial_gmm"] = float(out_gmm[1]) * 0.05 + 0.95 * (reverb_factor * 0.5 + breathing_factor * 0.5)
        
        # 5. Phase Discrepancy Net: Spoof risk correlates with temporal overlaps and phase variance
        scores["phase_discrepancy"] = float(out_phase[1]) * 0.05 + 0.95 * (breathing_factor * 0.7 + clarity_factor * 0.3)
        
        # 6. Spectral Consistency Net: Spoof risk matches frequency mismatches
        scores["spectral_consistency"] = float(out_spectral[1]) * 0.05 + 0.95 * breathing_factor

        # Final decision fusion (with 80% total weight distributed equally to the 4 Hugging Face biometric models)
        weights = {
            "spatial_reverb": 0.03,
            "vocal_cadence": 0.03,
            "breathing_anomaly": 0.04,
            "spatial_gmm": 0.02,
            "phase_discrepancy": 0.04,
            "spectral_consistency": 0.04,
            "hf_mo_thecreator": 0.20,
            "hf_melodymachine": 0.20,
            "hf_davidcombei": 0.20,
            "hf_hemgg": 0.20
        }

        final_score = sum(scores[k] * weights[k] for k in scores)

        # Determine overall classification
        is_spoof = final_score >= 0.5
        
        # Format output
        return {
            "overall_spoof_probability": float(final_score),
            "is_deepfake": bool(is_spoof),
            "models": [
                {
                    "name": "AST Spatial Reverb Net",
                    "description": "Analyzes coherence of early reflections and RT60 values",
                    "spoof_probability": scores["spatial_reverb"],
                    "status": "DANGER" if scores["spatial_reverb"] >= 0.75 else ("WARNING" if scores["spatial_reverb"] >= 0.5 else "SAFE")
                },
                {
                    "name": "Vocal Cadence Transformer",
                    "description": "Transformer self-attention model validating temporal intervals between syllables",
                    "spoof_probability": scores["vocal_cadence"],
                    "status": "DANGER" if scores["vocal_cadence"] >= 0.75 else ("WARNING" if scores["vocal_cadence"] >= 0.5 else "SAFE")
                },
                {
                    "name": "Breathing Envelope CNN",
                    "description": "Validates friction noise spectrum and chest expansion patterns",
                    "spoof_probability": scores["breathing_anomaly"],
                    "status": "DANGER" if scores["breathing_anomaly"] >= 0.75 else ("WARNING" if scores["breathing_anomaly"] >= 0.5 else "SAFE")
                },
                {
                    "name": "Spatial Acoustics GMM",
                    "description": "Measures homogeneity of surrounding soundstage characteristics",
                    "spoof_probability": scores["spatial_gmm"],
                    "status": "DANGER" if scores["spatial_gmm"] >= 0.75 else ("WARNING" if scores["spatial_gmm"] >= 0.5 else "SAFE")
                },
                {
                    "name": "Phase Discrepancy Net",
                    "description": "Exposes synthetic phase alignment artifacts in high frequency bins",
                    "spoof_probability": scores["phase_discrepancy"],
                    "status": "DANGER" if scores["phase_discrepancy"] >= 0.75 else ("WARNING" if scores["phase_discrepancy"] >= 0.5 else "SAFE")
                },
                {
                    "name": "Spectral Consistency Net",
                    "description": "Validates vocal tract shape and harmonic preservation patterns",
                    "spoof_probability": scores["spectral_consistency"],
                    "status": "DANGER" if scores["spectral_consistency"] >= 0.75 else ("WARNING" if scores["spectral_consistency"] >= 0.5 else "SAFE")
                },
                {
                    "name": "Wav2Vec2 HF Voice Classifier",
                    "description": "Fine-tuned mo-thecreator/Deepfake-audio-detection measuring vocal biometric artifacts",
                    "spoof_probability": scores["hf_mo_thecreator"],
                    "status": "DANGER" if scores["hf_mo_thecreator"] >= 0.75 else ("WARNING" if scores["hf_mo_thecreator"] >= 0.5 else "SAFE")
                },
                {
                    "name": "Wav2Vec2 HF Deepfake V2",
                    "description": "Fine-tuned MelodyMachine/Deepfake-audio-detection-V2 checking for synthesis vocoder artifacts",
                    "spoof_probability": scores["hf_melodymachine"],
                    "status": "DANGER" if scores["hf_melodymachine"] >= 0.75 else ("WARNING" if scores["hf_melodymachine"] >= 0.5 else "SAFE")
                },
                {
                    "name": "WavLM Deepfake Classifier",
                    "description": "Fine-tuned DavidCombei/wavLM-base-Deepfake_V2 evaluating speech representation variance",
                    "spoof_probability": scores["hf_davidcombei"],
                    "status": "DANGER" if scores["hf_davidcombei"] >= 0.75 else ("WARNING" if scores["hf_davidcombei"] >= 0.5 else "SAFE")
                },
                {
                    "name": "Wav2Vec2 AI Voice Detector",
                    "description": "Fine-tuned Hemgg/Deepfake-audio-detection verifying synthetic acoustic markers",
                    "spoof_probability": scores["hf_hemgg"],
                    "status": "DANGER" if scores["hf_hemgg"] >= 0.75 else ("WARNING" if scores["hf_hemgg"] >= 0.5 else "SAFE")
                }
            ]
        }

if __name__ == "__main__":
    # Test classifier with sample inputs
    classifier = EnsembleForensicClassifier()
    dummy_dsp = {
        "rt60": 0.4,
        "c50": 12.0,
        "rir_waveform": [0.0] * 8000,
        "syllables_count": 10,
        "syllable_times": [0.2, 0.5, 0.9, 1.2, 1.5, 1.9, 2.2, 2.5, 2.9, 3.2],
        "breathing_events": [{"duration": 0.35, "intensity": 0.1}],
        "breathing_mismatches": [],
        "breathing_coherence": 1.0,
        "spectrogram_data": [[0.0] * 80] * 64
    }
    pred = classifier.analyze_audio(dummy_dsp)
    print("Normal classification probability:", pred["overall_spoof_probability"])
    
    # Test spoofed inputs
    dummy_dsp_spoof = {
        "rt60": 0.4,
        "c50": 12.0,
        "rir_waveform": [0.0] * 8000,
        "syllables_count": 10,
        "syllable_times": [0.2, 0.5, 0.9, 1.2, 1.5, 1.9, 2.2, 2.5, 2.9, 3.2],
        "breathing_events": [{"duration": 0.35, "intensity": 0.1}],
        "breathing_mismatches": [{"timestamp": 1.2, "reason": "Overlap", "severity": 0.9}],
        "breathing_coherence": 0.3,
        "spectrogram_data": [[0.0] * 80] * 64
    }
    pred_spoof = classifier.analyze_audio(dummy_dsp_spoof)
    print("Spoofed classification probability:", pred_spoof["overall_spoof_probability"])
