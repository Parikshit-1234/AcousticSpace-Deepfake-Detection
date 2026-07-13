import sys
import os

print("Initialising Hugging Face model caching...")

try:
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    
    model_name = "garystafford/wav2vec2-deepfake-voice-detector"
    print(f"Downloading model '{model_name}' to local Hugging Face cache...")
    
    # Run the downloads (this will fetch files from the network and cache them locally)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    print("\n[SUCCESS] Hugging Face model cached successfully!")
    print("The backend server can now load this model instantly from cache.")
    
except Exception as e:
    print(f"\n[ERROR] Failed to download Hugging Face model: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
