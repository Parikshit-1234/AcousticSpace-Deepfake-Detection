import requests

url = "http://127.0.0.1:8000/api/analyze"
files = {"file": ("asvspoof_deepfake_spoof.wav", open("test_audios/asvspoof_deepfake_spoof.wav", "rb"), "audio/wav")}
data = {"demo_type": "auto"}

print("Sending POST request to analyze asvspoof_deepfake_spoof.wav...")
response = requests.post(url, files=files, data=data)
print("Status Code:", response.status_code)
try:
    print("Response JSON summary:")
    res = response.json()
    print("Filename:", res["filename"])
    print("Overall Spoof Probability:", res["analysis"]["overall_spoof_probability"])
    print("Is Deepfake:", res["analysis"]["is_deepfake"])
except Exception as e:
    print("Response text:", response.text)
