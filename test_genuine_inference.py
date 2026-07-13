import requests

url = "http://127.0.0.1:8000/api/analyze"
files = {"file": ("asvspoof_genuine_sample.wav", open("test_audios/asvspoof_genuine_sample.wav", "rb"), "audio/wav")}
data = {"demo_type": "auto"}

print("Sending POST request to analyze asvspoof_genuine_sample.wav...")
response = requests.post(url, files=files, data=data)
print("Status Code:", response.status_code)
try:
    print("Response JSON:")
    import json
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print("Response text:", response.text)
