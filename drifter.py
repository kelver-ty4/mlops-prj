import requests
import random
import time

URL = "http://localhost:8000/predict"

# Example payload generator

def generate_payload():
    drift_texts = [
        "🚀" * 500,
        "A" * 1000,
        "هادشي ماخدامش نهائيا",
        "1234567890" * 100,
        "@@@@@#####$$$$$%%%%%",
        "NULL NULL NULL NULL",
    ]

    return {
        "text": random.choice(drift_texts)
    }


# Continuous requests
while True:
    try:
        payload = generate_payload()

        response = requests.post(URL, json=payload)

        print(f"Sent: {payload}")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        print("-" * 50)

        # Small delay to avoid overwhelming localhost
        time.sleep(0.2)

    except Exception as e:
        print("Error:", e)
        time.sleep(1)
