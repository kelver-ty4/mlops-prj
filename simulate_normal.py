"""
Simulate normal production traffic matching training distribution.
Run for a while to bring drift metrics back to healthy.
"""
import requests
import pandas as pd
import random
import time

URL = "http://localhost:8000/predict"

# Load training data and build realistic request pool
df = pd.read_csv("data/dataset.csv")
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

# Weight by tag frequency so requests match training proportions
tag_weights = df["tag"].value_counts(normalize=True).to_dict()
tags = df["tag"].tolist()
texts = df["text"].tolist()

def generate_payload():
    idx = random.choices(range(len(df)), weights=[tag_weights[t] for t in tags], k=1)[0]
    return {"text": texts[idx]}

if __name__ == "__main__":
    print(f"Sending {len(df)} distinct texts with training distribution...")
    print(f"Tag proportions: {tag_weights}")
    while True:
        try:
            payload = generate_payload()
            response = requests.post(URL, json=payload, timeout=5)
            print(f"✓ {payload['text'][:60]}... → {response.json()['label']}")
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
