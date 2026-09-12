import time
import random
import httpx
import os
from datetime import datetime, timedelta

API_URL = os.getenv("API_URL", "https://elvaramlops-production-12a3.up.railway.app/predict-risk")

def generate_random_patient(pid: int):
    # 40% chance of generating a high/moderate deterioration risk patient
    is_high_risk = random.random() < 0.40
    now = datetime.utcnow()

    if is_high_risk:
        age = random.randint(70, 92)
        comorbidities = random.randint(2, 4)
        
        # Series of 3 vital readings showing deterioration over 6 hours
        vitals = [
            {
                "timestamp": (now - timedelta(hours=6)).isoformat(),
                "heart_rate": round(random.uniform(92.0, 102.0), 1),
                "temperature": round(random.uniform(37.8, 38.4), 1),
                "oxygen_saturation": round(random.uniform(94.0, 96.0), 1),
                "respiratory_rate": round(random.uniform(19.0, 22.0), 1),
                "blood_pressure": round(random.uniform(115.0, 125.0), 1)
            },
            {
                "timestamp": (now - timedelta(hours=3)).isoformat(),
                "heart_rate": round(random.uniform(105.0, 115.0), 1),
                "temperature": round(random.uniform(38.5, 39.1), 1),
                "oxygen_saturation": round(random.uniform(92.0, 94.0), 1),
                "respiratory_rate": round(random.uniform(22.0, 25.0), 1),
                "blood_pressure": round(random.uniform(102.0, 112.0), 1)
            },
            {
                "timestamp": now.isoformat(),
                "heart_rate": round(random.uniform(118.0, 135.0), 1),
                "temperature": round(random.uniform(39.2, 40.0), 1),
                "oxygen_saturation": round(random.uniform(88.0, 91.0), 1),
                "respiratory_rate": round(random.uniform(26.0, 32.0), 1),
                "blood_pressure": round(random.uniform(85.0, 98.0), 1)
            }
        ]
        
        labs = [
            {
                "timestamp": (now - timedelta(hours=4)).isoformat(),
                "white_cell_count": round(random.uniform(12.0, 15.0), 1),
                "crp": round(random.uniform(45.0, 75.0), 1),
                "lactate": round(random.uniform(2.1, 3.0), 2),
                "creatinine": round(random.uniform(1.4, 1.8), 2),
                "platelet_count": round(random.uniform(130.0, 170.0), 1)
            },
            {
                "timestamp": now.isoformat(),
                "white_cell_count": round(random.uniform(16.0, 24.0), 1),
                "crp": round(random.uniform(90.0, 160.0), 1),
                "lactate": round(random.uniform(3.2, 5.2), 2),
                "creatinine": round(random.uniform(2.0, 3.2), 2),
                "platelet_count": round(random.uniform(70.0, 120.0), 1)
            }
        ]
    else:
        age = random.randint(25, 65)
        comorbidities = random.randint(0, 1)
        
        vitals = [
            {
                "timestamp": now.isoformat(),
                "heart_rate": round(random.uniform(68.0, 82.0), 1),
                "temperature": round(random.uniform(36.5, 37.1), 1),
                "oxygen_saturation": round(random.uniform(97.0, 99.0), 1),
                "respiratory_rate": round(random.uniform(13.0, 16.0), 1),
                "blood_pressure": round(random.uniform(115.0, 128.0), 1)
            }
        ]
        labs = [
            {
                "timestamp": now.isoformat(),
                "white_cell_count": round(random.uniform(5.5, 8.5), 1),
                "crp": round(random.uniform(2.0, 6.0), 1),
                "lactate": round(random.uniform(0.9, 1.3), 2),
                "creatinine": round(random.uniform(0.8, 1.1), 2),
                "platelet_count": round(random.uniform(200.0, 320.0), 1)
            }
        ]

    return {
        "patient_id": pid,
        "age": age,
        "gender": random.choice(["Male", "Female"]),
        "comorbidity_count": comorbidities,
        "vitals": vitals,
        "labs": labs
    }

def main(total_requests: int = 30):
    print(f"--- Generating {total_requests} real-time sepsis prediction requests to target: {API_URL} ---")
    client = httpx.Client(timeout=15.0)
    
    counts = {"Low": 0, "Moderate": 0, "High": 0}
    for i in range(1, total_requests + 1):
        patient = generate_random_patient(pid=1000 + i)
        try:
            r = client.post(API_URL, json=patient)
            if r.status_code == 200:
                data = r.json()
                cat = data['risk_category']
                counts[cat] = counts.get(cat, 0) + 1
                print(f"[{i}/{total_requests}] Patient #{data['patient_id']} (Age {patient['age']}) -> Score: {data['sepsis_risk_score']:.4f} | Category: {cat}")
            else:
                print(f"[{i}/{total_requests}] Request failed with code {r.status_code}")
        except Exception as e:
            print(f"[{i}/{total_requests}] Error sending request: {e}")
        
        time.sleep(0.2)

    print("\n--- Simulation Summary ---")
    print(f"Low Risk: {counts.get('Low', 0)} | Moderate Risk: {counts.get('Moderate', 0)} | High Risk: {counts.get('High', 0)}")

if __name__ == "__main__":
    main()