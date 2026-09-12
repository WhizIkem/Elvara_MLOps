import time
import random
import httpx
import os
from datetime import datetime, timedelta

API_URL = os.getenv("API_URL", "https://elvaramlops-production-12a3.up.railway.app/predict-risk")

def generate_random_patient(pid: int):
    now = datetime.utcnow()

    age = random.randint(18, 95)
    comorbidities = min(random.randint(0, 6), max(0, age // 15))
    vital_count = random.randint(1, 3)
    lab_count = random.randint(1, 2)

    def bounded_gauss(mean, standard_deviation, minimum, maximum):
        return max(minimum, min(maximum, random.gauss(mean, standard_deviation)))

    vitals = []
    for observation in range(vital_count):
        timestamp = now - timedelta(hours=(vital_count - observation - 1) * 3)
        vitals.append({
            "timestamp": timestamp.isoformat(),
            "heart_rate": round(bounded_gauss(82, 18, 45, 140), 1),
            "temperature": round(bounded_gauss(37.2, 1.0, 35.0, 40.5), 1),
            "oxygen_saturation": round(bounded_gauss(96, 3, 86, 100), 1),
            "respiratory_rate": round(bounded_gauss(18, 5, 8, 35), 1),
            "blood_pressure": round(bounded_gauss(115, 20, 70, 180), 1)
        })

    labs = []
    for observation in range(lab_count):
        timestamp = now - timedelta(hours=(lab_count - observation - 1) * 4)
        labs.append({
            "timestamp": timestamp.isoformat(),
            "white_cell_count": round(bounded_gauss(9, 4, 2, 25), 1),
            "crp": round(bounded_gauss(20, 35, 0, 180), 1),
            "lactate": round(bounded_gauss(1.5, 0.9, 0.4, 5.5), 2),
            "creatinine": round(bounded_gauss(1.1, 0.6, 0.4, 3.5), 2),
            "platelet_count": round(bounded_gauss(240, 70, 50, 450), 1)
        })

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