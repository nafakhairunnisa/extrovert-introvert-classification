import requests
import json

# Endpoint model
endpoint = "http://localhost:5001/invocations"

# Data input
payload = {
    "dataframe_split": {
        "columns": [
            "Time_spent_Alone",
            "Stage_fear",
            "Social_event_attendance",
            "Going_outside",
            "Drained_after_socializing",
            "Friends_circle_size",
            "Post_frequency"
        ],
        "data": [[
            -0.932933857,
            0,
            0.969753942,
            -0.089633538,
            0,
            0.331197527,
            0.408089318
        ]]
    }
}

try:
    # Mengirim permintaan prediksi
    resp = requests.post(endpoint, json=payload, timeout=10)
    resp.raise_for_status()

    # Hasil
    result = {
        "status_code": resp.status_code,
        "prediction_result": resp.json()
    }

    # Menyimpan ke file JSON
    with open("inference_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Hasil berhasil disimpan di 'inference_result.json'")
except Exception as e:
    print("Terjadi kesalahan saat request:", str(e))