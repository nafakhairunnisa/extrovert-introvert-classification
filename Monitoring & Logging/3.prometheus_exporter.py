from flask import Flask, request, jsonify, Response
import requests
import time
import psutil 
import random
import threading
from prometheus_client import (Counter, Histogram, Gauge, 
                             generate_latest, CONTENT_TYPE_LATEST)

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency', ['endpoint'])
THROUGHPUT = Counter('http_throughput_total', 'Total number of requests')
ERROR_COUNT = Counter('http_errors_total', 'Total Errors', ['error_type'])

# Metrik untuk sistem
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU Usage Percentage')
GPU_USAGE = Gauge('gpu_usage_percent', 'GPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')
MODEL_PERFORMANCE = Gauge('model_prediction_value', 'Prediction Value')
ERROR_RATE = Gauge('model_error_rate', 'Error rate of predictions')
MODEL_MEMORY = Gauge('model_memory_mb', 'Memory usage of model')
REQUEST_DURATION = Gauge('request_duration_seconds', 'Duration of requests')
MODEL_LOAD = Gauge('model_load', 'Model load factor')
PREDICTION_TIME = Gauge('model_prediction_time', 'Time per prediction')

# Background thread untuk update metrik sistem
def update_system_metrics():
    while True:
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        RAM_USAGE.set(psutil.virtual_memory().percent)
        GPU_USAGE.set(random.uniform(0, 100))
        time.sleep(5)

# Simulasi request agar metrik model selalu terisi
def simulate_requests():
    while True:
        latency = random.uniform(0.1, 0.5)
        prediction = random.uniform(0, 1)

        REQUEST_LATENCY.labels(endpoint='/simulate').observe(latency)
        REQUEST_DURATION.set(random.uniform(0.05, 0.3))
        MODEL_PERFORMANCE.set(prediction)
        ERROR_RATE.set(random.uniform(0, 0.1))
        MODEL_MEMORY.set(random.uniform(100, 200))
        MODEL_LOAD.set(random.uniform(0.1, 0.9))
        PREDICTION_TIME.set(random.uniform(0.01, 0.2))
        THROUGHPUT.inc()
        
        time.sleep(5)

@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='started').inc()
    
    try:
        data = request.get_json()
        if not data:
            raise ValueError("Empty request body")
            
        # Forward ke model service
        response = requests.post(
            "http://localhost:5001/invocations",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        
        prediction = response.json()
        duration = time.time() - start_time
        
        # Update metrik
        REQUEST_LATENCY.labels(endpoint='/predict').observe(duration)
        REQUEST_DURATION.set(duration)
        PREDICTION_TIME.set(duration)
        MODEL_PERFORMANCE.set(float(prediction[0]))
        MODEL_MEMORY.set(random.uniform(100, 200))
        MODEL_LOAD.set(random.uniform(0.1, 0.9))
        ERROR_RATE.set(random.uniform(0, 0.05))
        THROUGHPUT.inc()
        
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='success').inc()
        
        return jsonify({
            "prediction": prediction,
            "latency": duration,
            "status": "success"
        })

    except requests.exceptions.Timeout as e:
        ERROR_COUNT.labels(error_type='timeout').inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='timeout').inc()
        ERROR_RATE.inc()
        return jsonify({"error": "Request timeout"}), 504
        
    except Exception as e:
        error_type = type(e).__name__
        ERROR_COUNT.labels(error_type=error_type).inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='error').inc()
        ERROR_RATE.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start system metrics thread
    threading.Thread(target=update_system_metrics, daemon=True).start()
    
    # Start simulation thread
    threading.Thread(target=simulate_requests, daemon=True).start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8001, threaded=True)