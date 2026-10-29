"""
FarmSphere AI - Virtual IoT Hardware Telemetry Simulator
Simulates ESP32 node transmitting capacitive soil moisture, DHT22, pH, and NPK data
to the FarmSphere backend ingestion service.
"""

import time
import random
import json
import urllib.request

BACKEND_URL = "http://localhost:5000/api/iot/ingest"
DEVICE_ID = "ESP32_HUB_01"
API_KEY = "FS_IOT_SECURE_TOKEN_9981"

def simulate_telemetry_stream(num_cycles=5, interval_seconds=3):
    print("=" * 60)
    print("   FARMSPHERE AI: VIRTUAL ESP32 SENSOR TELEMETRY STREAM")
    print("=" * 60)
    print(f"Target Gateway: {BACKEND_URL}")
    print(f"Device ID:      {DEVICE_ID}")
    
    for cycle in range(1, num_cycles + 1):
        # Realistic sensor fluctuations
        moisture = round(random.uniform(36.0, 44.0), 1)
        temp = round(random.uniform(20.5, 23.5), 1)
        humidity = round(random.uniform(55.0, 65.0), 1)
        soil_ph = round(random.uniform(6.8, 7.3), 2)
        n = round(random.uniform(110.0, 125.0), 1)
        p = round(random.uniform(50.0, 60.0), 1)
        k = round(random.uniform(38.0, 46.0), 1)
        battery = random.randint(92, 96)
        
        payload = {
            "device_id": DEVICE_ID,
            "api_key": API_KEY,
            "telemetry": {
                "soil_moisture_pct": moisture,
                "soil_ph": soil_ph,
                "temperature_c": temp,
                "humidity_pct": humidity,
                "nitrogen": n,
                "phosphorus": p,
                "potassium": k,
                "battery_pct": battery
            }
        }
        
        print(f"\n[Packet #{cycle}] Soil Moisture: {moisture}% | Temp: {temp}°C | pH: {soil_ph} | NPK: {n}/{p}/{k}")
        
        try:
            req = urllib.request.Request(
                BACKEND_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                res_body = json.loads(resp.read().decode())
                print(f"  -> Ingestion Status: {res_body.get('status')} | ID: {res_body.get('ingested_reading_id')}")
        except Exception as e:
            print(f"  -> Transmission simulated locally (Backend offline or not running): {e}")
            
        time.sleep(interval_seconds)
        
    print("\nTelemetry stream simulation finished.")

if __name__ == '__main__':
    simulate_telemetry_stream()
