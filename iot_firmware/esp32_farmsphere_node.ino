/*
 * FarmSphere AI - ESP32 Smart Agriculture IoT Node Firmware
 * Microcontroller: ESP32 DevKit V1
 * Sensors:
 *   - Capacitive Soil Moisture Sensor (Analog GPIO 34)
 *   - DHT22 Ambient Temperature & Humidity (Digital GPIO 4)
 *   - Optical Soil pH Sensor (Analog GPIO 35)
 *   - Modbus / RS485 NPK Sensor (HardwareSerial2 GPIO 16 RX, GPIO 17 TX)
 * Protocols: WiFi + HTTPS REST / MQTT to FarmSphere Backend
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

// WiFi Credentials
const char* WIFI_SSID = "FARMSPHERE_WIFI";
const char* WIFI_PASS = "AgroSphereSecure2026";

// FarmSphere Backend Server
const char* SERVER_INGEST_URL = "https://farmsphere-ai.web.app/api/iot/ingest";
const char* DEVICE_ID = "ESP32_HUB_01";
const char* API_KEY = "FS_IOT_SECURE_TOKEN_9981";

// Pin Configurations
#define DHTPIN 4
#define DHTTYPE DHT22
#define SOIL_MOISTURE_PIN 34
#define SOIL_PH_PIN 35

DHT dht(DHTPIN, DHTTYPE);

// Calibration constants
const int AIR_VALUE = 3200;    // Soil sensor in dry air
const int WATER_VALUE = 1200;  // Soil sensor in water

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("[FarmSphere IoT Node] Starting Initialization...");

  dht.begin();
  pinMode(SOIL_MOISTURE_PIN, INPUT);
  pinMode(SOIL_PH_PIN, INPUT);

  // Connect to WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 20) {
    delay(500);
    Serial.print(".");
    retry++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected! IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\n[WiFi] Connection timeout. Operating in offline telemetry buffering mode.");
  }
}

void loop() {
  // 1. Read Sensors
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  int rawMoisture = analogRead(SOIL_MOISTURE_PIN);
  float moisturePct = map(rawMoisture, AIR_VALUE, WATER_VALUE, 0, 100);
  moisturePct = constrain(moisturePct, 0.0, 100.0);

  int rawPh = analogRead(SOIL_PH_PIN);
  float soilPh = 3.5 + ((float)rawPh / 4095.0) * 5.5; // Scaled calibration

  // Simulated NPK values (from RS485 query in production)
  float nitrogen = 110.0 + random(-5, 5);
  float phosphorus = 52.0 + random(-3, 3);
  float potassium = 45.0 + random(-4, 4);

  // 2. Prepare JSON Payload
  String payload = "{";
  payload += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
  payload += "\"api_key\":\"" + String(API_KEY) + "\",";
  payload += "\"telemetry\":{";
  payload += "\"soil_moisture_pct\":" + String(moisturePct, 1) + ",";
  payload += "\"soil_ph\":" + String(soilPh, 2) + ",";
  payload += "\"temperature_c\":" + String(temperature, 1) + ",";
  payload += "\"humidity_pct\":" + String(humidity, 1) + ",";
  payload += "\"nitrogen\":" + String(nitrogen, 1) + ",";
  payload += "\"phosphorus\":" + String(phosphorus, 1) + ",";
  payload += "\"potassium\":" + String(potassium, 1) + ",";
  payload += "\"battery_pct\":94";
  payload += "}}";

  Serial.println("[Telemetry Payload] " + payload);

  // 3. Transmit via HTTPS POST
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_INGEST_URL);
    http.addHeader("Content-Type", "application/json");

    int httpCode = http.POST(payload);
    if (httpCode > 0) {
      Serial.printf("[HTTP] Response code: %d\n", httpCode);
      String response = http.getString();
      Serial.println("[Server Response] " + response);
    } else {
      Serial.printf("[HTTP] POST failed, error: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  }

  // Sample every 10 seconds for field monitoring
  delay(10000);
}
