/*
 * ============================================================
 *  CHROMA-SCAN  —  ESP8266 Master Controller
 * ============================================================
 *  Main brain of the spectrophotometer. Handles the OLED UI,
 *  4 buttons, active buzzer, WiFi local portal, dynamic AI 
 *  inference (MLP/Linear), and communicates with the Arduino Nano
 *  over Hardware Serial.
 *
 *  KSEF 2026  |  Project 6
 * ============================================================
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <LittleFS.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>
#include <math.h>

// ─── OLED & Screen Setup ────────────────────────────────────
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDR 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ─── Button & Buzzer Pins (Raw GPIO numbers) ────────────────
#define BTN_UP 12      // GPIO 12 (D6 on NodeMCU)
#define BTN_DOWN 13    // GPIO 13 (D7 on NodeMCU)
#define BTN_OK 14      // GPIO 14 (D5 on NodeMCU)
#define BTN_BACK 16    // GPIO 16 (D0 on NodeMCU)
#define BUZZER 15      // GPIO 15 (D8 on NodeMCU)

// ─── Predefined Standard Concentrations (mg/L) ─────────────
const float PRESETS[] = {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0};
#define NUM_PRESETS 8
const char* UNIT_LABELS[] = {"mg/L", "mol/L", "ppm"};
int currentUnitIndex = 0;

// ─── Dynamic AI Model Parameters ───────────────────────────
struct ModelParams {
  String modelType = "linear";
  float lin_w[5] = {0.0, 0.0, 0.0, 0.0, 0.0}; // intercept, wR, wG, wB, wAmb
  float mlp_w1[8][4];
  float mlp_b1[8];
  float mlp_w2[8];
  float mlp_b2 = 0.0;
  bool loaded = false;
};

ModelParams activeModel;

// ─── State Machine ──────────────────────────────────────────
enum State {
  STATE_WELCOME,
  STATE_MAIN_MENU,
  STATE_CAL_BLANK_PROMPT,
  STATE_CAL_BLANK_SCANNING,
  STATE_CAL_BLANK_DONE,
  STATE_ADD_STD_MENU,
  STATE_ADD_STD_PRESET,
  STATE_ADD_STD_MANUAL,
  STATE_ADD_STD_SCANNING,
  STATE_ADD_STD_DONE,
  STATE_MEASURE_SAMPLE_PROMPT,
  STATE_MEASURE_SAMPLE_SCANNING,
  STATE_MEASURE_SAMPLE_RESULTS,
  STATE_WIFI_PORTAL
};

State currentState = STATE_WELCOME;

// ─── UI & Selection Variables ──────────────────────────────
int menuIndex = 0;
int presetIndex = 0;
float manualConc = 0.0;
int concDigits[5] = {0, 0, 0, 0, 0}; // XXX.XX
int concDigitPos = 0;

// ─── Measurement Buffer ─────────────────────────────────────
float I0[3] = {1.0, 1.0, 1.0}; // Red, Green, Blue baseline
float rawAmbient = 0.0;
float rawR = 0.0;
float rawG = 0.0;
float rawB = 0.0;
float absorbance[3] = {0.0, 0.0, 0.0};
float concentrationResult = 0.0;
int bestChannel = 0; // index of highest absorbance
int resultPage = 0;
#define RESULT_PAGES 5
const char* CHANNEL_NAMES[] = {"Red", "Green", "Blue"};

// ─── Web Server ─────────────────────────────────────────────
ESP8266WebServer server(80);
bool wifiPortalActive = false;

// ─── Animations Helpers ─────────────────────────────────────
unsigned long lastFrameTime = 0;
float animAngle = 0;
float animWave = 0;
int animLaserY = 0;
bool animLaserDown = true;
int wifiPulseStep = 0;

// ═══════════════════════════════════════════════════════════
//  BUZZER & SOUND TONES
// ═══════════════════════════════════════════════════════════

void beepClick() {
  tone(BUZZER, 4000, 20);
}

void beepWelcome() {
  tone(BUZZER, 523, 100); delay(120);
  tone(BUZZER, 659, 100); delay(120);
  tone(BUZZER, 784, 150); delay(170);
}

void beepDone() {
  tone(BUZZER, 2000, 80); delay(120);
  tone(BUZZER, 2000, 80); delay(100);
}

void beepError() {
  for (int i = 0; i < 3; i++) {
    tone(BUZZER, 200, 80);
    delay(100);
  }
}

// ═══════════════════════════════════════════════════════════
//  DYNAMIC AI INFERENCE
// ═══════════════════════════════════════════════════════════

float calculateAbsorbance(float i0, float i) {
  if (i0 <= 0 || i <= 0) return 0.0;
  float ratio = i / i0;
  if (ratio >= 1.0) return 0.0;
  float A = -log10(ratio);
  return (isnan(A) || isinf(A)) ? 0.0 : A;
}

float runModelInference(float absR, float absG, float absB, float ambient) {
  if (!activeModel.loaded) {
    // Basic Beer-Lambert fallback (slope factor prediction)
    float maxAbs = absR;
    if (absG > maxAbs) maxAbs = absG;
    if (absB > maxAbs) maxAbs = absB;
    return maxAbs * 100.0;
  }
  
  if (activeModel.modelType == "linear") {
    float c = activeModel.lin_w[0] + 
              activeModel.lin_w[1] * absR + 
              activeModel.lin_w[2] * absG + 
              activeModel.lin_w[3] * absB + 
              activeModel.lin_w[4] * ambient;
    return (c < 0) ? 0.0 : c;
  } else if (activeModel.modelType == "mlp") {
    float inputs[4] = {absR, absG, absB, ambient};
    float hidden[8];
    
    // Hidden Layer (ReLU)
    for (int i = 0; i < 8; i++) {
      float sum = activeModel.mlp_b1[i];
      for (int j = 0; j < 4; j++) {
        sum += activeModel.mlp_w1[i][j] * inputs[j];
      }
      hidden[i] = (sum < 0) ? 0.0 : sum;
    }
    
    // Output Layer
    float out = activeModel.mlp_b2;
    for (int i = 0; i < 8; i++) {
      out += activeModel.mlp_w2[i] * hidden[i];
    }
    return (out < 0) ? 0.0 : out;
  }
  return 0.0;
}

bool loadModelParams() {
  if (!LittleFS.exists("/model_params.json")) {
    Serial1.println("No model_params.json file found, using firmware defaults.");
    activeModel.loaded = false;
    return false;
  }
  
  File file = LittleFS.open("/model_params.json", "r");
  if (!file) {
    Serial1.println("Failed to open model_params.json");
    return false;
  }
  
  JsonDocument doc;
  DeserializationError error = deserializeJson(doc, file);
  file.close();
  
  if (error) {
    Serial1.print("JSON Parse Error: ");
    Serial1.println(error.c_str());
    return false;
  }
  
  activeModel.modelType = doc["model_type"] | "linear";
  
  if (activeModel.modelType == "linear") {
    JsonArray w = doc["w"];
    for (int i = 0; i < 5; i++) {
      activeModel.lin_w[i] = w[i] | 0.0;
    }
  } else if (activeModel.modelType == "mlp") {
    JsonArray w1 = doc["w1"];
    JsonArray b1 = doc["b1"];
    JsonArray w2 = doc["w2"];
    activeModel.mlp_b2 = doc["b2"] | 0.0;
    
    for (int i = 0; i < 8; i++) {
      JsonArray row = w1[i];
      for (int j = 0; j < 4; j++) {
        activeModel.mlp_w1[i][j] = row[j] | 0.0;
      }
      activeModel.mlp_b1[i] = b1[i] | 0.0;
      activeModel.mlp_w2[i] = w2[i] | 0.0;
    }
  }
  
  activeModel.loaded = true;
  Serial1.println("AI model loaded successfully!");
  return true;
}

// ═══════════════════════════════════════════════════════════
//  STANDARDS DATABASE (LITTLEFS CSV FILE)
// ═══════════════════════════════════════════════════════════

void saveStandard(float conc, float ambient, float r, float g, float b) {
  bool exists = LittleFS.exists("/standards.csv");
  File file = LittleFS.open("/standards.csv", "a");
  if (!file) {
    Serial1.println("Failed to write to standards.csv");
    return;
  }
  if (!exists) {
    file.println("molarity,ambient,red,green,blue");
  }
  file.printf("%.4f,%.2f,%.2f,%.2f,%.2f\n", conc, ambient, r, g, b);
  file.close();
}

void clearStandards() {
  if (LittleFS.exists("/standards.csv")) {
    LittleFS.remove("/standards.csv");
  }
}

// ═══════════════════════════════════════════════════════════
//  UART COMMUNICATION WITH ARDUINO NANO
// ═══════════════════════════════════════════════════════════

String readNanoResponse() {
  unsigned long start = millis();
  String response = "";
  while (millis() - start < 6000) { // 6 second timeout
    if (Serial.available() > 0) {
      char c = Serial.read();
      if (c == '\n') {
        response.trim();
        return response;
      }
      response += c;
    }
    yield();
  }
  return "";
}

bool triggerNanoScan(float &ambient, float &r, float &g, float &b) {
  while(Serial.available() > 0) Serial.read(); // purge serial buffer
  
  Serial.println("SCAN");
  String response = readNanoResponse();
  
  if (response.startsWith("DATA:")) {
    response = response.substring(5);
    int firstComma = response.indexOf(',');
    int secondComma = response.indexOf(',', firstComma + 1);
    int thirdComma = response.indexOf(',', secondComma + 1);
    
    if (firstComma != -1 && secondComma != -1 && thirdComma != -1) {
      ambient = response.substring(0, firstComma).toFloat();
      r = response.substring(firstComma + 1, secondComma).toFloat();
      g = response.substring(secondComma + 1, thirdComma).toFloat();
      b = response.substring(thirdComma + 1).toFloat();
      return true;
    }
  }
  return false;
}

void setNanoLedColor(byte r, byte g, byte b) {
  Serial.printf("LED:%d,%d,%d\n", r, g, b);
}

// ═══════════════════════════════════════════════════════════
//  OLED RENDERING & ANIMATIONS
// ═══════════════════════════════════════════════════════════

void drawSpinningMolecule(int x, int y, int radius, float angle) {
  display.drawCircle(x, y, 6, SSD1306_WHITE);
  
  int ox1 = x + radius * cos(angle);
  int oy1 = y + radius * sin(angle);
  display.drawLine(x, y, ox1, oy1, SSD1306_WHITE);
  display.fillCircle(ox1, oy1, 3, SSD1306_WHITE);
  
  int ox2 = x + radius * cos(angle + PI);
  int oy2 = y + radius * sin(angle + PI);
  display.drawLine(x, y, ox2, oy2, SSD1306_WHITE);
  display.fillCircle(ox2, oy2, 3, SSD1306_WHITE);
}

void drawScanningLaser(int x, int y, int w, int h, int sweepY) {
  display.drawRoundRect(x, y, w, h, 3, SSD1306_WHITE);
  display.drawFastHLine(x - 3, y + sweepY, w + 6, SSD1306_WHITE);
}

void drawWiFiPulsing(int x, int y, int radius, int pulseStep) {
  display.fillCircle(x, y, 2, SSD1306_WHITE);
  if (pulseStep >= 1) display.drawCircle(x, y, radius, SSD1306_WHITE);
  if (pulseStep >= 2) display.drawCircle(x, y, radius * 2, SSD1306_WHITE);
  if (pulseStep >= 3) display.drawCircle(x, y, radius * 3, SSD1306_WHITE);
}

void updateAnimations() {
  unsigned long now = millis();
  if (now - lastFrameTime < 40) return;
  lastFrameTime = now;
  
  animAngle += 0.08;
  if (animAngle >= 2*PI) animAngle = 0;
  
  animWave += 0.15;
  if (animWave >= 2*PI) animWave = 0;
  
  if (animLaserDown) {
    animLaserY++;
    if (animLaserY >= 26) animLaserDown = false;
  } else {
    animLaserY--;
    if (animLaserY <= 2) animLaserDown = true;
  }
  
  static int pulseCounter = 0;
  pulseCounter++;
  if (pulseCounter >= 10) {
    pulseCounter = 0;
    wifiPulseStep = (wifiPulseStep + 1) % 4;
  }
}

void renderState() {
  display.clearDisplay();
  
  switch(currentState) {
    case STATE_WELCOME:
      display.drawRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, SSD1306_WHITE);
      drawSpinningMolecule(30, 32, 14, animAngle);
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(62, 18);
      display.print("CHROMA");
      display.setCursor(62, 28);
      display.print("SCAN");
      display.setCursor(62, 38);
      display.print("Edge AI v2");
      display.setCursor(20, 53);
      display.setTextSize(1);
      display.print("Press OK to start");
      break;
      
    case STATE_MAIN_MENU:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.print("--- MAIN MENU ---");
      for (int i = 0; i < 4; i++) {
        display.setCursor(10, 16 + i * 11);
        if (i == 0) display.print("Measure Sample");
        else if (i == 1) display.print("Calibrate (Blank)");
        else if (i == 2) display.print("Add Standard");
        else if (i == 3) display.print("WiFi Portal / AP");
        if (i == menuIndex) {
          display.setCursor(0, 16 + i * 11);
          display.print(">");
        }
      }
      break;
      
    case STATE_CAL_BLANK_PROMPT:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.print("Calibration Setup:");
      display.setCursor(0, 20);
      display.print("Insert BLANK cuvette");
      display.setCursor(0, 32);
      display.print("Press OK to scan.");
      break;
      
    case STATE_CAL_BLANK_SCANNING:
    case STATE_MEASURE_SAMPLE_SCANNING:
    case STATE_ADD_STD_SCANNING:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(44, 20);
      display.print("Scanning");
      display.setCursor(44, 32);
      display.print("Solution...");
      drawScanningLaser(15, 10, 18, 30, animLaserY);
      break;
      
    case STATE_CAL_BLANK_DONE:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.print("Baseline Configured!");
      display.setCursor(0, 16);
      display.printf("R:%d G:%d B:%d", (int)I0[0], (int)I0[1], (int)I0[2]);
      display.setCursor(0, 48);
      display.print("Press OK to proceed.");
      break;
      
    case STATE_ADD_STD_MENU:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.print("Enter Standard Conc:");
      display.setCursor(10, 20);
      display.print("Preset Values");
      display.setCursor(10, 32);
      display.print("Manual Entry");
      display.setCursor(0, 20 + menuIndex * 12);
      display.print(">");
      break;
      
    case STATE_ADD_STD_PRESET:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.print("Select Preset Conc:");
      display.setTextSize(2);
      display.setCursor(10, 20);
      display.print(PRESETS[presetIndex], 2);
      display.setTextSize(1);
      display.setCursor(10, 42);
      display.print(UNIT_LABELS[currentUnitIndex]);
      display.setCursor(0, 54);
      display.print("Up/Dn:Sel  OK:Scan");
      break;
      
    case STATE_ADD_STD_MANUAL:
      {
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        display.print("Manual Enter (XXX.XX):");
        char buf[10];
        snprintf_P(buf, sizeof(buf), PSTR("%d%d%d.%d%d"), concDigits[0], concDigits[1], concDigits[2], concDigits[3], concDigits[4]);
        display.setTextSize(2);
        display.setCursor(10, 20);
        display.print(buf);
        int cursorCol = concDigitPos;
        if (concDigitPos >= 3) cursorCol++;
        int cursorX = 10 + cursorCol * 12;
        display.drawLine(cursorX, 38, cursorX + 10, 38, SSD1306_WHITE);
        display.setTextSize(1);
        display.setCursor(0, 50);
        display.print("Up: +1  OK: Next/Scan");
      }
      break;
      
    case STATE_ADD_STD_DONE:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.print("Standard Point Saved!");
      display.setCursor(0, 16);
      display.printf("Conc: %.2f %s", manualConc, UNIT_LABELS[currentUnitIndex]);
      display.setCursor(0, 32);
      display.printf("Abs (Max): %.4f", absorbance[0]); // placeholder max channel abs
      display.setCursor(0, 52);
      display.print("Press OK to continue");
      break;
      
    case STATE_MEASURE_SAMPLE_PROMPT:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.print("Measure Sample:");
      display.setCursor(0, 20);
      display.print("Insert sample cuvette");
      display.setCursor(0, 32);
      display.print("Press OK to scan.");
      break;
      
    case STATE_MEASURE_SAMPLE_RESULTS:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(95, 0);
      display.printf("[%d/5]", resultPage + 1);
      
      switch (resultPage) {
        case 0:
          display.setCursor(0, 0); display.print("Absorbance Red:");
          display.setTextSize(2); display.setCursor(0, 18); display.print(absorbance[0], 4);
          break;
        case 1:
          display.setCursor(0, 0); display.print("Absorbance Green:");
          display.setTextSize(2); display.setCursor(0, 18); display.print(absorbance[1], 4);
          break;
        case 2:
          display.setCursor(0, 0); display.print("Absorbance Blue:");
          display.setTextSize(2); display.setCursor(0, 18); display.print(absorbance[2], 4);
          break;
        case 3:
          display.setCursor(0, 0); display.print("Best Wavelength:");
          display.setTextSize(2); display.setCursor(0, 18); display.print(CHANNEL_NAMES[bestChannel]);
          break;
        case 4:
          display.setCursor(0, 0); display.print("Concentration:");
          display.setTextSize(2); display.setCursor(0, 18); display.print(concentrationResult, 2);
          display.setTextSize(1); display.setCursor(0, 38); display.print(UNIT_LABELS[currentUnitIndex]);
          break;
      }
      display.setTextSize(1);
      display.setCursor(0, 52);
      display.print("OK: Next page  Back: Exit");
      break;
      
    case STATE_WIFI_PORTAL:
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.print("WiFi Training Portal");
      drawWiFiPulsing(100, 25, 6, wifiPulseStep);
      display.setCursor(0, 18);
      display.print("AP: Chroma-Scan-AP");
      display.setCursor(0, 30);
      display.print("IP: 192.168.4.1");
      display.setCursor(0, 46);
      display.print("Upload model configs.");
      display.setCursor(0, 56);
      display.print("Press BACK to exit");
      break;
  }
  
  display.display();
}

// ═══════════════════════════════════════════════════════════
//  LOCAL WEB SERVER & AP INTERFACE
// ═══════════════════════════════════════════════════════════

const char HTTP_INDEX[] PROGMEM = R"raw(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chroma-Scan Control Panel</title>
  <style>
    :root {
      --bg-color: #0b0f19;
      --panel-bg: rgba(20, 28, 47, 0.7);
      --border-color: rgba(56, 189, 248, 0.2);
      --glow-color: #38bdf8;
      --text-color: #f8fafc;
      --text-muted: #94a3b8;
    }
    body {
      background-color: var(--bg-color);
      color: var(--text-color);
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      margin: 0;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }
    .container {
      width: 100%;
      max-width: 650px;
    }
    header {
      text-align: center;
      margin-bottom: 30px;
    }
    h1 {
      font-size: 2.5rem;
      margin: 0;
      background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.3));
    }
    p.subtitle {
      color: var(--text-muted);
      margin: 5px 0 0 0;
    }
    .panel {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    h2 {
      margin-top: 0;
      font-size: 1.25rem;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 8px;
      color: var(--glow-color);
    }
    .btn {
      background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%);
      border: none;
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: bold;
      transition: all 0.2s ease;
      display: inline-block;
      text-decoration: none;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }
    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    .btn-danger {
      background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
      box-shadow: 0 0 12px rgba(239, 68, 68, 0.2);
    }
    .btn-danger:hover {
      box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }
    .file-input {
      display: none;
    }
    .file-label {
      border: 2px dashed var(--border-color);
      border-radius: 8px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      cursor: pointer;
      transition: border-color 0.2s ease;
    }
    .file-label:hover {
      border-color: var(--glow-color);
    }
    .file-label span {
      margin-top: 10px;
      color: var(--text-muted);
    }
    .status-badge {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.85rem;
      font-weight: bold;
    }
    .status-active {
      background-color: rgba(34, 197, 94, 0.2);
      color: #22c55e;
      border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .status-inactive {
      background-color: rgba(239, 68, 68, 0.2);
      color: #ef4444;
      border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .footer {
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-top: 40px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Chroma-Scan Panel</h1>
      <p class="subtitle">WiFi Edge AI Data Hub & Calibration Portal</p>
    </header>

    <div class="panel">
      <h2>Step 1: Export Calibration Dataset</h2>
      <p style="margin-bottom: 20px; color: var(--text-muted);">
        Download the collected RGB absorbance points from the spectrophotometer. You will upload this CSV file to the external Chroma-Train site to generate your optimal regression parameters.
      </p>
      <a href="/download_csv" class="btn">Download Dataset (.csv)</a>
      <a href="/clear_data" class="btn btn-danger" onclick="return confirm('Clear all collected standard data points?')">Clear Data</a>
    </div>

    <div class="panel">
      <h2>Step 2: Upload Trained AI Model</h2>
      <p style="margin-bottom: 20px; color: var(--text-muted);">
        Drag and drop the trained model configuration file (model_params.json) that you downloaded from Chroma-Train. This instantly replaces the active predictive logic.
      </p>
      <form action="/upload_model" method="POST" enctype="multipart/form-data" id="uploadForm">
        <label class="file-label">
          <svg style="width: 48px; height: 48px; fill: var(--text-muted);" viewBox="0 0 24 24">
            <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
          </svg>
          <span id="file-name-display">Click to Select model_params.json</span>
          <input type="file" name="model_file" class="file-input" id="model_file" accept=".json" onchange="updateFileName()">
        </label>
        <button type="submit" class="btn" style="margin-top: 20px; width: 100%;">Deploy Model to Edge</button>
      </form>
    </div>

    <div class="panel">
      <h2>Device Status</h2>
      <p>Active Prediction Model: <span class="status-badge status-active" id="model-status">Loading...</span></p>
    </div>

    <div class="footer">
      KSEF 2026 Project 6 | Powered by Galvaniy Technologies
    </div>
  </div>

  <script>
    function updateFileName() {
      const fileInput = document.getElementById('model_file');
      const nameDisplay = document.getElementById('file-name-display');
      if (fileInput.files.length > 0) {
        nameDisplay.textContent = fileInput.files[0].name;
      }
    }
    // Fetch active model parameter type
    fetch('/status').then(r => r.json()).then(data => {
      document.getElementById('model-status').textContent = data.model_loaded ? data.model_type.toUpperCase() : "NONE (DEFAULT SCAN)";
      if (!data.model_loaded) {
        document.getElementById('model-status').className = "status-badge status-inactive";
      }
    });
  </script>
</body>
</html>
)raw";

void handleRoot() {
  server.send_P(200, "text/html", HTTP_INDEX);
}

void handleStatus() {
  JsonDocument doc;
  doc["model_loaded"] = activeModel.loaded;
  doc["model_type"] = activeModel.modelType;
  String out;
  serializeJson(doc, out);
  server.send(200, "application/json", out);
}

void handleDownloadCsv() {
  if (!LittleFS.exists("/standards.csv")) {
    server.send(404, "text/plain", "No standard points gathered yet.");
    return;
  }
  File file = LittleFS.open("/standards.csv", "r");
  server.streamFile(file, "text/csv");
  file.close();
}

void handleClearData() {
  clearStandards();
  server.sendHeader("Location", "/");
  server.send(303);
}

void handleUploadModel() {
  HTTPUpload& upload = server.upload();
  if (upload.status == UPLOAD_FILE_START) {
    String filename = upload.filename;
    if (!filename.endsWith(".json")) {
      server.send(400, "text/plain", "Invalid file extension. Only .json allowed.");
      return;
    }
    // Delete old file if exists
    if (LittleFS.exists("/model_params.json")) {
      LittleFS.remove("/model_params.json");
    }
    // Open new file
    File file = LittleFS.open("/model_params.json", "w");
    if (!file) {
      server.send(500, "text/plain", "Failed to open params file for writing.");
      return;
    }
    file.close();
  } else if (upload.status == UPLOAD_FILE_WRITE) {
    File file = LittleFS.open("/model_params.json", "a");
    if (file) {
      file.write(upload.buf, upload.currentSize);
      file.close();
    }
  } else if (upload.status == UPLOAD_FILE_END) {
    if (loadModelParams()) {
      server.sendHeader("Location", "/");
      server.send(303);
    } else {
      server.send(400, "text/plain", "Failed to parse parameters. Ensure it is a valid model_params.json.");
    }
  }
}

// ═══════════════════════════════════════════════════════════
//  BUTTON CONTROLS (DEBOUNCED & THREAD-SAFE)
// ═══════════════════════════════════════════════════════════

bool readBtnDebounced(int pin) {
  static unsigned long lastPress[4] = {0, 0, 0, 0};
  int idx = 0;
  if (pin == BTN_UP) idx = 0;
  else if (pin == BTN_DOWN) idx = 1;
  else if (pin == BTN_OK) idx = 2;
  else if (pin == BTN_BACK) idx = 3;

  if (digitalRead(pin) == LOW) { // pullups make LOW the active state
    if (millis() - lastPress[idx] > 250) {
      lastPress[idx] = millis();
      beepClick();
      return true;
    }
  }
  return false;
}

// ═══════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════

void setup() {
  // Debug output
  Serial1.begin(115200); // GPIO 2 (TXD1) for debugging
  Serial1.println("\n--- CHROMA-SCAN MASTER CORE SETUP ---");
  
  // Hardware UART link to Arduino Nano
  Serial.begin(9600); // Pins RXD0/TXD0 to communicate with Nano
  
  // Pin Configurations
  pinMode(BTN_UP, INPUT_PULLUP);
  pinMode(BTN_DOWN, INPUT_PULLUP);
  pinMode(BTN_OK, INPUT_PULLUP);
  pinMode(BTN_BACK, INPUT_PULLUP);
  pinMode(BUZZER, OUTPUT);
  digitalWrite(BUZZER, LOW);
  
  // Filesystem Init
  if (LittleFS.begin()) {
    Serial1.println("FileSystem Mounted Successfully.");
    loadModelParams();
  } else {
    Serial1.println("LittleFS Mount Failed!");
  }
  
  // OLED Init
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial1.println("SSD1306 OLED Init Failed!");
    for (;;) {
      tone(BUZZER, 200, 200);
      delay(500);
    }
  }
  display.clearDisplay();
  display.display();
  
  beepWelcome();
  
  // Setup standard web requests
  server.on("/", HTTP_GET, handleRoot);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/download_csv", HTTP_GET, handleDownloadCsv);
  server.on("/clear_data", HTTP_GET, handleClearData);
  server.on("/upload_model", HTTP_POST, handleRoot, handleUploadModel);
}

// ═══════════════════════════════════════════════════════════
//  LOOP STATE MACHINE & FLOW
// ═══════════════════════════════════════════════════════════

void loop() {
  // Handle portal requests if WiFi is active
  if (wifiPortalActive) {
    server.handleClient();
  }
  
  updateAnimations();
  renderState();
  
  // Check hardware buttons
  bool upPressed = readBtnDebounced(BTN_UP);
  bool downPressed = readBtnDebounced(BTN_DOWN);
  bool okPressed = readBtnDebounced(BTN_OK);
  bool backPressed = readBtnDebounced(BTN_BACK);
  
  switch(currentState) {
    case STATE_WELCOME:
      if (okPressed) {
        currentState = STATE_MAIN_MENU;
        menuIndex = 0;
      }
      break;
      
    case STATE_MAIN_MENU:
      if (upPressed) {
        menuIndex = (menuIndex - 1 + 4) % 4;
      } else if (downPressed) {
        menuIndex = (menuIndex + 1) % 4;
      } else if (okPressed) {
        if (menuIndex == 0) {
          currentState = STATE_MEASURE_SAMPLE_PROMPT;
        } else if (menuIndex == 1) {
          currentState = STATE_CAL_BLANK_PROMPT;
        } else if (menuIndex == 2) {
          currentState = STATE_ADD_STD_MENU;
          menuIndex = 0;
        } else if (menuIndex == 3) {
          currentState = STATE_WIFI_PORTAL;
          WiFi.mode(WIFI_AP);
          WiFi.softAP("Chroma-Scan-AP", "chromascan123");
          server.begin();
          wifiPortalActive = true;
          Serial1.println("AP Mode Active: http://192.168.4.1/");
        }
      } else if (backPressed) {
        currentState = STATE_WELCOME;
      }
      break;
      
    case STATE_CAL_BLANK_PROMPT:
      if (okPressed) {
        {
          currentState = STATE_CAL_BLANK_SCANNING;
          renderState(); // render scanning view instantly
          float dummyAmb, r, g, b;
          if (triggerNanoScan(dummyAmb, r, g, b)) {
            I0[0] = r;
            I0[1] = g;
            I0[2] = b;
            rawAmbient = dummyAmb;
            clearStandards(); // Reset standards.csv for the new calibration run
            saveStandard(0.0, dummyAmb, r, g, b); // Automatically save the blank as 0.0 concentration
            beepDone();
            currentState = STATE_CAL_BLANK_DONE;
          } else {
            beepError();
            currentState = STATE_MAIN_MENU;
          }
        }
      } else if (backPressed) {
        currentState = STATE_MAIN_MENU;
      }
      break;
      
    case STATE_CAL_BLANK_DONE:
      if (okPressed || backPressed) {
        currentState = STATE_MAIN_MENU;
      }
      break;
      
    case STATE_ADD_STD_MENU:
      if (upPressed || downPressed) {
        menuIndex = (menuIndex == 0) ? 1 : 0;
      } else if (okPressed) {
        if (menuIndex == 0) {
          currentState = STATE_ADD_STD_PRESET;
          presetIndex = 0;
        } else {
          currentState = STATE_ADD_STD_MANUAL;
          concDigitPos = 0;
          memset(concDigits, 0, sizeof(concDigits));
        }
      } else if (backPressed) {
        currentState = STATE_MAIN_MENU;
      }
      break;
      
    case STATE_ADD_STD_PRESET:
      if (upPressed) {
        presetIndex = (presetIndex - 1 + NUM_PRESETS) % NUM_PRESETS;
      } else if (downPressed) {
        presetIndex = (presetIndex + 1) % NUM_PRESETS;
      } else if (okPressed) {
        {
          manualConc = PRESETS[presetIndex];
          currentState = STATE_ADD_STD_SCANNING;
          renderState();
          float dummyAmb, r, g, b;
          if (triggerNanoScan(dummyAmb, r, g, b)) {
            absorbance[0] = calculateAbsorbance(I0[0], r);
            absorbance[1] = calculateAbsorbance(I0[1], g);
            absorbance[2] = calculateAbsorbance(I0[2], b);
            saveStandard(manualConc, dummyAmb, r, g, b);
            beepDone();
            currentState = STATE_ADD_STD_DONE;
          } else {
            beepError();
            currentState = STATE_MAIN_MENU;
          }
        }
      } else if (backPressed) {
        currentState = STATE_ADD_STD_MENU;
      }
      break;
      
    case STATE_ADD_STD_MANUAL:
      if (upPressed) {
        concDigits[concDigitPos] = (concDigits[concDigitPos] + 1) % 10;
      } else if (okPressed) {
        concDigitPos++;
        if (concDigitPos >= 5) {
          {
            // Finished digits entry -> scan standard
            manualConc = concDigits[0]*100.0 + concDigits[1]*10.0 + concDigits[2]*1.0 + concDigits[3]*0.1 + concDigits[4]*0.01;
            currentState = STATE_ADD_STD_SCANNING;
            renderState();
            float dummyAmb, r, g, b;
            if (triggerNanoScan(dummyAmb, r, g, b)) {
              absorbance[0] = calculateAbsorbance(I0[0], r);
              absorbance[1] = calculateAbsorbance(I0[1], g);
              absorbance[2] = calculateAbsorbance(I0[2], b);
              saveStandard(manualConc, dummyAmb, r, g, b);
              beepDone();
              currentState = STATE_ADD_STD_DONE;
            } else {
              beepError();
              currentState = STATE_MAIN_MENU;
            }
          }
        }
      } else if (backPressed) {
        if (concDigitPos > 0) concDigitPos--;
        else currentState = STATE_ADD_STD_MENU;
      }
      break;
      
    case STATE_ADD_STD_DONE:
      if (okPressed || backPressed) {
        currentState = STATE_MAIN_MENU;
      }
      break;
      
    case STATE_MEASURE_SAMPLE_PROMPT:
      if (okPressed) {
        {
          currentState = STATE_MEASURE_SAMPLE_SCANNING;
          renderState();
          float dummyAmb, r, g, b;
          if (triggerNanoScan(dummyAmb, r, g, b)) {
            absorbance[0] = calculateAbsorbance(I0[0], r);
            absorbance[1] = calculateAbsorbance(I0[1], g);
            absorbance[2] = calculateAbsorbance(I0[2], b);
            
            // Determine best channel based on highest absorbance
            float maxAbs = absorbance[0];
            bestChannel = 0;
            for (int i = 1; i < 3; i++) {
              if (absorbance[i] > maxAbs) {
                maxAbs = absorbance[i];
                bestChannel = i;
              }
            }
            
            // Call Edge AI prediction
            concentrationResult = runModelInference(absorbance[0], absorbance[1], absorbance[2], dummyAmb);
            resultPage = 0;
            beepDone();
            currentState = STATE_MEASURE_SAMPLE_RESULTS;
          } else {
            beepError();
            currentState = STATE_MAIN_MENU;
          }
        }
      } else if (backPressed) {
        currentState = STATE_MAIN_MENU;
      }
      break;
      
    case STATE_MEASURE_SAMPLE_RESULTS:
      if (okPressed) {
        resultPage = (resultPage + 1) % RESULT_PAGES;
      } else if (backPressed) {
        currentState = STATE_MAIN_MENU;
      }
      break;
      
    case STATE_WIFI_PORTAL:
      if (backPressed) {
        server.stop();
        WiFi.softAPdisconnect(true);
        wifiPortalActive = false;
        currentState = STATE_MAIN_MENU;
      }
      break;
      
    default:
      break;
  }
}
