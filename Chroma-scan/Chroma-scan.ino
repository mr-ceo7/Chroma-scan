/*
 * ============================================================
 *  CHROMA-SCAN  —  ESP8266 Master Controller & Edge AI Portal
 * ============================================================
 *  Primary controller of the spectrophotometer system. Handles:
 *    - I2C SSD1306 OLED 128x64 display (SDA=GPIO4/D2, SCL=GPIO5/D1)
 *    - 4 Tactile Buttons (UP=12, DOWN=2, OK=14, BACK=13)
 *    - State machine navigation, calibration, and settings
 *    - Web Server AP dashboard and LittleFS csv standards log
 *    - Running Edge AI model inference (Linear/MLP)
 *    - Commands the Arduino Nano slave driver to scan or beep.
 *
 *  KSEF 2026  |  Project 6
 * ============================================================
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <LittleFS.h>
#include <ArduinoJson.h>
#include <math.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ─── OLED Configuration ─────────────────────────────────────
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDR 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ─── Pin Definitions ────────────────────────────────────────
#define STATUS_LED_PIN 16 // Onboard LED (D0 on NodeMCU, active-LOW)

#define BTN_UP 12
#define BTN_DOWN 2
#define BTN_OK 14
#define BTN_BACK 13

// ─── Preset Concentrations (mg/L) ───────────────────────────
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
  STATE_WIFI_PORTAL,
  STATE_SETTINGS
};
State currentState = STATE_WELCOME;
unsigned long stateStartTime = 0; // Timestamp of state entry for timed animations

// ─── UI & Selection Variables ──────────────────────────────
int menuIndex = 0;
float smoothMenuIndex = 0.0; // Interpolated menu index for smooth carousel transitions
int presetIndex = 0;
float manualConc = 0.0;
int concDigits[5] = {0, 0, 0, 0, 0}; // XXX.XX
int concDigitPos = 0;

// ─── Measurement Buffer ─────────────────────────────────────
float I0[3] = {1.0, 1.0, 1.0}; // Red, Green, Blue baseline raw counts
float rawAmbient = 0;
float rawR = 0;
float rawG = 0;
float rawB = 0;
float absorbance[3] = {0.0, 0.0, 0.0};
float concentrationResult = 0.0;
int bestChannel = 0;
int resultPage = 0;
#define RESULT_PAGES 2
const char* CHANNEL_NAMES[] = {"Red", "Green", "Blue"};

bool isCalibrated = false;

int scanStep = 0;
int scanSampleCount = 0;
#define NUM_SAMPLES 30

// ─── Animation Helpers ─────────────────────────────────────
unsigned long lastFrameTime = 0;
int moleculeStep = 0;
int animLaserY = 0;
bool animLaserDown = true;
int wifiPulseStep = 0;

// Molecule Animation Lookup Tables (radius = 14)
const int8_t MOLECULE_DX[] PROGMEM = {14, 10, 0, -10, -14, -10, 0, 10};
const int8_t MOLECULE_DY[] PROGMEM = {0, 10, 14, 10, 0, -10, -14, -10};

// ─── Web Server ─────────────────────────────────────────────
ESP8266WebServer server(80);
bool wifiPortalActive = false;

// Serial RX Buffer (eliminates String overhead)
char serialBuf[96];
int serialLen = 0;

// Forward Declarations
void changeState(State newState);
void handleButtonPresses(bool up, bool down, bool ok, bool back);
void onScanComplete();
void triggerBeep(const char* type);
void beepClick();
void beepWelcome();
void beepDone();
void beepError();

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
    Serial.println("DEBUG: No model_params.json file found, using firmware defaults.");
    activeModel.loaded = false;
    return false;
  }
  
  File file = LittleFS.open("/model_params.json", "r");
  if (!file) {
    Serial.println("DEBUG: Failed to open model_params.json");
    return false;
  }
  
  JsonDocument doc;
  DeserializationError error = deserializeJson(doc, file);
  file.close();
  
  if (error) {
    Serial.print("DEBUG: JSON Parse Error: ");
    Serial.println(error.c_str());
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
  Serial.print("DEBUG: AI model loaded successfully! Type: ");
  Serial.println(activeModel.modelType);
  return true;
}

// ═══════════════════════════════════════════════════════════
//  STANDARDS DATABASE (LITTLEFS CSV FILE)
// ═══════════════════════════════════════════════════════════

void saveStandard(float conc, float ambient, float r, float g, float b) {
  bool exists = LittleFS.exists("/standards.csv");
  File file = LittleFS.open("/standards.csv", "a");
  if (!file) {
    Serial.println("DEBUG: Failed to write to standards.csv");
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
//  LOCAL WEB SERVER & AP INTERFACE
// ═══════════════════════════════════════════════════════════

const char HTTP_INDEX[] PROGMEM = R"raw(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chroma-Scan Spectrophotometer Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
      --bg-dark: #070b19;
      --bg-card: rgba(16, 24, 48, 0.6);
      --border-glow: rgba(56, 189, 248, 0.12);
      --primary: #0ea5e9;
      --primary-glow: rgba(14, 165, 233, 0.2);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.15);
      --danger: #ef4444;
      --danger-glow: rgba(239, 68, 68, 0.15);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      
      --red-ch: #ff4d4d;
      --green-ch: #10e080;
      --blue-ch: #3b82f6;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Outfit', sans-serif;
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 10% 10%, rgba(14, 165, 233, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 90% 90%, rgba(129, 140, 248, 0.06) 0%, transparent 45%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px;
      line-height: 1.5;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .container {
      width: 100%;
      max-width: 1000px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 30px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-glow);
    }

    .logo-section h1 {
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(135deg, #0ea5e9 0%, #818cf8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-icon {
      width: 32px;
      height: 32px;
      fill: #0ea5e9;
      filter: drop-shadow(0 0 8px rgba(14, 165, 233, 0.5));
    }

    .status-pills {
      display: flex;
      gap: 12px;
    }

    .status-pill {
      background: rgba(15, 23, 42, 0.4);
      border: 1px solid var(--border-glow);
      padding: 6px 14px;
      border-radius: 99px;
      font-size: 0.85rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text-muted);
    }

    .status-pill.active {
      color: var(--text-main);
      border-color: rgba(56, 189, 248, 0.3);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background-color: var(--text-muted);
    }

    .status-dot.active {
      background-color: var(--success);
      box-shadow: 0 0 10px var(--success);
    }

    .dashboard-grid {
      display: grid;
      grid-template-columns: 1.55fr 1fr;
      gap: 24px;
    }

    @media (max-width: 868px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
      header {
        flex-direction: column;
        align-items: flex-start;
        gap: 16px;
      }
      .status-pills {
        width: 100%;
        justify-content: flex-start;
        flex-wrap: wrap;
      }
    }

    .card {
      background: var(--bg-card);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border-glow);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
      margin-bottom: 24px;
      transition: all 0.3s ease;
    }

    .card:hover {
      border-color: rgba(56, 189, 248, 0.2);
      box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }

    .card-title {
      font-size: 1.2rem;
      font-weight: 600;
      color: var(--primary);
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .card-title svg {
      width: 20px;
      height: 20px;
      fill: currentColor;
    }

    .chart-container {
      position: relative;
      width: 100%;
      background: rgba(7, 10, 19, 0.45);
      border-radius: 12px;
      padding: 12px;
      border: 1px solid rgba(255, 255, 255, 0.03);
    }

    canvas {
      display: block;
      width: 100%;
      cursor: crosshair;
    }

    .chart-legend {
      display: flex;
      justify-content: center;
      gap: 16px;
      margin-top: 16px;
      flex-wrap: wrap;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      cursor: pointer;
      padding: 6px 12px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
      transition: all 0.2s ease;
      user-select: none;
    }

    .legend-item:hover {
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(56, 189, 248, 0.2);
    }

    .legend-item.disabled {
      opacity: 0.35;
      text-decoration: line-through;
    }

    .legend-color {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }

    .color-red { background-color: var(--red-ch); box-shadow: 0 0 6px var(--red-ch); }
    .color-green { background-color: var(--green-ch); box-shadow: 0 0 6px var(--green-ch); }
    .color-blue { background-color: var(--blue-ch); box-shadow: 0 0 6px var(--blue-ch); }

    .table-wrapper {
      overflow-x: auto;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      background: rgba(7, 10, 19, 0.2);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.9rem;
    }

    th, td {
      padding: 14px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    th {
      background: rgba(10, 15, 30, 0.75);
      font-weight: 600;
      color: var(--text-muted);
    }

    tr:last-child td {
      border-bottom: none;
    }

    tr:hover td {
      background: rgba(255, 255, 255, 0.015);
    }

    .best-badge {
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      display: inline-block;
    }

    .best-red { background: rgba(239, 68, 68, 0.12); color: var(--red-ch); border: 1px solid rgba(239, 68, 68, 0.25); }
    .best-green { background: rgba(16, 185, 129, 0.12); color: var(--green-ch); border: 1px solid rgba(16, 185, 129, 0.25); }
    .best-blue { background: rgba(59, 130, 246, 0.12); color: var(--blue-ch); border: 1px solid rgba(59, 130, 246, 0.25); }

    .btn-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .btn {
      width: 100%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      background: linear-gradient(135deg, var(--primary) 0%, #3b82f6 100%);
      color: white;
      border: none;
      padding: 14px 20px;
      border-radius: 12px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
    }

    .btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 18px rgba(14, 165, 233, 0.3);
    }

    .btn-danger {
      background: linear-gradient(135deg, var(--danger) 0%, #b91c1c 100%);
      box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
    }

    .btn-danger:hover {
      box-shadow: 0 6px 18px rgba(239, 68, 68, 0.25);
    }

    .upload-zone {
      border: 2px dashed rgba(56, 189, 248, 0.2);
      border-radius: 12px;
      padding: 26px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(14, 165, 233, 0.01);
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--primary);
      background: rgba(14, 165, 233, 0.04);
    }

    .upload-zone svg {
      width: 38px;
      height: 38px;
      fill: var(--text-muted);
      margin-bottom: 10px;
      transition: fill 0.2s ease;
    }

    .upload-zone:hover svg {
      fill: var(--primary);
    }

    .file-input {
      display: none;
    }

    .info-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-bottom: 20px;
    }

    .info-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.9rem;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,0.03);
    }

    .info-row:last-child {
      border-bottom: none;
    }

    .info-label {
      color: var(--text-muted);
    }

    .info-value {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 500;
      color: var(--text-main);
    }

    .deploy-status {
      margin-top: 12px;
      font-size: 0.85rem;
      text-align: center;
      min-height: 20px;
    }

    .footer {
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-top: 40px;
      padding-bottom: 20px;
      width: 100%;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-section">
        <h1>
          <svg class="logo-icon" viewBox="0 0 24 24">
            <path d="M12 2A10 10 0 0 0 2 12a10 10 0 0 0 10 10 10 10 0 0 0 10-10A10 10 0 0 0 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
          </svg>
          Chroma-Scan
        </h1>
      </div>
      <div class="status-pills">
        <div class="status-pill active" id="model-pill">
          <div class="status-dot active"></div>
          Model: <span id="model-type">Loading...</span>
        </div>
        <div class="status-pill active">
          <div class="status-dot active"></div>
          Node MCU Active
        </div>
      </div>
    </header>

    <div class="dashboard-grid">
      <!-- Left side: chart and table -->
      <div class="main-column">
        <div class="card">
          <div class="card-title">
            <svg viewBox="0 0 24 24"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>
            Real-Time Calibration Curve
          </div>
          <div class="chart-container">
            <canvas id="curveChart"></canvas>
          </div>
          <div class="chart-legend">
            <div class="legend-item" id="legend-red" onclick="toggleChannel('red')">
              <div class="legend-color color-red"></div> Red (625nm)
            </div>
            <div class="legend-item" id="legend-green" onclick="toggleChannel('green')">
              <div class="legend-color color-green"></div> Green (525nm)
            </div>
            <div class="legend-item" id="legend-blue" onclick="toggleChannel('blue')">
              <div class="legend-color color-blue"></div> Blue (465nm)
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <svg viewBox="0 0 24 24"><path d="M4 6H2v14a2 2 0 0 0 2 2h14v-2H4zm16-4H8a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2zm0 14H8V4h12z"/></svg>
            Calibration Standards Log
          </div>
          <div class="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Molarity</th>
                  <th>Red Abs (A)</th>
                  <th>Green Abs (A)</th>
                  <th>Blue Abs (A)</th>
                  <th>Best Channel</th>
                </tr>
              </thead>
              <tbody id="data-tbody">
                <tr>
                  <td colspan="5" style="text-align:center;color:var(--text-muted);">Loading calibration points...</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Right side: controls and AI Deployer -->
      <div class="side-column">
        <div class="card">
          <div class="card-title">
            <svg viewBox="0 0 24 24"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.488.488 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 0 0-.48-.41h-3.84a.48.48 0 0 0-.48.4l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87a.49.49 0 0 0 .12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58a.49.49 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.4l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32a.49.49 0 0 0-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.63 3.6-3.6 3.6z"/></svg>
            System Status & Controls
          </div>
          <div class="info-list">
            <div class="info-row">
              <span class="info-label">Active Unit:</span>
              <span class="info-value" id="unit-display">-</span>
            </div>
            <div class="info-row">
              <span class="info-label">Blank Reference:</span>
              <span class="info-value" id="ref-display">-</span>
            </div>
          </div>
          <div class="btn-group">
            <a href="/download_csv" class="btn">
              <svg style="width:18px;height:18px;fill:currentColor" viewBox="0 0 24 24"><path d="M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/></svg>
              Download CSV Dataset
            </a>
            <button class="btn btn-danger" onclick="confirmAndClear()">
              <svg style="width:18px;height:18px;fill:currentColor" viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
              Clear Standards Log
            </button>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
            Edge AI Model Deployment
          </div>
          <form action="/upload_model" method="POST" enctype="multipart/form-data" id="uploadForm" onsubmit="uploadModel(event)">
            <div class="upload-zone" id="upload-zone">
              <svg viewBox="0 0 24 24">
                <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
              </svg>
              <span id="file-name-display" style="font-weight: 500;">Drag model_params.json here</span>
              <span style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Or click to browse</span>
              <input type="file" name="model_file" class="file-input" id="model_file" accept=".json" onchange="updateFileName()">
            </div>
            <button type="submit" class="btn" id="deploy-btn" style="margin-top: 15px;" disabled>
              Deploy Model to Edge
            </button>
          </form>
          <div class="deploy-status" id="deploy-status"></div>
        </div>
      </div>
    </div>

    <div class="footer">
      KSEF 2026 Project 6 | Calibration & Edge Inference Portal
    </div>
  </div>

  <script>
    let currentDataPoints = [];
    let unit = 'mg/L';
    let i0 = { r: 1, g: 1, b: 1 };
    let activeChannels = { red: true, green: true, blue: true };
    let mouseX = -1;
    let mouseY = -1;

    // Custom rounded rect helper for canvas tooltips
    function drawRoundedRect(ctx, x, y, width, height, radius) {
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + width - radius, y);
      ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
      ctx.lineTo(x + width, y + height - radius);
      ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
      ctx.lineTo(x + radius, y + height);
      ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
    }

    function parseCSV(text) {
      const lines = text.trim().split('\n');
      if (lines.length <= 1) return [];
      const points = [];
      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',');
        if (cols.length < 5) continue;
        points.push({
          conc: parseFloat(cols[0]),
          ambient: parseFloat(cols[1]),
          r: parseFloat(cols[2]),
          g: parseFloat(cols[3]),
          b: parseFloat(cols[4])
        });
      }
      return points;
    }

    function computeAbsorbances(dataPoints) {
      const blank = dataPoints.find(p => p.conc === 0.0);
      let refR = blank ? blank.r : i0.r;
      let refG = blank ? blank.g : i0.g;
      let refB = blank ? blank.b : i0.b;

      return dataPoints.map(p => {
        const absR = p.r > 0 ? -Math.log10(p.r / refR) : 0;
        const absG = p.g > 0 ? -Math.log10(p.g / refG) : 0;
        const absB = p.b > 0 ? -Math.log10(p.b / refB) : 0;
        
        let bestCh = 'Red';
        let maxAbs = Math.max(0, absR);
        if (absG > maxAbs) { maxAbs = absG; bestCh = 'Green'; }
        if (absB > maxAbs) { maxAbs = absB; bestCh = 'Blue'; }

        return {
          conc: p.conc,
          absR: Math.max(0, absR),
          absG: Math.max(0, absG),
          absB: Math.max(0, absB),
          bestCh: bestCh
        };
      });
    }

    function toggleChannel(color) {
      activeChannels[color] = !activeChannels[color];
      document.getElementById(`legend-${color}`).classList.toggle('disabled', !activeChannels[color]);
      drawChart();
    }

    function updateTable(data) {
      const tbody = document.getElementById('data-tbody');
      tbody.innerHTML = '';
      
      if (data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding: 24px;">No calibration standards logged yet.</td></tr>';
        return;
      }
      
      const sorted = [...data].sort((a,b) => a.conc - b.conc);
      sorted.forEach(row => {
        const tr = document.createElement('tr');
        const isBlank = row.conc === 0.0;
        const label = isBlank ? '0.000 (Blank Ref)' : row.conc.toFixed(3);
        
        tr.innerHTML = `
          <td><strong>${label}</strong></td>
          <td style="color:var(--red-ch)">${row.absR.toFixed(4)}</td>
          <td style="color:var(--green-ch)">${row.absG.toFixed(4)}</td>
          <td style="color:var(--blue-ch)">${row.absB.toFixed(4)}</td>
          <td><span class="best-badge best-${row.bestCh.toLowerCase()}">${row.bestCh}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    function drawChart() {
      const canvas = document.getElementById('curveChart');
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;
      
      ctx.clearRect(0, 0, width, height);
      
      // Filter out raw points and check counts
      const validPoints = currentDataPoints.filter(p => !isNaN(p.conc));
      if (validPoints.length === 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.font = '14px Outfit';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Awaiting scans from spectrophotometer...', width / 2, height / 2);
        return;
      }

      const paddingLeft = 55;
      const paddingRight = 20;
      const paddingTop = 20;
      const paddingBottom = 40;
      
      const maxConc = Math.max(...validPoints.map(p => p.conc), 1.0);
      const maxAbs = Math.max(...validPoints.map(p => Math.max(p.absR, p.absG, p.absB)), 0.1);
      
      const xLim = maxConc * 1.15;
      const yLim = maxAbs * 1.15;
      
      // Grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      
      const steps = 5;
      for (let i = 0; i <= steps; i++) {
        const val = (xLim * i) / steps;
        const px = paddingLeft + ((width - paddingLeft - paddingRight) * i) / steps;
        
        ctx.beginPath();
        ctx.moveTo(px, paddingTop);
        ctx.lineTo(px, height - paddingBottom);
        ctx.stroke();
        
        ctx.fillText(val.toFixed(1), px, height - paddingBottom + 8);
      }
      
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= steps; i++) {
        const val = (yLim * i) / steps;
        const py = height - paddingBottom - ((height - paddingTop - paddingBottom) * i) / steps;
        
        ctx.beginPath();
        ctx.moveTo(paddingLeft, py);
        ctx.lineTo(width - paddingRight, py);
        ctx.stroke();
        
        ctx.fillText(val.toFixed(2), paddingLeft - 8, py);
      }
      
      // Labels
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '500 11px Outfit';
      ctx.textAlign = 'center';
      ctx.fillText(`Concentration (${unit})`, paddingLeft + (width - paddingLeft - paddingRight)/2, height - 12);
      
      ctx.save();
      ctx.translate(15, paddingTop + (height - paddingTop - paddingBottom)/2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Absorbance (A)', 0, 0);
      ctx.restore();
      
      const mapX = (x) => paddingLeft + ((width - paddingLeft - paddingRight) * x) / xLim;
      const mapY = (y) => height - paddingBottom - ((height - paddingTop - paddingBottom) * y) / yLim;
      
      const channels = [
        { key: 'absR', color: 'rgba(239, 68, 68, 0.85)', active: activeChannels.red, name: 'Red' },
        { key: 'absG', color: 'rgba(16, 185, 129, 0.85)', active: activeChannels.green, name: 'Green' },
        { key: 'absB', color: 'rgba(59, 130, 246, 0.85)', active: activeChannels.blue, name: 'Blue' }
      ];
      
      const sorted = [...validPoints].sort((a,b) => a.conc - b.conc);
      
      channels.forEach(ch => {
        if (!ch.active) return;
        
        ctx.beginPath();
        ctx.strokeStyle = ch.color;
        ctx.lineWidth = 2.5;
        ctx.shadowBlur = 4;
        ctx.shadowColor = ch.color;
        
        sorted.forEach((p, idx) => {
          const px = mapX(p.conc);
          const py = mapY(p[ch.key]);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.shadowBlur = 0;
        
        sorted.forEach(p => {
          const px = mapX(p.conc);
          const py = mapY(p[ch.key]);
          
          ctx.beginPath();
          ctx.arc(px, py, 4, 0, 2*Math.PI);
          ctx.fillStyle = '#070b19';
          ctx.fill();
          ctx.strokeStyle = ch.color;
          ctx.lineWidth = 2;
          ctx.stroke();
        });
      });
      
      // Hover detection
      let hover = null;
      let minDist = 16;
      
      if (mouseX >= paddingLeft && mouseX <= width - paddingRight &&
          mouseY >= paddingTop && mouseY <= height - paddingBottom) {
          
          sorted.forEach(p => {
            const px = mapX(p.conc);
            channels.forEach(ch => {
              if (!ch.active) return;
              const py = mapY(p[ch.key]);
              const d = Math.hypot(mouseX - px, mouseY - py);
              if (d < minDist) {
                minDist = d;
                hover = { p, px, py, chColor: ch.color };
              }
            });
          });
      }
      
      if (hover) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.setLineDash([4, 4]);
        ctx.moveTo(hover.px, paddingTop);
        ctx.lineTo(hover.px, height - paddingBottom);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.beginPath();
        ctx.arc(hover.px, hover.py, 8, 0, 2*Math.PI);
        ctx.fillStyle = 'rgba(56, 189, 248, 0.2)';
        ctx.fill();
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Tooltip drawing
        const tw = 140;
        const th = 78;
        const tx = Math.min(hover.px + 12, width - tw - 10);
        const ty = Math.max(hover.py - th - 12, 10);
        
        ctx.fillStyle = 'rgba(10, 16, 32, 0.95)';
        ctx.strokeStyle = hover.chColor;
        ctx.lineWidth = 1.5;
        drawRoundedRect(ctx, tx, ty, tw, th, 10);
        ctx.fill();
        ctx.stroke();
        
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 11px Outfit';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(`Conc: ${hover.p.conc.toFixed(3)}`, tx + 12, ty + 12);
        
        ctx.font = '500 10px JetBrains Mono';
        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
        ctx.fillText(`R: ${hover.p.absR.toFixed(4)}`, tx + 12, ty + 28);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.9)';
        ctx.fillText(`G: ${hover.p.absG.toFixed(4)}`, tx + 12, ty + 42);
        ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
        ctx.fillText(`B: ${hover.p.absB.toFixed(4)}`, tx + 12, ty + 56);
      }
    }

    function resizeCanvas() {
      const canvas = document.getElementById('curveChart');
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      
      canvas.style.width = '100%';
      canvas.style.height = '300px';
      
      canvas.width = rect.width * dpr;
      canvas.height = 300 * dpr;
      
      const ctx = canvas.getContext('2d');
      ctx.resetTransform();
      ctx.scale(dpr, dpr);
      
      drawChart();
    }

    async function fetchData() {
      try {
        const statusRes = await fetch('/status');
        if (statusRes.ok) {
          const status = await statusRes.json();
          unit = status.unit || 'mg/L';
          i0 = { r: status.i0_r || 1, g: status.i0_g || 1, b: status.i0_b || 1 };
          
          document.getElementById('model-type').textContent = status.model_loaded ? status.model_type.toUpperCase() : 'NONE (DEFAULT)';
          document.getElementById('model-pill').classList.toggle('active', status.model_loaded);
          document.getElementById('unit-display').textContent = unit;
          document.getElementById('ref-display').textContent = `R:${i0.r.toFixed(0)} G:${i0.g.toFixed(0)} B:${i0.b.toFixed(0)}`;
        }
        
        const csvRes = await fetch('/download_csv');
        if (csvRes.ok) {
          const csvText = await csvRes.text();
          const raw = parseCSV(csvText);
          currentDataPoints = computeAbsorbances(raw);
          updateTable(currentDataPoints);
          drawChart();
        } else {
          currentDataPoints = [];
          updateTable([]);
          drawChart();
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }

    function confirmAndClear() {
      if (confirm("Are you sure you want to clear all calibration data points?")) {
        fetch('/clear_data').then(() => {
          fetchData();
        });
      }
    }

    function initMouseEvents() {
      const canvas = document.getElementById('curveChart');
      canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        mouseX = (e.clientX - rect.left) * (canvas.width / rect.width) / dpr;
        mouseY = (e.clientY - rect.top) * (canvas.height / rect.height) / dpr;
        drawChart();
      });
      
      canvas.addEventListener('mouseleave', () => {
        mouseX = -1;
        mouseY = -1;
        drawChart();
      });
    }

    function initUploader() {
      const zone = document.getElementById('upload-zone');
      const input = document.getElementById('model_file');
      zone.addEventListener('click', () => input.click());
      
      zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
      });
      
      zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
      
      zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
          input.files = e.dataTransfer.files;
          updateFileName();
        }
      });
    }

    function updateFileName() {
      const input = document.getElementById('model_file');
      const label = document.getElementById('file-name-display');
      if (input.files.length > 0) {
        label.textContent = input.files[0].name;
        document.getElementById('deploy-btn').disabled = false;
      }
    }

    async function uploadModel(e) {
      e.preventDefault();
      const form = document.getElementById('uploadForm');
      const formData = new FormData(form);
      const btn = document.getElementById('deploy-btn');
      const status = document.getElementById('deploy-status');
      
      btn.disabled = true;
      status.textContent = 'Uploading parameters to coprocessor flash...';
      status.style.color = 'var(--text-muted)';
      
      try {
        const res = await fetch('/upload_model', {
          method: 'POST',
          body: formData
        });
        if (res.ok) {
          status.textContent = 'Model deployed and active successfully!';
          status.style.color = 'var(--success)';
          setTimeout(() => {
            status.textContent = '';
            fetchData();
          }, 2000);
        } else {
          const txt = await res.text();
          status.textContent = `Deployment Error: ${txt}`;
          status.style.color = 'var(--danger)';
          btn.disabled = false;
        }
      } catch (err) {
        status.textContent = `Network Error: ${err.message}`;
        status.style.color = 'var(--danger)';
        btn.disabled = false;
      }
    }

    // Initialize
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('DOMContentLoaded', () => {
      fetchData();
      initMouseEvents();
      initUploader();
      resizeCanvas();
      // Auto refresh data every 3 seconds (real-time previews!)
      setInterval(fetchData, 3000);
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
  doc["unit"] = UNIT_LABELS[currentUnitIndex];
  doc["i0_r"] = I0[0];
  doc["i0_g"] = I0[1];
  doc["i0_b"] = I0[2];
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
  I0[0] = 1.0; I0[1] = 1.0; I0[2] = 1.0;
  isCalibrated = false;
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
//  UI HELPERS & DRAWING FUNCTIONS
// ═══════════════════════════════════════════════════════════

void changeState(State newState) {
  // Exit actions
  if (currentState == STATE_WIFI_PORTAL && newState != STATE_WIFI_PORTAL) {
    server.stop();
    WiFi.softAPdisconnect(true);
    wifiPortalActive = false;
  }
  
  currentState = newState;
  stateStartTime = millis();
  
  // Enter actions
  if (currentState == STATE_WIFI_PORTAL) {
    WiFi.mode(WIFI_AP);
    WiFi.softAP("Chroma-Scan-AP", "chromascan123");
    server.begin();
    wifiPortalActive = true;
  }
}

void triggerBeep(const char* type) {
  Serial.print("BEEP:");
  Serial.println(type);
}

void beepClick()   { triggerBeep("CLICK"); }
void beepWelcome() { triggerBeep("WELCOME"); }
void beepDone()    { triggerBeep("DONE"); }
void beepError()   { triggerBeep("ERROR"); }

bool readBtnDebounced(int pin) {
  static unsigned long lastPressTime[16] = {0};
  if (digitalRead(pin) == LOW) {
    if (millis() - lastPressTime[pin] > 200) {
      lastPressTime[pin] = millis();
      return true;
    }
  }
  return false;
}

void drawCuvette(int x, int y) {
  display.drawRect(x, y, 10, 20, SSD1306_WHITE);
  display.fillRect(x + 1, y + 8, 8, 11, SSD1306_WHITE);
  display.drawFastHLine(x - 2, y, 14, SSD1306_WHITE);
  display.drawFastHLine(x - 12, y + 10, 10, SSD1306_WHITE);
  display.drawLine(x - 4, y + 8, x - 2, y + 10, SSD1306_WHITE);
  display.drawLine(x - 4, y + 12, x - 2, y + 10, SSD1306_WHITE);
}

void drawCalibrateIcon(int x, int y) {
  display.drawRect(x + 10, y + 2, 12, 28, SSD1306_WHITE);
  display.drawFastHLine(x + 8, y + 2, 16, SSD1306_WHITE);
  display.fillRect(x + 11, y + 16, 10, 13, SSD1306_WHITE);
  display.drawFastHLine(x, y + 22, 8, SSD1306_WHITE);
  display.drawFastHLine(x + 24, y + 22, 8, SSD1306_WHITE);
}

void drawMeasureIcon(int x, int y) {
  display.drawRect(x + 10, y + 2, 12, 28, SSD1306_WHITE);
  display.drawFastHLine(x + 8, y + 2, 16, SSD1306_WHITE);
  display.fillRect(x + 11, y + 16, 10, 13, SSD1306_WHITE);
  display.drawFastHLine(x, y + 20, 32, SSD1306_WHITE);
  display.drawLine(x + 26, y + 12, x + 30, y + 16, SSD1306_WHITE);
  display.drawLine(x + 30, y + 12, x + 26, y + 16, SSD1306_WHITE);
}

void drawAddStdIcon(int x, int y) {
  display.drawFastVLine(x + 4, y + 4, 24, SSD1306_WHITE);
  display.drawFastHLine(x + 4, y + 28, 24, SSD1306_WHITE);
  display.drawLine(x + 8, y + 24, x + 24, y + 8, SSD1306_WHITE);
  display.fillCircle(x + 8, y + 24, 2, SSD1306_WHITE);
  display.fillCircle(x + 16, y + 16, 2, SSD1306_WHITE);
  display.fillCircle(x + 24, y + 8, 2, SSD1306_WHITE);
  display.drawFastHLine(x + 24, y + 2, 6, SSD1306_WHITE);
  display.drawFastVLine(x + 27, y - 1, 6, SSD1306_WHITE);
}

void drawWiFiIcon(int x, int y) {
  display.fillCircle(x + 16, y + 28, 2, SSD1306_WHITE);
  display.drawCircleHelper(x + 16, y + 28, 7, 1, SSD1306_WHITE);
  display.drawCircleHelper(x + 16, y + 28, 14, 1, SSD1306_WHITE);
  display.drawCircleHelper(x + 16, y + 28, 21, 1, SSD1306_WHITE);
}

void drawSettingsIcon(int x, int y) {
  display.drawCircle(x + 16, y + 16, 8, SSD1306_WHITE);
  display.drawCircle(x + 16, y + 16, 3, SSD1306_WHITE);
  display.drawFastVLine(x + 16, y + 4, 4, SSD1306_WHITE);
  display.drawFastVLine(x + 16, y + 24, 4, SSD1306_WHITE);
  display.drawFastHLine(x + 4, y + 16, 4, SSD1306_WHITE);
  display.drawFastHLine(x + 24, y + 16, 4, SSD1306_WHITE);
  display.drawLine(x + 7, y + 7, x + 10, y + 10, SSD1306_WHITE);
  display.drawLine(x + 22, y + 22, x + 25, y + 25, SSD1306_WHITE);
  display.drawLine(x + 25, y + 7, x + 22, y + 10, SSD1306_WHITE);
  display.drawLine(x + 10, y + 22, x + 7, y + 25, SSD1306_WHITE);
}

void drawBottomBar(const char* left, const char* right) {
  display.drawFastHLine(0, 52, 128, SSD1306_WHITE);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  if (left) {
    display.setCursor(2, 55);
    display.print(left);
  }
  if (right) {
    int len = strlen(right);
    display.setCursor(128 - len * 6 - 2, 55);
    display.print(right);
  }
}

void drawScanProgress(int step) {
  for (int i = 0; i < 4; i++) {
    int bx = 14 + i * 26;
    if (i < step - 1)
      display.fillRoundRect(bx, 46, 24, 6, 1, SSD1306_WHITE);
    else if (i == step - 1) {
      display.drawRoundRect(bx, 46, 24, 6, 1, SSD1306_WHITE);
      int fillW = (int)((long)scanSampleCount * 22 / NUM_SAMPLES);
      if (fillW > 0) display.fillRect(bx + 1, 47, fillW, 4, SSD1306_WHITE);
    } else
      display.drawRoundRect(bx, 46, 24, 6, 1, SSD1306_WHITE);
  }
  display.setCursor(108, 46);
  display.print(std::min(step, 4));
  display.print(F("/4"));
}

void drawSpinningMolecule(int x, int y, int step) {
  display.drawCircle(x, y, 6, SSD1306_WHITE);
  
  int idx1 = step % 8;
  int dx1 = (int8_t)pgm_read_byte(&MOLECULE_DX[idx1]);
  int dy1 = (int8_t)pgm_read_byte(&MOLECULE_DY[idx1]);
  display.drawLine(x, y, x + dx1, y + dy1, SSD1306_WHITE);
  display.fillCircle(x + dx1, y + dy1, 3, SSD1306_WHITE);
  
  int idx2 = (step + 4) % 8;
  int dx2 = (int8_t)pgm_read_byte(&MOLECULE_DX[idx2]);
  int dy2 = (int8_t)pgm_read_byte(&MOLECULE_DY[idx2]);
  display.drawLine(x, y, x + dx2, y + dy2, SSD1306_WHITE);
  display.fillCircle(x + dx2, y + dy2, 3, SSD1306_WHITE);
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

void drawSmallCheck(int x, int y) {
  display.drawLine(x, y + 2, x + 2, y + 4, SSD1306_WHITE);
  display.drawLine(x + 2, y + 4, x + 6, y, SSD1306_WHITE);
}

void updateAnimations() {
  unsigned long now = millis();
  if (now - lastFrameTime < 40) return;
  lastFrameTime = now;
  
  // Smooth carousel menu index interpolation
  float diff = menuIndex - smoothMenuIndex;
  if (abs(diff) > 0.01) {
    smoothMenuIndex += diff * 0.25;
  } else {
    smoothMenuIndex = menuIndex;
  }
  
  static int moleculeTimer = 0;
  moleculeTimer++;
  if (moleculeTimer >= 2) {
    moleculeTimer = 0;
    moleculeStep = (moleculeStep + 1) % 8;
  }
  
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
    case STATE_WELCOME: {
      unsigned long elapsed = millis() - stateStartTime;
      display.setTextColor(SSD1306_WHITE);
      
      if (elapsed < 2000) {
        // Welcome Screen 1: Company Name
        display.drawRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, SSD1306_WHITE);
        display.setTextSize(2);
        display.setCursor((SCREEN_WIDTH - 96) / 2, 12);
        display.print(F("Galvaniy"));
        
        display.setTextSize(1);
        display.setCursor((SCREEN_WIDTH - 72) / 2, 38);
        display.print(F("Technologies"));
      } 
      else if (elapsed < 4000) {
        // Welcome Screen 2: Product Name
        display.drawRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, SSD1306_WHITE);
        display.setTextSize(2);
        display.setCursor((SCREEN_WIDTH - 72) / 2, 10);
        display.print(F("Chroma"));
        display.setCursor((SCREEN_WIDTH - 48) / 2, 30);
        display.print(F("Scan"));
        
        display.setTextSize(1);
        display.setCursor((SCREEN_WIDTH - 24) / 2, 52);
        display.print(F("v1.0"));
      } 
      else {
        // Welcome Screen 3: Ready prompt
        display.drawRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, SSD1306_WHITE);
        
        display.setTextSize(1);
        display.setCursor((SCREEN_WIDTH - 96) / 2, 16);
        display.print(F("Chroma-Scan Ready"));
        
        display.setTextSize(2);
        display.setCursor((SCREEN_WIDTH - 96) / 2, 32);
        display.print(F("Press OK"));
        
        // Blink indicator
        if ((millis() / 500) % 2 == 0) {
          display.setTextSize(1);
          display.setCursor((SCREEN_WIDTH - 66) / 2, 52);
          display.print(F("to begin..."));
        }
      }
      break;
    }

    case STATE_MAIN_MENU:
      // Sleek title bar with status indicators
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("CHROMA-SCAN"));
      
      // Calibrated status dot/ring in header
      if (isCalibrated) {
        display.fillCircle(118, 5, 2, SSD1306_WHITE);
        display.setCursor(96, 2);
        display.print(F("CAL"));
      } else {
        display.drawCircle(118, 5, 2, SSD1306_WHITE);
        display.setCursor(96, 2);
        display.print(F("UNCAL"));
      }
      
      // Draw items
      for (int idx = 0; idx < 5; idx++) {
        int xOffset = 48 + (int)round((idx - smoothMenuIndex) * 128.0);
        if (xOffset < -32 || xOffset > 128) continue;
        
        // Draw Icon
        if (idx == 0) drawCalibrateIcon(xOffset, 14);
        else if (idx == 1) drawMeasureIcon(xOffset, 14);
        else if (idx == 2) drawAddStdIcon(xOffset, 14);
        else if (idx == 3) drawWiFiIcon(xOffset, 14);
        else if (idx == 4) drawSettingsIcon(xOffset, 14);
        
        // Draw Text Label centered below the icon
        display.setTextColor(SSD1306_WHITE);
        if (idx == 0) {
          display.setCursor(xOffset + 16 - 45, 43);
          display.print(F("Calibrate Blank"));
        } else if (idx == 1) {
          if (!isCalibrated) {
            display.setCursor(xOffset + 16 - 54, 43);
            display.print(F("Measure Sample [!]"));
          } else {
            display.setCursor(xOffset + 16 - 42, 43);
            display.print(F("Measure Sample"));
          }
        } else if (idx == 2) {
          if (!isCalibrated) {
            display.setCursor(xOffset + 16 - 48, 43);
            display.print(F("Add Standard [!]"));
          } else {
            display.setCursor(xOffset + 16 - 36, 43);
            display.print(F("Add Standard"));
          }
        } else if (idx == 3) {
          display.setCursor(xOffset + 16 - 42, 43);
          display.print(F("WiFi AP Portal"));
        } else if (idx == 4) {
          display.setCursor(xOffset + 16 - 24, 43);
          display.print(F("Settings"));
        }
      }
      
      // Left/Right Carousel indicators (arrows)
      if (menuIndex > 0) {
        display.fillTriangle(4, 26, 8, 23, 8, 29, SSD1306_WHITE);
      }
      if (menuIndex < 4) {
        display.fillTriangle(123, 26, 119, 23, 119, 29, SSD1306_WHITE);
      }
      
      drawBottomBar("BACK", "OK Select");
      break;

    case STATE_CAL_BLANK_PROMPT:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("CALIBRATE SYSTEM"));
      display.setCursor(4, 18);
      display.print(F("Insert BLANK cuvette"));
      display.setCursor(4, 30);
      display.print(F("with pure solvent."));
      drawBottomBar("BACK", "OK Go");
      break;

    case STATE_CAL_BLANK_SCANNING:
    case STATE_MEASURE_SAMPLE_SCANNING:
    case STATE_ADD_STD_SCANNING:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      if (currentState == STATE_CAL_BLANK_SCANNING)
        display.print(F("CALIBRATING BLANK"));
      else if (currentState == STATE_ADD_STD_SCANNING)
        display.print(F("SCANNING STANDARD"));
      else
        display.print(F("MEASURING SAMPLE"));
      drawScanningLaser(10, 18, 18, 24, animLaserY % 20);
      display.setCursor(38, 18);
      display.print(F("Channel:"));
      display.setTextSize(2);
      display.setCursor(38, 28);
      if (scanStep >= 1 && scanStep <= 4) {
        if (scanStep == 1) display.print(F("Ambient"));
        else if (scanStep == 2) display.print(F("Red"));
        else if (scanStep == 3) display.print(F("Green"));
        else if (scanStep == 4) display.print(F("Blue"));
      } else {
        display.print(F("Done"));
      }
      display.setTextSize(1);
      drawScanProgress(scanStep);
      break;

    case STATE_CAL_BLANK_DONE:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("CALIBRATION OK"));
      drawSmallCheck(110, 3);
      display.setCursor(4, 16);
      display.print(F("Baseline Set!"));
      display.setCursor(4, 28);
      display.print(F("R: ")); display.print(I0[0], 0);
      display.setCursor(66, 28);
      display.print(F("G: ")); display.print(I0[1], 0);
      display.setCursor(4, 40);
      display.print(F("B: ")); display.print(I0[2], 0);
      display.setCursor(66, 40);
      display.print(F("Amb: ")); display.print(rawAmbient, 0);
      drawBottomBar(NULL, "OK Next");
      break;

    case STATE_ADD_STD_MENU:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("ADD STANDARD POINT"));
      for (int i = 0; i < 2; i++) {
        int y = 20 + i * 12;
        display.setTextColor(SSD1306_WHITE);
        if (i == menuIndex) {
          display.drawRoundRect(2, y - 1, 124, 11, 2, SSD1306_WHITE);
        }
        display.setCursor(6, y + 1);
        if (i == 0) display.print(F("Preset Values"));
        else if (i == 1) display.print(F("Manual Entry"));
      }
      drawBottomBar("BACK", "OK Next");
      break;

    case STATE_ADD_STD_PRESET:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("CHOOSE PRESET CONC"));
      {
        int prevIdx = (presetIndex - 1 + NUM_PRESETS) % NUM_PRESETS;
        display.setCursor(24, 18);
        display.print(PRESETS[prevIdx], 2);
      }
      display.drawRoundRect(12, 28, 104, 20, 3, SSD1306_WHITE);
      display.setTextSize(2);
      display.setCursor(20, 31);
      display.print(PRESETS[presetIndex], 2);
      display.setTextSize(1);
      display.setCursor(80, 34);
      display.print(UNIT_LABELS[currentUnitIndex]);
      {
        int nextIdx = (presetIndex + 1) % NUM_PRESETS;
        display.setCursor(24, 52);
        display.print(PRESETS[nextIdx], 2);
      }
      drawBottomBar("BACK", "OK Scan");
      break;

    case STATE_ADD_STD_MANUAL:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("MANUAL CONC ENTRY"));
      display.setCursor(4, 16);
      display.print(F("Enter concentration:"));
      display.setTextSize(2);
      display.setCursor(18, 30);
      for (int i = 0; i < 5; i++) {
        if (i == 3) display.print(F("."));
        display.print(concDigits[i]);
      }
      display.setTextSize(1);
      display.setCursor(94, 36);
      display.print(UNIT_LABELS[currentUnitIndex]);
      
      int cursorX;
      cursorX = 18 + concDigitPos * 12;
      if (concDigitPos >= 3) cursorX += 12;
      display.drawFastHLine(cursorX, 46, 10, SSD1306_WHITE);
      
      drawBottomBar("BACK", "OK Next");
      break;

    case STATE_ADD_STD_DONE:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("STANDARD SAVED"));
      drawSmallCheck(110, 3);
      display.setCursor(4, 16);
      display.print(F("Value logged:"));
      display.setCursor(4, 28);
      display.print(manualConc, 2);
      display.print(F(" "));
      display.print(UNIT_LABELS[currentUnitIndex]);
      drawBottomBar(NULL, "OK Next");
      break;

    case STATE_MEASURE_SAMPLE_PROMPT:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("MEASURE SAMPLE"));
      display.setCursor(4, 18);
      display.print(F("Insert cuvette to"));
      display.setCursor(4, 30);
      display.print(F("measure concentration"));
      drawBottomBar("BACK", "OK Go");
      break;

    case STATE_MEASURE_SAMPLE_RESULTS:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("RESULTS"));
      display.setCursor(100, 2);
      display.print(resultPage + 1);
      display.print(F("/"));
      display.print(RESULT_PAGES);
      
      if (resultPage == 0) {
        display.setCursor(4, 16);
        display.print(F("Ch   Absorbance (A)"));
        display.drawFastHLine(4, 25, 120, SSD1306_WHITE);
        for (int i = 0; i < 3; i++) {
          int y = 28 + i * 8;
          display.setCursor(4, y);
          if (i == bestChannel) display.print(F("*"));
          else display.print(F(" "));
          display.print(CHANNEL_NAMES[i]);
          display.setCursor(56, y);
          display.print(absorbance[i], 4);
        }
      } else {
        display.setCursor(4, 16);
        display.print(F("Final Concentration"));
        display.setTextSize(2);
        display.setCursor(4, 26);
        display.print(concentrationResult, 2);
        display.setTextSize(1);
        display.setCursor(76, 32);
        display.print(UNIT_LABELS[currentUnitIndex]);
        display.setCursor(4, 42);
        display.print(F("Best channel: "));
        display.print(CHANNEL_NAMES[bestChannel]);
      }
      drawBottomBar("BACK Menu", "Page");
      break;

    case STATE_WIFI_PORTAL:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("WIFI PORTAL"));
      drawWiFiPulsing(108, 32, 5, wifiPulseStep);
      display.setCursor(4, 18);
      display.print(F("SSID:"));
      display.setCursor(4, 28);
      display.print(F(" Chroma-Scan-AP"));
      display.setCursor(4, 38);
      display.print(F("IP: 192.168.4.1"));
      drawBottomBar("BACK Stop", NULL);
      break;

    case STATE_SETTINGS:
      display.setTextColor(SSD1306_WHITE);
      display.drawFastHLine(0, 12, SCREEN_WIDTH, SSD1306_WHITE);
      display.setCursor(6, 2);
      display.print(F("SETTINGS"));
      for (int i = 0; i < 2; i++) {
        int y = 20 + i * 12;
        display.setTextColor(SSD1306_WHITE);
        if (i == menuIndex) {
          display.drawRoundRect(2, y - 1, 124, 11, 2, SSD1306_WHITE);
          display.fillTriangle(4, y + 1, 4, y + 7, 8, y + 4, SSD1306_WHITE);
        }
        display.setCursor(12, y + 1);
        if (i == 0) {
          display.print(F("Units: "));
          display.print(UNIT_LABELS[currentUnitIndex]);
        } else if (i == 1) {
          display.print(F("Back to Menu"));
        }
      }
      drawBottomBar("BACK", "OK Select");
      break;
  }
}

void startScan(State scanState) {
  scanStep = 1;
  scanSampleCount = 0;
  changeState(scanState);
  Serial.println("SCAN");
}

void onScanComplete() {
  if (currentState == STATE_CAL_BLANK_SCANNING) {
    I0[0] = rawR;
    I0[1] = rawG;
    I0[2] = rawB;
    isCalibrated = true;
    saveStandard(0.0, rawAmbient, rawR, rawG, rawB);
    beepDone();
    changeState(STATE_CAL_BLANK_DONE);
  }
  else if (currentState == STATE_ADD_STD_SCANNING) {
    absorbance[0] = calculateAbsorbance(I0[0], rawR);
    absorbance[1] = calculateAbsorbance(I0[1], rawG);
    absorbance[2] = calculateAbsorbance(I0[2], rawB);
    
    bestChannel = 0;
    float maxAbs = absorbance[0];
    if (absorbance[1] > maxAbs) { maxAbs = absorbance[1]; bestChannel = 1; }
    if (absorbance[2] > maxAbs) { maxAbs = absorbance[2]; bestChannel = 2; }
    
    saveStandard(manualConc, rawAmbient, rawR, rawG, rawB);
    beepDone();
    changeState(STATE_ADD_STD_DONE);
  }
  else if (currentState == STATE_MEASURE_SAMPLE_SCANNING) {
    absorbance[0] = calculateAbsorbance(I0[0], rawR);
    absorbance[1] = calculateAbsorbance(I0[1], rawG);
    absorbance[2] = calculateAbsorbance(I0[2], rawB);
    
    bestChannel = 0;
    float maxAbs = absorbance[0];
    if (absorbance[1] > maxAbs) { maxAbs = absorbance[1]; bestChannel = 1; }
    if (absorbance[2] > maxAbs) { maxAbs = absorbance[2]; bestChannel = 2; }
    
    concentrationResult = runModelInference(absorbance[0], absorbance[1], absorbance[2], rawAmbient);
    beepDone();
    changeState(STATE_MEASURE_SAMPLE_RESULTS);
    resultPage = 0;
  }
}

void handleButtonPresses(bool up, bool down, bool ok, bool back) {
  switch (currentState) {
    case STATE_WELCOME:
      if (ok) {
        beepClick();
        changeState(STATE_MAIN_MENU);
        menuIndex = 0;
      }
      break;
      
    case STATE_MAIN_MENU:
      if (up) {
        beepClick();
        menuIndex = (menuIndex - 1 + 5) % 5;
      } else if (down) {
        beepClick();
        menuIndex = (menuIndex + 1) % 5;
      } else if (ok) {
        if (menuIndex == 0) {
          beepClick();
          changeState(STATE_CAL_BLANK_PROMPT);
        } else if (menuIndex == 1) {
          if (!isCalibrated) {
            beepError();
          } else {
            beepClick();
            changeState(STATE_MEASURE_SAMPLE_PROMPT);
          }
        } else if (menuIndex == 2) {
          if (!isCalibrated) {
            beepError();
          } else {
            beepClick();
            changeState(STATE_ADD_STD_MENU);
            menuIndex = 0;
          }
        } else if (menuIndex == 3) {
          beepClick();
          changeState(STATE_WIFI_PORTAL);
        } else if (menuIndex == 4) {
          beepClick();
          changeState(STATE_SETTINGS);
          menuIndex = 0;
        }
      } else if (back) {
        beepClick();
        changeState(STATE_WELCOME);
      }
      break;
      
    case STATE_CAL_BLANK_PROMPT:
      if (back) {
        beepClick();
        changeState(STATE_MAIN_MENU);
      } else if (ok) {
        beepClick();
        startScan(STATE_CAL_BLANK_SCANNING);
      }
      break;
      
    case STATE_CAL_BLANK_DONE:
      if (ok || back) {
        beepClick();
        changeState(STATE_MAIN_MENU);
      }
      break;
      
    case STATE_ADD_STD_MENU:
      if (up || down) {
        beepClick();
        menuIndex = (menuIndex == 0) ? 1 : 0;
      } else if (back) {
        beepClick();
        changeState(STATE_MAIN_MENU);
      } else if (ok) {
        beepClick();
        if (menuIndex == 0) {
          changeState(STATE_ADD_STD_PRESET);
          presetIndex = 0;
        } else {
          changeState(STATE_ADD_STD_MANUAL);
          concDigitPos = 0;
          for (int i = 0; i < 5; i++) concDigits[i] = 0;
        }
      }
      break;
      
    case STATE_ADD_STD_PRESET:
      if (up) {
        beepClick();
        presetIndex = (presetIndex - 1 + NUM_PRESETS) % NUM_PRESETS;
      } else if (down) {
        beepClick();
        presetIndex = (presetIndex + 1) % NUM_PRESETS;
      } else if (back) {
        beepClick();
        changeState(STATE_ADD_STD_MENU);
        menuIndex = 0;
      } else if (ok) {
        beepClick();
        manualConc = PRESETS[presetIndex];
        startScan(STATE_ADD_STD_SCANNING);
      }
      break;
      
    case STATE_ADD_STD_MANUAL:
      if (up) {
        beepClick();
        concDigits[concDigitPos] = (concDigits[concDigitPos] + 1) % 10;
      } else if (down) {
        beepClick();
        concDigits[concDigitPos] = (concDigits[concDigitPos] + 9) % 10;
      } else if (back) {
        beepClick();
        if (concDigitPos > 0) {
          concDigitPos--;
        } else {
          changeState(STATE_ADD_STD_MENU);
          menuIndex = 0;
        }
      } else if (ok) {
        beepClick();
        if (concDigitPos < 4) {
          concDigitPos++;
        } else {
          manualConc = concDigits[0] * 100.0 + concDigits[1] * 10.0 + concDigits[2] + concDigits[3] * 0.1 + concDigits[4] * 0.01;
          startScan(STATE_ADD_STD_SCANNING);
        }
      }
      break;
      
    case STATE_ADD_STD_DONE:
      if (ok || back) {
        beepClick();
        changeState(STATE_MAIN_MENU);
      }
      break;
      
    case STATE_MEASURE_SAMPLE_PROMPT:
      if (back) {
        beepClick();
        changeState(STATE_MAIN_MENU);
      } else if (ok) {
        beepClick();
        startScan(STATE_MEASURE_SAMPLE_SCANNING);
      }
      break;
      
    case STATE_MEASURE_SAMPLE_RESULTS:
      if (up || down) {
        beepClick();
        resultPage = (resultPage + 1) % RESULT_PAGES;
      } else if (ok || back) {
        beepClick();
        changeState(STATE_MAIN_MENU);
      }
      break;
      
    case STATE_WIFI_PORTAL:
      if (ok || back) {
        beepClick();
        changeState(STATE_MAIN_MENU);
      }
      break;
      
    case STATE_SETTINGS:
      if (up || down) {
        beepClick();
        menuIndex = (menuIndex == 0) ? 1 : 0;
      } else if (back) {
        beepClick();
        changeState(STATE_MAIN_MENU);
      } else if (ok) {
        beepClick();
        if (menuIndex == 0) {
          currentUnitIndex = (currentUnitIndex + 1) % 3;
        } else {
          changeState(STATE_MAIN_MENU);
        }
      }
      break;
      
    default:
      break;
  }
}

// ═══════════════════════════════════════════════════════════
//  UART COMMANDS INTERPRETATION
// ═══════════════════════════════════════════════════════════

void handleCommand(char* cmd) {
  digitalWrite(STATUS_LED_PIN, LOW);
  delay(10);
  digitalWrite(STATUS_LED_PIN, HIGH);

  if (strncmp(cmd, "SCAN_PROGRESS:", 14) == 0) {
    char* p = cmd + 14;
    scanStep = atoi(p);
    p = strchr(p, ',');
    if (p) {
      scanSampleCount = atoi(p + 1);
    }
  }
  else if (strncmp(cmd, "SCAN_DATA:", 10) == 0) {
    char* p = cmd + 10;
    rawAmbient = strtof(p, &p);
    if (*p == ',') p++;
    rawR = strtof(p, &p);
    if (*p == ',') p++;
    rawG = strtof(p, &p);
    if (*p == ',') p++;
    rawB = strtof(p, &p);
    
    onScanComplete();
  }
}

// ═══════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════

void setup() {
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);
  
  pinMode(BTN_UP, INPUT_PULLUP);
  pinMode(BTN_DOWN, INPUT_PULLUP);
  pinMode(BTN_OK, INPUT_PULLUP);
  pinMode(BTN_BACK, INPUT_PULLUP);

  Serial.begin(9600); 
  delay(100);
  Serial.println("DEBUG: Chroma-Scan Coprocessor System Booting...");

  // OLED Init
  Serial.println("DEBUG: Running I2C bus recovery routine...");
  pinMode(5, OUTPUT);
  pinMode(4, INPUT_PULLUP);
  for (int i = 0; i < 10; i++) {
    digitalWrite(5, HIGH);
    delayMicroseconds(5);
    digitalWrite(5, LOW);
    delayMicroseconds(5);
  }
  
  Serial.println("DEBUG: Initialising I2C with Wire.begin(4, 5)...");
  Wire.begin(4, 5);
  Wire.setClock(100000); // 100 kHz standard rate for high signal integrity over jumpers
  
  Serial.println("DEBUG: Scanning I2C bus...");
  byte error, address;
  int nDevices = 0;
  for(address = 1; address < 127; address++ ) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("DEBUG: I2C device found at address 0x");
      if (address<16) Serial.print("0");
      Serial.print(address,HEX);
      Serial.println("  !");
      nDevices++;
    }
    else if (error==4) {
      Serial.print("DEBUG: Unknown error at address 0x");
      if (address<16) Serial.print("0");
      Serial.println(address,HEX);
    }
  }
  if (nDevices == 0)
    Serial.println("DEBUG: No I2C devices found on bus!");
  else
    Serial.println("DEBUG: I2C scan complete.");
  
  Serial.println("DEBUG: Calling display.begin...");
  if(!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR, true, false)) {
    Serial.println("DEBUG: SSD1306 allocation failed! Check I2C address (0x3C/0x3D) or SDA/SCL wire connections.");
    // Blink rapid warning pattern on onboard LED to signal display error
    for(int i = 0; i < 20; i++) {
      digitalWrite(STATUS_LED_PIN, LOW); delay(50);
      digitalWrite(STATUS_LED_PIN, HIGH); delay(50);
    }
  } else {
    Serial.println("DEBUG: SSD1306 display initialized successfully.");
    Wire.setClock(100000); // Override 400kHz set by display.begin() back to robust 100kHz
    display.clearDisplay();
    display.display();
    
    // Force contrast and display on state in case of low-voltage/charge-pump issues
    display.ssd1306_command(SSD1306_DISPLAYON);
    display.ssd1306_command(SSD1306_SETCONTRAST);
    display.ssd1306_command(255);
  }

  // Filesystem Init
  if (LittleFS.begin()) {
    Serial.println("DEBUG: LittleFS FileSystem Mounted Successfully.");
    loadModelParams();
  } else {
    Serial.println("DEBUG: LittleFS Mount Failed! Attempting format...");
    if (LittleFS.format()) {
      Serial.println("DEBUG: LittleFS formatted successfully.");
      if (LittleFS.begin()) {
        Serial.println("DEBUG: LittleFS Mounted after format.");
        loadModelParams();
      }
    }
  }
  
  server.on("/", HTTP_GET, handleRoot);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/download_csv", HTTP_GET, handleDownloadCsv);
  server.on("/clear_data", HTTP_GET, handleClearData);
  server.on("/upload_model", HTTP_POST, handleRoot, handleUploadModel);

  beepWelcome();
  stateStartTime = millis();
}

// ═══════════════════════════════════════════════════════════
//  LOOP
// ═══════════════════════════════════════════════════════════

void loop() {
  if (wifiPortalActive) {
    server.handleClient();
  }
  
  // Non-blocking status LED indicator
  static unsigned long lastLedBlink = 0;
  static bool ledState = false;
  unsigned long currentMillis = millis();



  if (wifiPortalActive) {
    if (currentMillis - lastLedBlink >= 100) {
      lastLedBlink = currentMillis;
      ledState = !ledState;
      digitalWrite(STATUS_LED_PIN, ledState ? LOW : HIGH);
    }
  } else {
    unsigned long elapsed = currentMillis - lastLedBlink;
    if (ledState) {
      if (elapsed >= 100) {
        ledState = false;
        digitalWrite(STATUS_LED_PIN, HIGH);
        lastLedBlink = currentMillis;
      }
    } else {
      if (elapsed >= 900) {
        ledState = true;
        digitalWrite(STATUS_LED_PIN, LOW);
        lastLedBlink = currentMillis;
      }
    }
  }
  
  // Update screen and animation steps at a controlled 20 FPS (every 50ms) to avoid flooding I2C bus
  static unsigned long lastDisplayUpdate = 0;
  if (currentMillis - lastDisplayUpdate >= 50) {
    lastDisplayUpdate = currentMillis;
    updateAnimations();
    renderState();
    display.display();
  }
  
  // Read debounced buttons
  bool upPressed    = readBtnDebounced(BTN_UP);
  bool downPressed  = readBtnDebounced(BTN_DOWN);
  bool okPressed    = readBtnDebounced(BTN_OK);
  bool backPressed  = readBtnDebounced(BTN_BACK);
  
  if (upPressed || downPressed || okPressed || backPressed) {
    handleButtonPresses(upPressed, downPressed, okPressed, backPressed);
  }
  
  // Listen for progress/data from Arduino Nano slave
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialLen > 0) {
        serialBuf[serialLen] = '\0';
        char* cmdStart = serialBuf;
        while (*cmdStart == ' ') cmdStart++;
        if (*cmdStart != '\0') {
          handleCommand(cmdStart);
        }
        serialLen = 0;
      }
    } else if (serialLen < sizeof(serialBuf) - 1) {
      serialBuf[serialLen++] = c;
    }
  }
}
