/*
 * ============================================================
 *  CHROMA-SCAN  —  Digital RGB Spectrophotometer
 * ============================================================
 *  A low-cost alternative to commercial spectrophotometers.
 *  Uses Beer-Lambert Law (A = -log10(I/I0)) with an RGB LED
 *  and LDR to perform colorimetric analysis.
 *
 *  Hardware:
 *    - Arduino Nano
 *    - I2C LCD 16x2 (addr 0x27)
 *    - Common Anode RGB LED (LOW=ON, HIGH=OFF)
 *      - Red   → pin 2
 *      - Blue  → pin 3
 *      - Green → pin 4
 *    - Buzzer+LED → pin 13
 *    - LDR + 1kΩ voltage divider → A0
 *    - Push button → pin 12 (active HIGH)
 *
 *  KSEF 2026  |  Project 6
 * ============================================================
 */

#include <LiquidCrystal_I2C.h>
#include <Wire.h>
#include <math.h>

// ─── Pin Definitions (PWM pins for smooth fading) ───────────
#define RED_LED 5
#define BLUE_LED 3
#define GREEN_LED 6
#define BUZZER 13
#define LDR_PIN A0
#define BUTTON_PIN 12

// ─── Animation Constants ────────────────────────────────────
#define ANIM_STEP_MS 15 // ms between animation frames
#define ANIM_SPEED 2    // color increment per step

// ─── Common Anode Helpers (inverted logic) ──────────────────
#define LED_ON LOW
#define LED_OFF HIGH

// ─── Timing Constants ───────────────────────────────────────
#define DEBOUNCE_MS 50
#define SHORT_PRESS_MS 50   // minimum for valid press
#define LONG_PRESS_MS 800   // threshold for long press
#define SETTLE_TIME_MS 300  // LED settle before reading
#define NUM_SAMPLES 30      // analog reads averaged
#define SAMPLE_DELAY_MS 5   // delay between analog reads
#define LCD_CYCLE_DELAY 200 // min ms between LCD updates

// ─── Calibration Limits ─────────────────────────────────────
#define MAX_STANDARDS 8

// ─── Predefined Standard Concentrations (mg/L) ─────────────
const float PRESETS[] = {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0};
#define NUM_PRESETS 8

// ─── LCD ────────────────────────────────────────────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ─── Unit System ────────────────────────────────────────────
enum UnitType { UNIT_MG_L, UNIT_MOL_L, UNIT_PPM, UNIT_COUNT };
const char *unitLabels[] = {"mg/L", "mol/L", "ppm"};
UnitType currentUnit = UNIT_MG_L;

// ─── State Machine ──────────────────────────────────────────
enum State {
  STATE_WELCOME,
  STATE_SETTINGS,
  STATE_CAL_BLANK,
  STATE_CAL_MEASURING,
  STATE_CAL_DONE,
  STATE_STD_MENU,
  STATE_STD_MODE_SELECT, // choose manual or preset
  STATE_SET_CONC_MANUAL,
  STATE_SET_CONC_PRESET,
  STATE_MEASURE_STD,
  STATE_STD_DONE,
  STATE_MEASURE_SAMPLE,
  STATE_SAMPLING,
  STATE_RESULTS
};

State currentState = STATE_WELCOME;

// ─── Button ─────────────────────────────────────────────────
enum ButtonEvent { BTN_NONE, BTN_SHORT, BTN_LONG };

// ─── Measurement Data ───────────────────────────────────────
float I0[3] = {0, 0, 0};         // baseline (blank) for R, G, B
float sampleI[3] = {0, 0, 0};    // sample readings
float absorbance[3] = {0, 0, 0}; // absorbance for R, G, B
const char *channelNames[] = {"Red", "Green", "Blue"};
int ledPins[] = {RED_LED, GREEN_LED, BLUE_LED};

// ─── Calibration Curve ──────────────────────────────────────
float stdConcentrations[MAX_STANDARDS];
float stdAbsorbances[MAX_STANDARDS];
int numStandards = 0;
float slopeM = 0; // A = m*C + b
float interceptB = 0;
bool curveValid = false;

// ─── Results ────────────────────────────────────────────────
int bestChannel = 0; // index of highest absorbance
float concentration = 0;
int resultPage = 0; // cycling index (0-4)
#define RESULT_PAGES 5

// ─── Concentration Entry ────────────────────────────────────
// Manual: digit-by-digit for XXX.XX
float manualConc = 0;
int concDigits[5] = {0, 0, 0, 0, 0}; // 3 integer + 2 decimal digits
int concDigitPos = 0;

// Preset
int presetIndex = 0;

// ─── Std entry mode ─────────────────────────────────────────
int stdEntryMode = 0; // 0 = manual, 1 = preset

// ═══════════════════════════════════════════════════════════
//  BUZZER  FEEDBACK
// ═══════════════════════════════════════════════════════════

void beepClick() {
  // Soft click for every button press
  tone(BUZZER, 4000, 20);
  delay(25);
  noTone(BUZZER);
}

void beepWelcome() {
  // Rising 3-note chime
  tone(BUZZER, 523, 100);
  delay(120); // C5
  tone(BUZZER, 659, 100);
  delay(120); // E5
  tone(BUZZER, 784, 150);
  delay(170); // G5
  noTone(BUZZER);
}

void beepCalDone() {
  // Two short high beeps
  tone(BUZZER, 2000, 80);
  delay(120);
  tone(BUZZER, 2000, 80);
  delay(100);
  noTone(BUZZER);
}

void beepStdSaved() {
  // Single medium beep
  tone(BUZZER, 1200, 150);
  delay(170);
  noTone(BUZZER);
}

void beepMeasureDone() {
  // Descending 3-note
  tone(BUZZER, 784, 100);
  delay(120); // G5
  tone(BUZZER, 659, 100);
  delay(120); // E5
  tone(BUZZER, 523, 150);
  delay(170); // C5
  noTone(BUZZER);
}

void beepError() {
  // Rapid low buzz
  for (int i = 0; i < 4; i++) {
    tone(BUZZER, 200, 60);
    delay(80);
  }
  noTone(BUZZER);
}

void beepSettings() {
  // Two-tone notification
  tone(BUZZER, 880, 60);
  delay(80);
  tone(BUZZER, 1100, 60);
  delay(80);
  noTone(BUZZER);
}

// ═══════════════════════════════════════════════════════════
//  LED  HELPERS  (Common Anode)
// ═══════════════════════════════════════════════════════════

void allLedsOff() {
  // For PWM pins, 255 = fully OFF on common anode
  analogWrite(RED_LED, 255);
  analogWrite(GREEN_LED, 255);
  analogWrite(BLUE_LED, 255);
}

void ledOn(int pin) {
  analogWrite(pin, 0); // 0 = full brightness on common anode
}

void ledOff(int pin) {
  analogWrite(pin, 255); // 255 = off on common anode
}

/*
 * setRGB()
 * Sets RGB color with 0-255 brightness values.
 * Inverts for common anode: 0 brightness = 255 PWM (off),
 * 255 brightness = 0 PWM (full on).
 */
void setRGB(byte r, byte g, byte b) {
  analogWrite(RED_LED, 255 - r);
  analogWrite(GREEN_LED, 255 - g);
  analogWrite(BLUE_LED, 255 - b);
}

// ═══════════════════════════════════════════════════════════
//  RAINBOW  ANIMATION  (non-blocking)
// ═══════════════════════════════════════════════════════════

unsigned int animHue = 0; // 0-1535 for full rainbow cycle
unsigned long lastAnimStep = 0;

/*
 * hueToRGB()
 * Converts a hue value (0-1535) to smooth RGB.
 * 6 phases: R→Y→G→C→B→M→R
 */
void hueToRGB(unsigned int hue, byte &r, byte &g, byte &b) {
  unsigned int phase = hue / 256;
  byte fade = hue % 256;

  switch (phase) {
  case 0:
    r = 255;
    g = fade;
    b = 0;
    break; // Red → Yellow
  case 1:
    r = 255 - fade;
    g = 255;
    b = 0;
    break; // Yellow → Green
  case 2:
    r = 0;
    g = 255;
    b = fade;
    break; // Green → Cyan
  case 3:
    r = 0;
    g = 255 - fade;
    b = 255;
    break; // Cyan → Blue
  case 4:
    r = fade;
    g = 0;
    b = 255;
    break; // Blue → Magenta
  case 5:
    r = 255;
    g = 0;
    b = 255 - fade;
    break; // Magenta → Red
  default:
    r = 255;
    g = 0;
    b = 0;
    break;
  }
}

/*
 * updateAnimation()
 * Call every loop iteration. Advances the rainbow
 * smoothly if enough time has passed.
 */
void updateAnimation() {
  unsigned long now = millis();
  if (now - lastAnimStep < ANIM_STEP_MS)
    return;
  lastAnimStep = now;

  byte r, g, b;
  hueToRGB(animHue, r, g, b);
  setRGB(r, g, b);

  animHue += ANIM_SPEED;
  if (animHue >= 1536)
    animHue = 0;
}

/*
 * stopAnimation()
 * Instantly stops the animation and turns off all LEDs.
 */
void stopAnimation() { allLedsOff(); }

// ═══════════════════════════════════════════════════════════
//  LCD  HELPERS
// ═══════════════════════════════════════════════════════════

void lcdPrint(const char *line1, const char *line2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);
}

void lcdPrintLine(int row, const char *text) {
  lcd.setCursor(0, row);
  lcd.print("                ");
  lcd.setCursor(0, row);
  lcd.print(text);
}

// ═══════════════════════════════════════════════════════════
//  BUTTON  HANDLING
// ═══════════════════════════════════════════════════════════

ButtonEvent readButton() {
  static bool lastState = LOW;
  static unsigned long pressStart = 0;
  static bool pressed = false;
  static bool longConfirmed = false;

  bool current = digitalRead(BUTTON_PIN);

  // Button just pressed
  if (current == HIGH && lastState == LOW) {
    pressStart = millis();
    pressed = true;
    longConfirmed = false;
  }

  // While button is still held — check if long press threshold reached
  if (current == HIGH && pressed && !longConfirmed) {
    unsigned long held = millis() - pressStart;
    if (held >= LONG_PRESS_MS) {
      // Play a confirmation tone so user knows they can release
      tone(BUZZER, 1500, 100);
      delay(110);
      noTone(BUZZER);
      longConfirmed = true;
    }
  }

  // Button just released
  if (current == LOW && lastState == HIGH && pressed) {
    pressed = false;
    unsigned long duration = millis() - pressStart;

    if (longConfirmed) {
      lastState = current;
      beepClick();
      return BTN_LONG;
    } else if (duration >= SHORT_PRESS_MS) {
      lastState = current;
      beepClick();
      return BTN_SHORT;
    }
    longConfirmed = false;
  }

  lastState = current;
  return BTN_NONE;
}

// ═══════════════════════════════════════════════════════════
//  MEASUREMENT  ENGINE
// ═══════════════════════════════════════════════════════════

/*
 * measureLight()
 * Turns on a single LED, waits for the LDR to settle,
 * takes multiple readings and returns the average.
 * All other LEDs are OFF during measurement.
 * Uses inverted logic for common anode RGB LED.
 */
float measureLight(int ledPin) {
  // Ensure all LEDs are off first
  allLedsOff();
  delay(50); // dark settle

  // Turn on target LED
  ledOn(ledPin);
  delay(SETTLE_TIME_MS);

  // Take averaged readings
  long total = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    total += analogRead(LDR_PIN);
    delay(SAMPLE_DELAY_MS);
  }

  ledOff(ledPin);
  delay(50); // dark recovery

  return (float)total / NUM_SAMPLES;
}

/*
 * measureAllChannels()
 * Measures transmitted light for Red, Green, Blue
 * and stores results in the provided array.
 * Shows animated progress on LCD during measurement.
 * Each LED stays on during its entire scan phase.
 */
void measureAllChannels(float *readings) {
  const char *labels[] = {"Red  ", "Green", "Blue "};
  const char *dots[] = {".  ", ".. ", "..."};

  for (int i = 0; i < 3; i++) {
    // Turn off all LEDs first (dark settle)
    allLedsOff();
    delay(500);

    // Turn ON this channel's LED — it stays on the whole time
    ledOn(ledPins[i]);

    // Show which channel is being measured
    lcd.setCursor(0, 1);
    lcd.print("                ");
    lcd.setCursor(0, 1);
    lcd.print("Scan: ");
    lcd.print(labels[i]);

    // Animate dots while LED settles
    for (int d = 0; d < 3; d++) {
      lcd.setCursor(12, 1);
      lcd.print(dots[d]);
      delay(200);
    }

    // Take averaged readings (LED is already on and settled)
    long total = 0;
    for (int s = 0; s < NUM_SAMPLES; s++) {
      total += analogRead(LDR_PIN);
      delay(SAMPLE_DELAY_MS);
    }
    readings[i] = (float)total / NUM_SAMPLES;

    // Show result while LED is still on
    lcd.setCursor(0, 1);
    lcd.print("                ");
    lcd.setCursor(0, 1);
    lcd.print(labels[i]);
    lcd.print(": ");
    lcd.print((int)readings[i]);
    lcd.print(" OK");
    delay(700);

    // Now turn off the LED before next channel
    ledOff(ledPins[i]);
    delay(100);
  }

  // Final pause
  lcd.setCursor(0, 1);
  lcd.print("  Complete!     ");
  delay(600);
}

/*
 * calculateAbsorbance()
 * Beer-Lambert: A = -log10(I / I0)
 * Returns 0 if I >= I0 (no absorption) or on error.
 */
float calculateAbsorbance(float i0, float i) {
  if (i0 <= 0 || i <= 0)
    return 0.0;
  float ratio = i / i0;
  if (ratio >= 1.0)
    return 0.0; // no absorption or error
  float A = -log10(ratio);
  if (isnan(A) || isinf(A))
    return 0.0;
  return A;
}

// ═══════════════════════════════════════════════════════════
//  CALIBRATION  CURVE  (Linear Regression)
// ═══════════════════════════════════════════════════════════

/*
 * linearRegression()
 * Performs least-squares fit: A = m*C + b
 * Uses the best channel's absorbance values.
 * Returns true if valid.
 */
bool linearRegression() {
  if (numStandards < 2)
    return false;

  float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  int n = numStandards;

  for (int i = 0; i < n; i++) {
    float x = stdConcentrations[i];
    float y = stdAbsorbances[i];
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumX2 += x * x;
  }

  float denom = (n * sumX2 - sumX * sumX);
  if (abs(denom) < 1e-10)
    return false;

  slopeM = (n * sumXY - sumX * sumY) / denom;
  interceptB = (sumY - slopeM * sumX) / n;

  return true;
}

/*
 * calcConcentration()
 * From the calibration curve: C = (A - b) / m
 */
float calcConcentration(float abs_val) {
  if (abs(slopeM) < 1e-10)
    return 0.0;
  float c = (abs_val - interceptB) / slopeM;
  return (c < 0) ? 0.0 : c;
}

// ═══════════════════════════════════════════════════════════
//  CONCENTRATION  ENTRY  HELPERS
// ═══════════════════════════════════════════════════════════

float digitsToConc() {
  float val = concDigits[0] * 100.0 + concDigits[1] * 10.0 +
              concDigits[2] * 1.0 + concDigits[3] * 0.1 + concDigits[4] * 0.01;
  return val;
}

void displayConcEntry() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Conc(");
  lcd.print(unitLabels[currentUnit]);
  lcd.print("):");

  lcd.setCursor(0, 1);
  char buf[10];
  snprintf(buf, sizeof(buf), "%d%d%d.%d%d", concDigits[0], concDigits[1],
           concDigits[2], concDigits[3], concDigits[4]);
  lcd.print(buf);

  // Position the blinking cursor under the active digit
  int cursorCol = concDigitPos;
  if (concDigitPos >= 3)
    cursorCol++; // skip the decimal point
  lcd.setCursor(cursorCol, 1);
  lcd.blink();
}

// ═══════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════

void setup() {
  // Pin modes
  pinMode(RED_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(BUTTON_PIN, INPUT); // external pull-down assumed

  // All LEDs off (HIGH for common anode)
  allLedsOff();

  // LCD init
  lcd.init();
  lcd.backlight();

  // Welcome
  lcdPrint("    Galvaniy    ", "  Technologies  ");
  beepWelcome();
  delay(2000);

  lcdPrint("  Chroma-Scan ", "    v1.0    ");
  delay(2000);

  lcdPrint("Press btn to", "begin...");
  currentState = STATE_WELCOME;

  Serial.begin(9600);
  Serial.println(F("Chroma-Scan Ready"));
}

// ═══════════════════════════════════════════════════════════
//  MAIN  LOOP  —  STATE  MACHINE
// ═══════════════════════════════════════════════════════════

void loop() {
  ButtonEvent btn = readButton();

  // Run rainbow animation during idle states
  if (btn == BTN_NONE) {
    switch (currentState) {
    case STATE_WELCOME:
    case STATE_SETTINGS:
    case STATE_CAL_BLANK:
    case STATE_CAL_DONE:
    case STATE_STD_MENU:
    case STATE_STD_MODE_SELECT:
    case STATE_SET_CONC_MANUAL:
    case STATE_SET_CONC_PRESET:
    case STATE_STD_DONE:
    case STATE_MEASURE_SAMPLE:
    case STATE_RESULTS:
      updateAnimation();
      break;
    default:
      break;
    }
  } else {
    // Button was pressed — stop animation immediately
    stopAnimation();
  }

  switch (currentState) {

  // ─────────────────────────────────────────────────────
  case STATE_WELCOME:
    if (btn == BTN_SHORT) {
      currentState = STATE_SETTINGS;
      lcd.noBlink();
      lcdPrint("Settings", "Srt:Unit Lng:Skip");
      beepSettings();
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_SETTINGS:
    if (btn == BTN_SHORT) {
      currentUnit = (UnitType)((currentUnit + 1) % UNIT_COUNT);
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Unit: ");
      lcd.print(unitLabels[currentUnit]);
      lcd.setCursor(0, 1);
      lcd.print("Srt:Next Lng:OK");
      beepSettings();
    } else if (btn == BTN_LONG) {
      currentState = STATE_CAL_BLANK;
      lcd.noBlink();
      lcdPrint("Insert BLANK", "Press to calibr.");
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_CAL_BLANK:
    if (btn == BTN_SHORT) {
      currentState = STATE_CAL_MEASURING;
      lcdPrint("Calibrating...", "Please wait...");

      // Measure baseline for all 3 channels
      measureAllChannels(I0);

      // Display I0 values
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Baseline saved!");
      lcd.setCursor(0, 1);
      char buf[17];
      snprintf(buf, sizeof(buf), "R:%d G:%d B:%d", (int)I0[0], (int)I0[1],
               (int)I0[2]);
      lcd.print(buf);

      Serial.print(F("I0 R="));
      Serial.print(I0[0]);
      Serial.print(F(" G="));
      Serial.print(I0[1]);
      Serial.print(F(" B="));
      Serial.println(I0[2]);

      beepCalDone();
      currentState = STATE_CAL_DONE;
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_CAL_DONE:
    if (btn == BTN_SHORT) {
      currentState = STATE_STD_MENU;
      lcd.noBlink();
      char buf[17];
      snprintf(buf, sizeof(buf), "Standards: %d/%d", numStandards,
               MAX_STANDARDS);
      lcdPrint(buf, "Srt:Add Lng:Done");
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_STD_MENU:
    if (btn == BTN_SHORT) {
      if (numStandards >= MAX_STANDARDS) {
        beepError();
        lcdPrint("Max standards!", "Long press: Done");
      } else {
        currentState = STATE_STD_MODE_SELECT;
        stdEntryMode = 0;
        lcdPrint("Entry mode:", "Srt:Manual Lng:Pre");
      }
    } else if (btn == BTN_LONG) {
      if (numStandards >= 2) {
        curveValid = linearRegression();
        if (curveValid) {
          lcd.clear();
          lcd.setCursor(0, 0);
          lcd.print("Curve ready!");
          lcd.setCursor(0, 1);
          lcd.print("m=");
          lcd.print(slopeM, 4);
          delay(1500);
        } else {
          lcdPrint("Curve error!", "Check standards");
          beepError();
          delay(1500);
        }
      } else if (numStandards == 0) {
        curveValid = false;
      } else {
        lcdPrint("Need min 2 stds", "Add more or skip");
        beepError();
        delay(1500);
        break;
      }
      currentState = STATE_MEASURE_SAMPLE;
      lcdPrint("Insert SAMPLE", "Press to measure");
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_STD_MODE_SELECT:
    if (btn == BTN_SHORT) {
      stdEntryMode = 0;
      concDigitPos = 0;
      memset(concDigits, 0, sizeof(concDigits));
      currentState = STATE_SET_CONC_MANUAL;
      displayConcEntry();
    } else if (btn == BTN_LONG) {
      stdEntryMode = 1;
      presetIndex = 0;
      currentState = STATE_SET_CONC_PRESET;
      lcd.noBlink();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Select conc:");
      lcd.setCursor(0, 1);
      lcd.print("> ");
      lcd.print(PRESETS[presetIndex], 2);
      lcd.print(" ");
      lcd.print(unitLabels[currentUnit]);
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_SET_CONC_MANUAL:
    if (btn == BTN_SHORT) {
      concDigits[concDigitPos] = (concDigits[concDigitPos] + 1) % 10;
      displayConcEntry();
    } else if (btn == BTN_LONG) {
      concDigitPos++;
      if (concDigitPos >= 5) {
        manualConc = digitsToConc();
        lcd.noBlink();
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Std C=");
        lcd.print(manualConc, 2);
        lcd.setCursor(0, 1);
        lcd.print("Measuring...");

        currentState = STATE_MEASURE_STD;

        float readings[3];
        measureAllChannels(readings);

        float bestAbs = 0;
        int bestCh = 0;
        for (int i = 0; i < 3; i++) {
          float a = calculateAbsorbance(I0[i], readings[i]);
          if (a > bestAbs) {
            bestAbs = a;
            bestCh = i;
          }
        }

        stdConcentrations[numStandards] = manualConc;
        stdAbsorbances[numStandards] = bestAbs;
        numStandards++;

        Serial.print(F("Std #"));
        Serial.print(numStandards);
        Serial.print(F(" C="));
        Serial.print(manualConc);
        Serial.print(F(" A="));
        Serial.print(bestAbs, 4);
        Serial.print(F(" Ch="));
        Serial.println(channelNames[bestCh]);

        lcd.clear();
        lcd.setCursor(0, 0);
        char buf[17];
        snprintf(buf, sizeof(buf), "Std #%d saved!", numStandards);
        lcd.print(buf);
        lcd.setCursor(0, 1);
        lcd.print("A=");
        lcd.print(bestAbs, 4);
        lcd.print(" @");
        lcd.print(channelNames[bestCh][0]);

        beepStdSaved();
        currentState = STATE_STD_DONE;
      } else {
        displayConcEntry();
      }
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_SET_CONC_PRESET:
    if (btn == BTN_SHORT) {
      presetIndex = (presetIndex + 1) % NUM_PRESETS;
      lcd.setCursor(0, 1);
      lcd.print("                ");
      lcd.setCursor(0, 1);
      lcd.print("> ");
      lcd.print(PRESETS[presetIndex], 2);
      lcd.print(" ");
      lcd.print(unitLabels[currentUnit]);
    } else if (btn == BTN_LONG) {
      float selectedConc = PRESETS[presetIndex];
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Std C=");
      lcd.print(selectedConc, 2);
      lcd.setCursor(0, 1);
      lcd.print("Measuring...");

      currentState = STATE_MEASURE_STD;

      float readings[3];
      measureAllChannels(readings);

      float bestAbs = 0;
      int bestCh = 0;
      for (int i = 0; i < 3; i++) {
        float a = calculateAbsorbance(I0[i], readings[i]);
        if (a > bestAbs) {
          bestAbs = a;
          bestCh = i;
        }
      }

      stdConcentrations[numStandards] = selectedConc;
      stdAbsorbances[numStandards] = bestAbs;
      numStandards++;

      Serial.print(F("Std #"));
      Serial.print(numStandards);
      Serial.print(F(" C="));
      Serial.print(selectedConc);
      Serial.print(F(" A="));
      Serial.print(bestAbs, 4);
      Serial.print(F(" Ch="));
      Serial.println(channelNames[bestCh]);

      lcd.clear();
      lcd.setCursor(0, 0);
      char buf[17];
      snprintf(buf, sizeof(buf), "Std #%d saved!", numStandards);
      lcd.print(buf);
      lcd.setCursor(0, 1);
      lcd.print("A=");
      lcd.print(bestAbs, 4);
      lcd.print(" @");
      lcd.print(channelNames[bestCh][0]);

      beepStdSaved();
      currentState = STATE_STD_DONE;
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_MEASURE_STD:
    // Transient state — handled inline above
    break;

  // ─────────────────────────────────────────────────────
  case STATE_STD_DONE:
    if (btn == BTN_SHORT) {
      currentState = STATE_STD_MENU;
      char buf[17];
      snprintf(buf, sizeof(buf), "Standards: %d/%d", numStandards,
               MAX_STANDARDS);
      lcdPrint(buf, "Srt:Add Lng:Done");
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_MEASURE_SAMPLE:
    if (btn == BTN_SHORT) {
      currentState = STATE_SAMPLING;
      lcdPrint("Analyzing...", "R.. G.. B..");

      // Measure all channels
      measureAllChannels(sampleI);

      // Calculate absorbance for each channel
      float maxAbs = 0;
      bestChannel = 0;
      for (int i = 0; i < 3; i++) {
        absorbance[i] = calculateAbsorbance(I0[i], sampleI[i]);
        if (absorbance[i] > maxAbs) {
          maxAbs = absorbance[i];
          bestChannel = i;
        }
      }

      // Calculate concentration if curve is valid
      if (curveValid) {
        concentration = calcConcentration(absorbance[bestChannel]);
      } else {
        concentration = -1; // indicate no curve
      }

      Serial.println(F("--- Sample Results ---"));
      for (int i = 0; i < 3; i++) {
        Serial.print(channelNames[i]);
        Serial.print(F(": I="));
        Serial.print(sampleI[i]);
        Serial.print(F(" A="));
        Serial.println(absorbance[i], 4);
      }
      Serial.print(F("Best: "));
      Serial.println(channelNames[bestChannel]);
      if (curveValid) {
        Serial.print(F("Conc: "));
        Serial.print(concentration, 2);
        Serial.print(F(" "));
        Serial.println(unitLabels[currentUnit]);
      }

      beepMeasureDone();

      // Show first result page
      resultPage = 0;
      currentState = STATE_RESULTS;
      displayResultPage();
    } else if (btn == BTN_LONG) {
      // Full reset — back to calibration
      numStandards = 0;
      curveValid = false;
      currentState = STATE_CAL_BLANK;
      lcdPrint("== RESET ==", "Insert BLANK");
      beepError();
      delay(1000);
      lcdPrint("Insert BLANK", "Press to calibr.");
    }
    break;

  // ─────────────────────────────────────────────────────
  case STATE_RESULTS:
    if (btn == BTN_SHORT) {
      resultPage = (resultPage + 1) % RESULT_PAGES;
      displayResultPage();
    } else if (btn == BTN_LONG) {
      currentState = STATE_MEASURE_SAMPLE;
      lcdPrint("Insert SAMPLE", "Press to measure");
    }
    break;
  }
}

// ═══════════════════════════════════════════════════════════
//  RESULTS  DISPLAY
// ═══════════════════════════════════════════════════════════

void displayResultPage() {
  lcd.clear();

  switch (resultPage) {
  case 0: // Red Absorbance
    lcd.setCursor(0, 0);
    lcd.print("Abs(Red):");
    lcd.setCursor(0, 1);
    lcd.print(absorbance[0], 4);
    lcd.setCursor(10, 1);
    lcd.print("[1/5]");
    break;

  case 1: // Green Absorbance
    lcd.setCursor(0, 0);
    lcd.print("Abs(Green):");
    lcd.setCursor(0, 1);
    lcd.print(absorbance[1], 4);
    lcd.setCursor(10, 1);
    lcd.print("[2/5]");
    break;

  case 2: // Blue Absorbance
    lcd.setCursor(0, 0);
    lcd.print("Abs(Blue):");
    lcd.setCursor(0, 1);
    lcd.print(absorbance[2], 4);
    lcd.setCursor(10, 1);
    lcd.print("[3/5]");
    break;

  case 3: // Best wavelength
    lcd.setCursor(0, 0);
    lcd.print("Best Channel:");
    lcd.setCursor(0, 1);
    lcd.print(channelNames[bestChannel]);
    lcd.print(" A=");
    lcd.print(absorbance[bestChannel], 4);
    break;

  case 4: // Concentration
    lcd.setCursor(0, 0);
    lcd.print("Concentration:");
    lcd.setCursor(0, 1);
    if (curveValid && concentration >= 0) {
      lcd.print(concentration, 2);
      lcd.print(" ");
      lcd.print(unitLabels[currentUnit]);
    } else {
      lcd.print("No cal curve");
    }
    break;
  }
}
