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
 *    - Red LED    → pin 2
 *    - Blue LED   → pin 3
 *    - Green LED  → pin 4
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

// ─── Pin Definitions ────────────────────────────────────────
#define RED_LED 2
#define BLUE_LED 3
#define GREEN_LED 4
#define BUZZER 13
#define LDR_PIN A0
#define BUTTON_PIN 12

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
  // Clear the row first
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
  static bool longConfirmed = false; // true once long-press tone has played

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
      // Already confirmed as long press
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
 */
float measureLight(int ledPin) {
  // Ensure all LEDs are off first
  digitalWrite(RED_LED, LOW);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(BLUE_LED, LOW);
  delay(50); // dark settle

  // Turn on target LED
  digitalWrite(ledPin, HIGH);
  delay(SETTLE_TIME_MS);

  // Take averaged readings
  long total = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    total += analogRead(LDR_PIN);
    delay(SAMPLE_DELAY_MS);
  }

  digitalWrite(ledPin, LOW);
  delay(50); // dark recovery

  return (float)total / NUM_SAMPLES;
}

/*
 * measureAllChannels()
 * Measures transmitted light for Red, Green, Blue
 * and stores results in the provided array.
 */
void measureAllChannels(float *readings) {
  for (int i = 0; i < 3; i++) {
    readings[i] = measureLight(ledPins[i]);
  }
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
    return false; // vertical line / no spread

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

/*
 * digitsToConc()
 * Converts the 5-digit array (XXX.XX) to a float.
 */
float digitsToConc() {
  float val = concDigits[0] * 100.0 + concDigits[1] * 10.0 +
              concDigits[2] * 1.0 + concDigits[3] * 0.1 + concDigits[4] * 0.01;
  return val;
}

/*
 * displayConcEntry()
 * Shows the current digit entry on LCD with cursor indicator.
 */
void displayConcEntry() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Conc(");
  lcd.print(unitLabels[currentUnit]);
  lcd.print("):");

  lcd.setCursor(0, 1);
  // Format: XXX.XX
  char buf[10];
  snprintf(buf, sizeof(buf), "%d%d%d.%d%d", concDigits[0], concDigits[1],
           concDigits[2], concDigits[3], concDigits[4]);
  lcd.print(buf);

  // Show cursor position indicator
  lcd.setCursor(10, 1);
  lcd.print("^");
  lcd.setCursor(10, 1);

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

  // All LEDs off
  digitalWrite(RED_LED, LOW);
  digitalWrite(BLUE_LED, LOW);
  digitalWrite(GREEN_LED, LOW);

  // LCD init
  lcd.init();
  lcd.backlight();

  // Welcome
  lcdPrint("  Chroma-Scan ", "    v1.0    ");
  beepWelcome();
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
      // Cycle through units
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
      // Done adding standards → proceed to measurement
      if (numStandards >= 2) {
        curveValid = linearRegression();
        if (curveValid) {
          lcd.clear();
          lcd.setCursor(0, 0);
          lcd.print("Curve ready!");
          lcd.setCursor(0, 1);
          char buf[17];
          // Show slope info
          lcd.print("m=");
          lcd.print(slopeM, 4);
          delay(1500);
        } else {
          lcdPrint("Curve error!", "Check standards");
          beepError();
          delay(1500);
        }
      } else if (numStandards == 0) {
        // No cal curve - direct measurement only
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
      // Manual digit entry
      stdEntryMode = 0;
      concDigitPos = 0;
      memset(concDigits, 0, sizeof(concDigits));
      currentState = STATE_SET_CONC_MANUAL;
      displayConcEntry();
    } else if (btn == BTN_LONG) {
      // Preset selection
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
      // Increment current digit (0-9)
      concDigits[concDigitPos] = (concDigits[concDigitPos] + 1) % 10;
      displayConcEntry();
    } else if (btn == BTN_LONG) {
      // Move to next digit or confirm
      concDigitPos++;
      if (concDigitPos >= 5) {
        // All digits entered → measure this standard
        manualConc = digitsToConc();
        lcd.noBlink();
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Std C=");
        lcd.print(manualConc, 2);
        lcd.setCursor(0, 1);
        lcd.print("Measuring...");

        currentState = STATE_MEASURE_STD;

        // Perform measurement
        float readings[3];
        measureAllChannels(readings);

        // Find best channel (highest absorbance)
        float bestAbs = 0;
        int bestCh = 0;
        for (int i = 0; i < 3; i++) {
          float a = calculateAbsorbance(I0[i], readings[i]);
          if (a > bestAbs) {
            bestAbs = a;
            bestCh = i;
          }
        }

        // Store calibration point (use best channel absorbance)
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
        lcd.print(channelNames[bestCh][0]); // R, G, or B

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
      // Cycle through presets
      presetIndex = (presetIndex + 1) % NUM_PRESETS;
      lcd.setCursor(0, 1);
      lcd.print("                ");
      lcd.setCursor(0, 1);
      lcd.print("> ");
      lcd.print(PRESETS[presetIndex], 2);
      lcd.print(" ");
      lcd.print(unitLabels[currentUnit]);
    } else if (btn == BTN_LONG) {
      // Confirm preset → measure this standard
      float selectedConc = PRESETS[presetIndex];
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Std C=");
      lcd.print(selectedConc, 2);
      lcd.setCursor(0, 1);
      lcd.print("Measuring...");

      currentState = STATE_MEASURE_STD;

      // Perform measurement
      float readings[3];
      measureAllChannels(readings);

      // Find best channel
      float bestAbs = 0;
      int bestCh = 0;
      for (int i = 0; i < 3; i++) {
        float a = calculateAbsorbance(I0[i], readings[i]);
        if (a > bestAbs) {
          bestAbs = a;
          bestCh = i;
        }
      }

      // Store
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
      // Return to standards menu
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
      // Cycle through result pages
      resultPage = (resultPage + 1) % RESULT_PAGES;
      displayResultPage();
    } else if (btn == BTN_LONG) {
      // New sample measurement
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
  char line1[17];
  char line2[17];

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
