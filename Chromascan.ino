/*
 * ============================================================
 *  CHROMA-SCAN  —  Arduino Nano Slave Driver (DAQ & Actuators)
 * ============================================================
 *  Acts as a dedicated peripheral for the ESP8266. Handles:
 *    - Common Anode RGB LED: Red (6), Green (7), Blue (4)
 *    - LDR Sensor: A0
 *    - Buzzer: 13
 *    - Responds to UART serial commands from the ESP8266 master.
 *
 *  KSEF 2026  |  Project 6
 * ============================================================
 */

#define RED_LED 6
#define BLUE_LED 4
#define GREEN_LED 7
#define BUZZER 13
#define LDR_PIN A0

#define SETTLE_TIME_MS 300
#define NUM_SAMPLES 30

#define LED_ON LOW
#define LED_OFF HIGH

char rxBuf[48];
uint8_t rxLen = 0;

void playTone(unsigned int frequency, unsigned long duration_ms) {
  if (frequency == 0) {
    delay(duration_ms);
    return;
  }
  unsigned long period_us = 1000000UL / frequency;
  unsigned long half_period_us = period_us / 2;
  unsigned long cycles = (duration_ms * 1000UL) / period_us;
  
  for (unsigned long i = 0; i < cycles; i++) {
    digitalWrite(BUZZER, HIGH);
    delayMicroseconds(half_period_us);
    digitalWrite(BUZZER, LOW);
    delayMicroseconds(half_period_us);
  }
}

void beepClick() {
  playTone(4000, 20);
}

void beepWelcome() {
  playTone(523, 100); delay(20);
  playTone(659, 100); delay(20);
  playTone(784, 150);
}

void beepDone() {
  playTone(2000, 80); delay(40);
  playTone(2000, 80);
}

void beepError() {
  for (int i = 0; i < 3; i++) {
    playTone(200, 80);
    delay(20);
  }
}

void allLedsOff() {
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);
  digitalWrite(RED_LED, LED_OFF);
  digitalWrite(GREEN_LED, LED_OFF);
  digitalWrite(BLUE_LED, LED_OFF);
}

void ledOn(int pin) {
  digitalWrite(pin, LED_ON);
}

void ledOff(int pin) {
  digitalWrite(pin, LED_OFF);
}

void executeScan() {
  int ambient = 0;
  int red = 0;
  int green = 0;
  int blue = 0;
  
  // Step 1: Ambient
  allLedsOff();
  delay(50);
  long sum = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum += analogRead(LDR_PIN);
    Serial.print(F("SCAN_PROGRESS:1,")); Serial.println(i + 1);
    delay(5);
  }
  ambient = sum / NUM_SAMPLES;
  
  // Step 2: Red
  ledOn(RED_LED);
  delay(SETTLE_TIME_MS);
  sum = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum += analogRead(LDR_PIN);
    Serial.print(F("SCAN_PROGRESS:2,")); Serial.println(i + 1);
    delay(5);
  }
  red = sum / NUM_SAMPLES;
  ledOff(RED_LED);
  
  // Step 3: Green
  ledOn(GREEN_LED);
  delay(SETTLE_TIME_MS);
  sum = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum += analogRead(LDR_PIN);
    Serial.print(F("SCAN_PROGRESS:3,")); Serial.println(i + 1);
    delay(5);
  }
  green = sum / NUM_SAMPLES;
  ledOff(GREEN_LED);
  
  // Step 4: Blue
  ledOn(BLUE_LED);
  delay(SETTLE_TIME_MS);
  sum = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum += analogRead(LDR_PIN);
    Serial.print(F("SCAN_PROGRESS:4,")); Serial.println(i + 1);
    delay(5);
  }
  blue = sum / NUM_SAMPLES;
  ledOff(BLUE_LED);
  
  allLedsOff();
  
  // Send final scan data
  Serial.print(F("SCAN_DATA:"));
  Serial.print(ambient); Serial.print(",");
  Serial.print(red); Serial.print(",");
  Serial.print(green); Serial.print(",");
  Serial.println(blue);
}

void handleCommand(char* cmd) {
  if (strncmp(cmd, "SCAN", 4) == 0) {
    executeScan();
  } 
  else if (strncmp(cmd, "BEEP:CLICK", 10) == 0) {
    beepClick();
  }
  else if (strncmp(cmd, "BEEP:WELCOME", 12) == 0) {
    beepWelcome();
  }
  else if (strncmp(cmd, "BEEP:DONE", 9) == 0) {
    beepDone();
  }
  else if (strncmp(cmd, "BEEP:ERROR", 10) == 0) {
    beepError();
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER, OUTPUT);
  pinMode(LDR_PIN, INPUT);
  allLedsOff();
  
  // Flash status beep on start
  beepWelcome();
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen > 0) {
        rxBuf[rxLen] = '\0';
        handleCommand(rxBuf);
        rxLen = 0;
      }
    } else if (rxLen < sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    }
  }
}
