/*
 * ============================================================
 *  CHROMA-SCAN  —  Arduino Nano DAQ (Data Acquisition)
 * ============================================================
 *  Acts as a dedicated sensor frontend for the ESP8266.
 *  Receives serial commands over UART to sequence the RGB LED,
 *  read the LDR sensor, and control LED status lighting.
 *
 *  Hardware:
 *    - Arduino Nano
 *    - Common Anode RGB LED (Red: 6, Green: 7, Blue: 4)
 *    - LDR + 1kΩ voltage divider (A0)
 * ============================================================
 */

#define RED_LED 6
#define BLUE_LED 4
#define GREEN_LED 7
#define LDR_PIN A0

#define SETTLE_TIME_MS 300
#define NUM_SAMPLES 30
#define SAMPLE_DELAY_MS 5

void setup() {
  Serial.begin(9600); // Hardware Serial
  
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);
  
  allLedsOff();
  Serial.println(F("CHROMA_DAQ_READY"));
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd == "SCAN") {
      runScan();
    } else if (cmd.startsWith("LED:")) {
      setLedFromCmd(cmd);
    } else if (cmd == "PING") {
      Serial.println(F("PONG"));
    }
  }
}

void allLedsOff() {
  analogWrite(RED_LED, 255);
  analogWrite(GREEN_LED, 255);
  analogWrite(BLUE_LED, 255);
}

void ledOn(int pin) {
  analogWrite(pin, 0);
}

void ledOff(int pin) {
  analogWrite(pin, 255);
}

void setRGB(byte r, byte g, byte b) {
  analogWrite(RED_LED, 255 - r);
  analogWrite(GREEN_LED, 255 - g);
  analogWrite(BLUE_LED, 255 - b);
}

float measureChannel(int ledPin) {
  if (ledPin != -1) {
    ledOn(ledPin);
    delay(SETTLE_TIME_MS);
  } else {
    allLedsOff();
    delay(50); // settle dark
  }

  long total = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    total += analogRead(LDR_PIN);
    delay(SAMPLE_DELAY_MS);
  }

  if (ledPin != -1) {
    ledOff(ledPin);
  }
  return (float)total / NUM_SAMPLES;
}

void runScan() {
  // 1. Measure Ambient (all LEDs off)
  float ambient = measureChannel(-1);
  
  // 2. Measure Red
  float red = measureChannel(RED_LED);
  
  // 3. Measure Green
  float green = measureChannel(GREEN_LED);
  
  // 4. Measure Blue
  float blue = measureChannel(BLUE_LED);
  
  // Send data over serial
  Serial.print(F("DATA:"));
  Serial.print(ambient, 2);
  Serial.print(F(","));
  Serial.print(red, 2);
  Serial.print(F(","));
  Serial.print(green, 2);
  Serial.print(F(","));
  Serial.println(blue, 2);
}

void setLedFromCmd(String cmd) {
  // Format: LED:r,g,b
  int firstComma = cmd.indexOf(',');
  int secondComma = cmd.indexOf(',', firstComma + 1);
  
  if (firstComma != -1 && secondComma != -1) {
    int r = cmd.substring(4, firstComma).toInt();
    int g = cmd.substring(firstComma + 1, secondComma).toInt();
    int b = cmd.substring(secondComma + 1).toInt();
    
    r = constrain(r, 0, 255);
    g = constrain(g, 0, 255);
    b = constrain(b, 0, 255);
    
    setRGB(r, g, b);
  }
}
