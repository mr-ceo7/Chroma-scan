/*
 * Simple ESP8266 Blink Test
 * Blinks the onboard LED (typically connected to GPIO 2 / LED_BUILTIN)
 */

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, LOW);   // Turn the LED ON (Active-LOW on most ESP8266 modules)
  delay(200);                      // Fast blink for verification
  digitalWrite(LED_BUILTIN, HIGH);  // Turn the LED OFF
  delay(200);
}
