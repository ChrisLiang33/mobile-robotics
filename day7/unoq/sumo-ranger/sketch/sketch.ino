// UNO Q microcontroller side: HC-SR04 ultrasonic range finder + LED matrix.
//
// Wiring (HC-SR04 -> UNO Q):
//   VCC  -> 5V          GND -> GND
//   TRIG -> D7          (3.3 V trigger pulse is enough for the HC-SR04)
//   ECHO -> D8 through a divider: ECHO --[1 kOhm]--+--[2 kOhm]-- GND, the + goes
//           to D8. The echo pin outputs 5 V and UNO Q pins are 3.3 V.
//
// Every 60 ms: 10 us trigger pulse, time the echo, convert to cm, hand it to
// the Linux side with Bridge.notify("range", cm). No echo -> -1.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

const int TRIG_PIN = 7;
const int ECHO_PIN = 8;
const unsigned long ECHO_TIMEOUT_US = 25000;   // ~4 m round trip
const unsigned long PERIOD_MS = 60;

Arduino_LED_Matrix matrix;
const uint8_t FRAME_SIZE = 8 * 13;
uint8_t frame[FRAME_SIZE] = {0};
unsigned long lastPing = 0;

void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
}

// Timed by hand with micros() instead of pulseIn(), which not every core
// implements. diag codes: 0 = echo never went high (no sensor response),
// 1 = echo stuck high (wiring / level problem), 2 = echo too long (nothing in range)
int lastDiag = -1;

float readCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  unsigned long t0 = micros();
  if (digitalRead(ECHO_PIN) == HIGH) {            // should be low before the echo starts
    while (digitalRead(ECHO_PIN) == HIGH) { if (micros() - t0 > ECHO_TIMEOUT_US) { lastDiag = 1; return -1.0f; } }
  }
  while (digitalRead(ECHO_PIN) == LOW) {          // wait for the echo pulse to start
    if (micros() - t0 > ECHO_TIMEOUT_US) { lastDiag = 0; return -1.0f; }
  }
  unsigned long start = micros();
  while (digitalRead(ECHO_PIN) == HIGH) {         // measure it
    if (micros() - start > ECHO_TIMEOUT_US) { lastDiag = 2; return -1.0f; }
  }
  lastDiag = 3;
  return (micros() - start) / 58.0f;              // 343 m/s, there and back
}

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw", draw);
}

void loop() {
  matrix.draw(frame);
  if (millis() - lastPing >= PERIOD_MS) {
    lastPing = millis();
    float cm = readCm();
    Bridge.notify("range", cm);
    static unsigned long lastDiagMs = 0;
    if (millis() - lastDiagMs > 1000) { lastDiagMs = millis(); Bridge.notify("diag", lastDiag, (int)digitalRead(ECHO_PIN)); }
  }
  delay(5);
}
