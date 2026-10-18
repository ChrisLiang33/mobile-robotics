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

float readCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  unsigned long us = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (us == 0) return -1.0f;
  return us / 58.0f;                             // 343 m/s, there and back
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
  }
  delay(5);
}
