// UNO Q microcontroller side: read the MPU-6050 (6-axis IMU) over I2C at
// SAMPLE_HZ and hand every sample to the Linux side with
// Bridge.notify("imu", ax, ay, az, gx, gy, gz)  -- accel in g, gyro in deg/s.
//
// Wiring (MPU-6050 module -> UNO Q):  VCC -> 3.3V (or 5V: the module has its
// own regulator), GND -> GND, SDA -> SDA, SCL -> SCL (the two pins next to
// AREF; A4/A5 on the UNO layout).  AD0 left unconnected = address 0x68.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>
#include <vector>

const uint8_t MPU = 0x68;
const unsigned long SAMPLE_HZ = 50;
const unsigned long PERIOD_US = 1000000UL / SAMPLE_HZ;

Arduino_LED_Matrix matrix;
const uint8_t FRAME_SIZE = 8 * 13;
uint8_t frame[FRAME_SIZE] = {0};
unsigned long lastSample = 0;
bool imuOk = false;

void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
}

bool mpuBegin() {
  Wire.beginTransmission(MPU);
  Wire.write(0x6B); Wire.write(0x00);            // PWR_MGMT_1: wake up
  if (Wire.endTransmission() != 0) return false;
  Wire.beginTransmission(MPU);
  Wire.write(0x1C); Wire.write(0x08);            // ACCEL_CONFIG: +-4 g (8192 LSB/g)
  Wire.endTransmission();
  Wire.beginTransmission(MPU);
  Wire.write(0x1B); Wire.write(0x08);            // GYRO_CONFIG: +-500 deg/s (65.5 LSB/(deg/s))
  Wire.endTransmission();
  Wire.beginTransmission(MPU);
  Wire.write(0x1A); Wire.write(0x03);            // CONFIG: 44 Hz low-pass
  Wire.endTransmission();
  return true;
}

bool mpuRead(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);                               // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU, 14) != 14) return false;
  int16_t raw[7];
  for (int i = 0; i < 7; i++) { raw[i] = (Wire.read() << 8) | Wire.read(); }
  ax = raw[0] / 8192.0f; ay = raw[1] / 8192.0f; az = raw[2] / 8192.0f;   // raw[3] = temperature
  gx = raw[4] / 65.5f;   gy = raw[5] / 65.5f;   gz = raw[6] / 65.5f;
  return true;
}

void setup() {
  Wire.begin();
  imuOk = mpuBegin();
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.notify("status", imuOk ? 1 : 0);
}

void loop() {
  matrix.draw(frame);
  unsigned long now = micros();
  if (now - lastSample >= PERIOD_US) {
    lastSample = now;
    float ax, ay, az, gx, gy, gz;
    if (!imuOk) { imuOk = mpuBegin(); }
    if (imuOk && mpuRead(ax, ay, az, gx, gy, gz)) {
      Bridge.notify("imu", ax, ay, az, gx, gy, gz);
    } else {
      static unsigned long lastErr = 0;
      if (millis() - lastErr > 2000) { lastErr = millis(); Bridge.notify("status", 0); imuOk = false; }
    }
  }
}
