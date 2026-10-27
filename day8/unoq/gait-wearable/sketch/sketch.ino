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

const uint8_t IMU_ADDR = 0x68;
const unsigned long SAMPLE_HZ = 50;
const unsigned long PERIOD_US = 1000000UL / SAMPLE_HZ;

Arduino_LED_Matrix matrix;
const uint8_t FRAME_SIZE = 8 * 13;
uint8_t frame[FRAME_SIZE] = {0};
unsigned long lastSample = 0;
bool imuOk = false;
TwoWire *bus = &Wire;
int busId = -1;

void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
}

bool mpuBegin() {
  bus->beginTransmission(IMU_ADDR);
  bus->write(0x6B); bus->write(0x00);            // PWR_MGMT_1: wake up
  if (bus->endTransmission() != 0) return false;
  bus->beginTransmission(IMU_ADDR);
  bus->write(0x1C); bus->write(0x08);            // ACCEL_CONFIG: +-4 g (8192 LSB/g)
  bus->endTransmission();
  bus->beginTransmission(IMU_ADDR);
  bus->write(0x1B); bus->write(0x08);            // GYRO_CONFIG: +-500 deg/s (65.5 LSB/(deg/s))
  bus->endTransmission();
  bus->beginTransmission(IMU_ADDR);
  bus->write(0x1A); bus->write(0x03);            // CONFIG: 44 Hz low-pass
  bus->endTransmission();
  return true;
}

bool mpuRead(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  bus->beginTransmission(IMU_ADDR);
  bus->write(0x3B);                               // ACCEL_XOUT_H
  if (bus->endTransmission(false) != 0) return false;
  if (bus->requestFrom((int)IMU_ADDR, 14) != 14) return false;
  int16_t raw[7];
  for (int i = 0; i < 7; i++) { int hi = bus->read(); int lo = bus->read(); raw[i] = (int16_t)((hi << 8) | lo); }
  ax = raw[0] / 8192.0f; ay = raw[1] / 8192.0f; az = raw[2] / 8192.0f;   // raw[3] = temperature
  gx = raw[4] / 65.5f;   gy = raw[5] / 65.5f;   gz = raw[6] / 65.5f;
  return true;
}

// The UNO Q has three I2C buses (Wire, Wire1, Wire2). Find the one with the IMU.
bool findImu() {
  TwoWire *buses[3] = {&Wire, &Wire1, &Wire2};
  for (int i = 0; i < 3; i++) {
    buses[i]->begin();
    buses[i]->beginTransmission(IMU_ADDR);
    if (buses[i]->endTransmission() == 0) { bus = buses[i]; busId = i; return mpuBegin(); }
  }
  return false;
}

void setup() {
  imuOk = findImu();
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.notify("status", imuOk ? 1 : 0, busId);
}

void loop() {
  matrix.draw(frame);
  unsigned long now = micros();
  if (now - lastSample >= PERIOD_US) {
    lastSample = now;
    float ax, ay, az, gx, gy, gz;
    if (!imuOk) { imuOk = findImu(); }
    if (imuOk && mpuRead(ax, ay, az, gx, gy, gz)) {
      Bridge.notify("imu", ax, ay, az, gx, gy, gz);
    } else {
      static unsigned long lastErr = 0;
      if (millis() - lastErr > 4000) {
        lastErr = millis(); Bridge.notify("status", 0, -1); imuOk = false;
      }
    }
  }
}
