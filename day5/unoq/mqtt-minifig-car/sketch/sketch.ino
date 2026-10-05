// UNO Q microcontroller side: LED matrix display + DC motor drive.
// The Linux side (python/main.py) calls "draw" with a 8x13 frame and
// "drive" with a signed speed (-100..100). If no drive command arrives
// for WATCHDOG_MS the motors stop on their own.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

Arduino_LED_Matrix matrix;

const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
const uint8_t FRAME_SIZE = FRAME_ROWS * FRAME_COLS;
uint8_t frame[FRAME_SIZE] = {0};

// --- Motor driver wiring: Seeed/Cytron Maker Drive (MX1508) ---------------
// Two PWM inputs per motor:
//   forward  = A: PWM, B: 0      backward = A: 0, B: PWM      stop = 0, 0
//   Maker Drive M1A/M1B <- left motor pins,  M2A/M2B <- right motor pins
// UNO Q notes (ArduinoCore-zephyr):
//   * do NOT call pinMode() on these pins -- on the UNO Q it breaks PWM;
//     analogWrite() configures the pin itself
//   * keep off D3 and D11: PWM there disables other pins on cores < 0.55.2
//     (update the core in App Lab if it's older)
const int M_LEFT_A  = 5;
const int M_LEFT_B  = 6;
const int M_RIGHT_A = 9;
const int M_RIGHT_B = 10;
const int LEFT_SIGN  = +1;   // flip one of these if a wheel spins the wrong
const int RIGHT_SIGN = +1;   // way. Both +1 here: this car's motors are wired
                             // with opposite polarity, which already cancels
                             // their mirror-image mounting.

const unsigned long WATCHDOG_MS = 1000;
unsigned long lastDriveMs = 0;
int currentSpeed = 0;

void setMotor(int pinA, int pinB, int s) {
  int pwm = map(constrain(abs(s), 0, 100), 0, 100, 0, 255);
  if (s > 0)      { analogWrite(pinA, pwm); analogWrite(pinB, 0); }
  else if (s < 0) { analogWrite(pinA, 0);   analogWrite(pinB, pwm); }
  else            { analogWrite(pinA, 0);   analogWrite(pinB, 0); }
}

// Called from Python: signed speed in percent, -100..100.
void drive(int speed) {
  currentSpeed = constrain(speed, -100, 100);
  lastDriveMs = millis();
  setMotor(M_LEFT_A,  M_LEFT_B,  currentSpeed * LEFT_SIGN);
  setMotor(M_RIGHT_A, M_RIGHT_B, currentSpeed * RIGHT_SIGN);
}

// Called from Python with a new 8x13 frame (row-major brightness 0-7).
void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
}

void setup() {
  drive(0);   // analogWrite() sets up the pins; no pinMode() (see note above)

  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.provide("drive", drive);
}

void loop() {
  matrix.draw(frame);
  if (currentSpeed != 0 && millis() - lastDriveMs > WATCHDOG_MS) {
    drive(0);   // Linux side went quiet -> don't keep rolling
  }
  delay(10);
}
