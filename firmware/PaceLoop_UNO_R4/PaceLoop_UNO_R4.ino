// PaceLoop UNO R4 Minima: buttons + LED + MPU6050/MPU6500 over Wire.
// Serial at 115200: IMU,micros,ax,ay,az,gx,gy,gz,temp_C
// Acceleration is m/s^2 (includes gravity); gyro is rad/s.
#include <Wire.h>
#include <string.h>
#include <stdarg.h>
#include "PaceLoopCore.h"

const int BUTTON_FAST = 2, BUTTON_MED = 3, BUTTON_SLOW = 4, LED_PIN = 9;
const uint32_t SAMPLE_US = 20000;  // 50 Hz
uint32_t targetInterval = 4000;  // Boot target interval: four seconds.
uint32_t nextSample = 0, lastProbe = 0, lastState = 0;
bool imuReady = false, streaming = true, commandOverflow = false;
uint8_t imuAddress = 0, deviceId = 0;
char command[64];
size_t commandLength = 0;
paceloop::Control control;
paceloop::Rules rules;
paceloop::Led led;
paceloop::Button buttons[3];

// Bounded USB output queue; sampling continues when the reader stalls.
char tx[1024];
size_t txHead = 0, txSize = 0;
void queueLine(bool sample, const char *format, ...) {
  if (!Serial) return;
  char line[192];
  va_list args;
  va_start(args, format);
  int length = vsnprintf(line, sizeof(line), format, args);
  va_end(args);
  if (length <= 0 || (size_t)length >= sizeof(line)) return;
  size_t reserve = sample ? 256 : 0;
  if (txSize + length + reserve > sizeof(tx)) return;
  for (int i = 0; i < length; ++i) tx[(txHead + txSize++) % sizeof(tx)] = line[i];
}

void flushOutput() {
  if (!Serial) { txHead = txSize = 0; return; }
  size_t count = Serial.availableForWrite();
  if (count > txSize) count = txSize;
  if (count > sizeof(tx) - txHead) count = sizeof(tx) - txHead;
  if (count) {
    size_t written = Serial.write((const uint8_t *)(tx + txHead), count);
    txHead = (txHead + written) % sizeof(tx); txSize -= written;
  }
}

void reportControl() {
  queueLine(false, "CONTROL,%s\n", control.mode == paceloop::PC ? "PC" : "RULES");
}

void reportRules(uint32_t stamp, uint32_t now) {
  const char *results[] = {"NONE", "FIRST", "OK", "TOO_FAST"};
  queueLine(false, "RULE_STATE,%lu,%s,%lu,%lu,%lu,%s\n",
      (unsigned long)stamp, rules.ready ? "READY" : "CALIBRATING",
      (unsigned long)rules.count, (unsigned long)rules.previousMs,
      (unsigned long)(rules.count ? now - rules.lastEventMs : 0), results[rules.lastResult]);
}

void selectControl(paceloop::Mode mode, uint32_t now) {
  control.select(mode, now);
  rules.reset();
  led.indicate(mode, now);
  reportControl();
  if (mode == paceloop::RULES) reportRules(micros(), now);
}

void beginIMUBus() {
  // Core 1.6.0 lacks reset_on_timeout; reinitialize Wire after a failed transaction.
  Wire.begin();
  Wire.setClock(100000);
  Wire.setWireTimeout(20000, false);
#if defined(ARDUINO_UNOR4_MINIMA)
  // Use the breakout's pull-ups instead of weak pull-ups to the R4's 5V rail.
  R_IOPORT_PinCfg(&g_ioport_ctrl, g_pin_cfg[SDA].pin,
      IOPORT_CFG_DRIVE_MID | IOPORT_CFG_PERIPHERAL_PIN | IOPORT_PERIPHERAL_IIC);
  R_IOPORT_PinCfg(&g_ioport_ctrl, g_pin_cfg[SCL].pin,
      IOPORT_CFG_DRIVE_MID | IOPORT_CFG_PERIPHERAL_PIN | IOPORT_PERIPHERAL_IIC);
#endif
}

bool readRegisters(uint8_t address, uint8_t reg, uint8_t *out, size_t count) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(address, count, true) != count) return false;
  for (size_t i = 0; i < count; ++i) out[i] = Wire.read();
  return true;
}

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(imuAddress);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission(true) == 0;
}

void reportIMU() {
  queueLine(false, "IMU_STATUS,%s,0x%02x,0x%02x\n",
      imuReady ? "OK" : "NOT_FOUND", imuAddress, deviceId);
}

bool startIMU() {
  beginIMUBus();
  imuAddress = 0; deviceId = 0;
  for (uint8_t address = 0x68; address <= 0x69; ++address) {
    uint8_t identity = 0;
    if (readRegisters(address, 0x75, &identity, 1)) {
      imuAddress = address; deviceId = identity;
      if (identity == 0x68 || identity == 0x70) break;
    }
  }
  if (deviceId != 0x68 && deviceId != 0x70) return false;
  // PLL X gyro clock, all axes enabled, 21 Hz DLPF, 50 Hz sample rate,
  // +/-500 deg/s gyro and +/-4g acceleration.
  if (!writeRegister(0x6B, 0x01) || !writeRegister(0x6C, 0x00) ||
      !writeRegister(0x1A, 0x04) || !writeRegister(0x19, 19) ||
      !writeRegister(0x1B, 0x08) || !writeRegister(0x1C, 0x08)) return false;
  // MPU6500 has a separate accelerometer filter register (20 Hz).
  if (deviceId == 0x70) {
    uint8_t filter = 0;
    if (!writeRegister(0x1D, 0x04) ||
        !readRegisters(imuAddress, 0x1D, &filter, 1) || filter != 4) return false;
  }
  uint8_t values[4];
  if (!readRegisters(imuAddress, 0x19, values, 4)) return false;
  return values[0] == 19 && values[1] == 4 && values[2] == 8 && values[3] == 8;
}

int16_t signedWord(const uint8_t *bytes) {
  return (int16_t)(((uint16_t)bytes[0] << 8) | bytes[1]);
}

void sampleIMU() {
  uint8_t raw[14];
  if (!readRegisters(imuAddress, 0x3B, raw, sizeof(raw))) {
    imuReady = false;
    rules.reset();
    queueLine(false, "IMU_STATUS,READ_ERROR\n");
    return;
  }
  const float accelScale = 9.80665f / 8192.0f;
  const float gyroScale = 0.01745329252f / 65.5f;
  uint32_t stamp = micros(), now = millis();
  float a[3], g[3];
  for (int i = 0; i < 3; ++i) {
    a[i] = signedWord(raw + 2*i) * accelScale;
    g[i] = signedWord(raw + 8 + 2*i) * gyroScale;
  }
  float temperature = deviceId == 0x70
      ? signedWord(raw + 6) / 333.87f + 21.0f
      : signedWord(raw + 6) / 340.0f + 36.53f;
  paceloop::Event event = paceloop::NO_EVENT;
  if (control.mode == paceloop::RULES) {
    event = rules.update(now, a, g, targetInterval);
    if (event == paceloop::TOO_FAST) led.alert(now);
  }
  if (streaming) queueLine(true, "IMU,%lu,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.2f\n",
      (unsigned long)stamp, a[0], a[1], a[2], g[0], g[1], g[2], temperature);
  if (control.mode == paceloop::RULES && (event != paceloop::NO_EVENT || now - lastState >= 250)) {
    lastState = now; reportRules(stamp, now);
  }
}

void processCommand() {
  command[commandLength] = '\0';
  if (commandOverflow) { commandLength = 0; commandOverflow = false; return; }
  uint32_t now = millis();
  if (strcmp(command, "ALERT") == 0) {
    if (control.mode == paceloop::PC && control.hostLive(now)) {
      led.alert(now);
      queueLine(false, "ACK,ALERT\n");
    } else queueLine(false, "ERR,ALERT_REQUIRES_PC\n");
  } else if (strcmp(command, "HOST_PING") == 0) {
    control.ping(now);
    reportControl();
  } else if (strcmp(command, "SET_CONTROL,PC") == 0) {
    control.ping(now);
    selectControl(paceloop::PC, now);
  } else if (strcmp(command, "SET_CONTROL,RULES") == 0) {
    selectControl(paceloop::RULES, now);
  } else if (strcmp(command, "GET_CONTROL") == 0) {
    reportControl();
    if (control.mode == paceloop::RULES) reportRules(micros(), now);
  } else if (strcmp(command, "CALIBRATE") == 0) {
    rules.reset();
    if (control.mode == paceloop::RULES) reportRules(micros(), now);
  } else if (strcmp(command, "GET_MODE") == 0) {
    queueLine(false, "MODE,%lu\n", (unsigned long)targetInterval);
  } else if (strcmp(command, "GET_IMU") == 0) {
    reportIMU();
  } else if (strcmp(command, "STREAM,1") == 0) {
    streaming = true;
  } else if (strcmp(command, "STREAM,0") == 0) {
    streaming = false;
  } else if (strcmp(command, "GET_INFO") == 0) {
    queueLine(false, "INFO,PaceLoop_IMU_v3,%s,50Hz,4g,500dps,DUAL_MODE\n",
        deviceId == 0x70 ? "MPU6500-compatible" :
        deviceId == 0x68 ? "MPU6050-compatible" : "UNKNOWN");
  }
  commandLength = 0;
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_FAST, INPUT_PULLUP);
  pinMode(BUTTON_MED, INPUT_PULLUP);
  pinMode(BUTTON_SLOW, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  imuReady = startIMU();
  nextSample = micros() + 100000;
  led.indicate(paceloop::RULES, millis());
}

void loop() {
  const int pins[3] = {BUTTON_FAST, BUTTON_MED, BUTTON_SLOW};
  const uint32_t modes[3] = {3000, 5000, 15000};
  uint32_t now = millis();
  for (int i = 0; i < 3; ++i) {
    paceloop::Press press = buttons[i].update(digitalRead(pins[i]) == LOW, now, i == 2);
    if (press == paceloop::LONG_PRESS) {
      selectControl(control.mode == paceloop::PC ? paceloop::RULES : paceloop::PC, now);
    } else if (press == paceloop::SHORT_PRESS && targetInterval != modes[i]) {
      targetInterval = modes[i];
      queueLine(false, "MODE,%lu\n", (unsigned long)targetInterval);
    }
  }
  // Process at most 64 command bytes per loop to bound sampling latency.
  for (int budget = 0; budget < 64 && Serial.available(); ++budget) {
    char c = Serial.read();
    if (c == '\n') processCommand();
    else if (c != '\r') {
      if (commandLength < sizeof(command) - 1) command[commandLength++] = c;
      else commandOverflow = true;
    }
  }
  if (control.expire(now)) {
    rules.reset(); led.indicate(paceloop::RULES, now); reportControl();
    reportRules(micros(), now);
  }
  if (!imuReady && now - lastProbe >= 2000) {
    lastProbe = now; imuReady = startIMU();
    if (Serial) reportIMU();
    nextSample = micros() + 100000;
  }
  if (imuReady && (int32_t)(micros() - nextSample) >= 0) {
    nextSample = micros() + SAMPLE_US;
    sampleIMU();
  }
  bool light = led.value(millis());
  digitalWrite(LED_PIN, light ? HIGH : LOW);
  digitalWrite(LED_BUILTIN, light ? HIGH : LOW);
  flushOutput();
}
