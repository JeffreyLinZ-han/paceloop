// PaceLoop UNO R4 Minima: buttons + LED + MPU6050/MPU6500 over Wire.
// Serial at 115200: IMU,micros,ax,ay,az,gx,gy,gz,temp_C
// Acceleration is m/s^2 (includes gravity); gyro is rad/s.
#include <Wire.h>
#include <string.h>

const int BUTTON_FAST = 2, BUTTON_MED = 3, BUTTON_SLOW = 4, LED_PIN = 9;
const uint32_t SAMPLE_US = 20000;  // 50 Hz
uint32_t targetInterval = 4000;  // Boot target interval: four seconds.
uint32_t alertStart = 0, lastBlink = 0, nextSample = 0, lastProbe = 0;
bool alertActive = false, ledState = false, imuReady = false, streaming = true;
uint8_t imuAddress = 0, deviceId = 0;
char command[64];
size_t commandLength = 0;

void beginIMUBus() {
  // Reinitialize the R4 controller after a failed transaction. Its Wire
  // timeout API does not implement reset_on_timeout in core 1.6.0.
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
  Serial.print("IMU_STATUS,");
  Serial.print(imuReady ? "OK" : "NOT_FOUND");
  Serial.print(",0x"); Serial.print(imuAddress, HEX);
  Serial.print(",0x"); Serial.println(deviceId, HEX);
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

void sendIMU() {
  uint8_t raw[14];
  if (!readRegisters(imuAddress, 0x3B, raw, sizeof(raw))) {
    imuReady = false;
    Serial.println("IMU_STATUS,READ_ERROR");
    return;
  }
  const float accelScale = 9.80665f / 8192.0f;
  const float gyroScale = 0.01745329252f / 65.5f;
  Serial.print("IMU,"); Serial.print(micros());
  for (int i = 0; i < 3; ++i) { Serial.print(','); Serial.print(signedWord(raw + 2*i) * accelScale, 5); }
  for (int i = 0; i < 3; ++i) { Serial.print(','); Serial.print(signedWord(raw + 8 + 2*i) * gyroScale, 5); }
  float temperature = deviceId == 0x70
      ? signedWord(raw + 6) / 333.87f + 21.0f
      : signedWord(raw + 6) / 340.0f + 36.53f;
  Serial.print(','); Serial.println(temperature, 2);
}

void processCommand() {
  command[commandLength] = '\0';
  if (strcmp(command, "ALERT") == 0) {
    alertActive = true; alertStart = lastBlink = millis(); ledState = true;
    digitalWrite(LED_PIN, HIGH);
    Serial.println("ACK,ALERT");
  } else if (strcmp(command, "GET_MODE") == 0) {
    Serial.print("MODE,"); Serial.println(targetInterval);
  } else if (strcmp(command, "GET_IMU") == 0) {
    reportIMU();
  } else if (strcmp(command, "STREAM,1") == 0) {
    streaming = true;
  } else if (strcmp(command, "STREAM,0") == 0) {
    streaming = false;
  } else if (strcmp(command, "GET_INFO") == 0) {
    Serial.print("INFO,PaceLoop_IMU_v2,");
    Serial.print(deviceId == 0x70 ? "MPU6500-compatible" :
                 deviceId == 0x68 ? "MPU6050-compatible" : "UNKNOWN");
    Serial.println(",50Hz,4g,500dps");
  }
  commandLength = 0;
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_FAST, INPUT_PULLUP);
  pinMode(BUTTON_MED, INPUT_PULLUP);
  pinMode(BUTTON_SLOW, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  delay(100);
  imuReady = startIMU();
  nextSample = micros() + 100000;
}

void loop() {
  const int pins[3] = {BUTTON_FAST, BUTTON_MED, BUTTON_SLOW};
  const uint32_t modes[3] = {3000, 5000, 15000};
  for (int i = 0; i < 3; ++i) {
    if (digitalRead(pins[i]) == LOW && targetInterval != modes[i]) {
      targetInterval = modes[i];
      Serial.print("MODE,"); Serial.println(targetInterval);
    }
  }
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') processCommand();
    else if (c != '\r' && commandLength < sizeof(command) - 1) command[commandLength++] = c;
  }
  uint32_t now = millis();
  if (alertActive) {
    if (now - alertStart >= 3000) {
      alertActive = false; ledState = false; digitalWrite(LED_PIN, LOW);
    } else if (now - lastBlink >= 150) {
      lastBlink = now; ledState = !ledState; digitalWrite(LED_PIN, ledState);
    }
  }
  if (!imuReady && now - lastProbe >= 2000) {
    lastProbe = now; imuReady = startIMU();
    if (Serial) reportIMU();
    nextSample = micros() + 100000;
  }
  if (imuReady && streaming && Serial && (int32_t)(micros() - nextSample) >= 0) {
    nextSample = micros() + SAMPLE_US;
    sendIMU();
  }
}
