# Hardware setup

## Connections

The firmware targets an Arduino UNO R4 Minima and an MPU6050-compatible or MPU6500-compatible I2C module.

| Module | UNO R4 Minima |
| --- | --- |
| SDA | SDA / A4 |
| SCL | SCL / A5 |
| GND | GND |
| VCC | Supply supported by the breakout board |

Check the breakout's regulator specification before using 5 V. A bare IMU chip and a regulated breakout have different supply requirements. The R4's SDA/A4 and SCL/A5 labels identify the same I2C bus; see the [Arduino reference](https://docs.arduino.cc/tutorials/uno-r4-minima/cheat-sheet/).

The firmware probes addresses `0x68` and `0x69`. It supports `WHO_AM_I=0x68` and `0x70`, and reports the detected interface. The tested HW123 breakout returns `0x70` and uses the MPU6500-compatible configuration.

The sketch uses the breakout's external pull-ups and disables the R4 core's internal 5 V pull-ups after bus initialization. Sampling is 50 Hz, with +/-4 g acceleration, +/-500 degrees/s angular velocity, and approximately 20 Hz low-pass filtering.

## Upload firmware

In Arduino IDE, select **Arduino UNO R4 Minima**, choose the board's port, and upload `arduino/PaceLoop_UNO_R4/PaceLoop_UNO_R4.ino`. The sketch uses the core Wire library; no additional sensor library is required.

With Arduino CLI and the Renesas UNO core installed:

```powershell
arduino-cli compile --fqbn arduino:renesas_uno:minima arduino/PaceLoop_UNO_R4
arduino-cli upload --fqbn arduino:renesas_uno:minima --port COM3 arduino/PaceLoop_UNO_R4
```

Close Serial Monitor, Serial Plotter, and other applications using the port before running PaceLoop.

## Buttons and feedback

Buttons connect D2, D3, or D4 to GND; the firmware enables internal pull-ups. They select 3-, 5-, or 15-second target intervals. The boot target is four seconds and appears as **CUSTOM**.

An `ALERT` command flashes the D9 LED output for three seconds. Use an appropriate series resistor for an external LED.

## Serial protocol

The link uses 115200 baud and newline-terminated ASCII messages.

```text
IMU,micros,ax,ay,az,gx,gy,gz,temp_C
MODE,milliseconds
IMU_STATUS,state,address,device_id
```

Acceleration is in m/s^2 and includes gravity. Angular velocity is in rad/s. The device timestamp is a wrapping 32-bit microsecond counter. Python handles timestamp rollover and detects device restarts.

| Command | Response or effect |
| --- | --- |
| `GET_MODE` | Current `MODE` value |
| `GET_INFO` | Firmware, sensor interface, sampling rate, and ranges |
| `GET_IMU` | Current IMU status |
| `STREAM,1` / `STREAM,0` | Enable / disable sample output |
| `ALERT` | Flash LED; return `ACK,ALERT` |

Read errors pause streaming while the firmware retries bus initialization. A stream timeout pauses following and timing. Reconnect USB and restart the application after a USB disconnection; calibration and lift teaching are repeated after a sensor restart or a long sample gap.
