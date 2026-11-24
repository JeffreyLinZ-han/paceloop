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

In Arduino IDE, select **Arduino UNO R4 Minima**, choose the board's port, and upload `firmware/PaceLoop_UNO_R4/PaceLoop_UNO_R4.ino`. The sketch uses the core Wire library; no additional sensor library is required.

With Arduino CLI and the Renesas UNO core installed:

```powershell
arduino-cli compile --fqbn arduino:renesas_uno:minima firmware/PaceLoop_UNO_R4
arduino-cli upload --fqbn arduino:renesas_uno:minima --port COM3 firmware/PaceLoop_UNO_R4
```

Close Serial Monitor, Serial Plotter, and other applications using the port before running PaceLoop.

## Buttons and feedback

Buttons connect D2, D3, or D4 to GND; the firmware enables internal pull-ups.
All buttons have 30 ms debounce. A short press is applied on release.

| Button | Short press | Hold |
| --- | --- | --- |
| D2 | 3-second target | No extra action |
| D3 | 5-second target | No extra action |
| D4 | 15-second target | After 1.2 s, toggle PC model / Arduino rules once |

Releasing D4 after a hold does not select the 15-second target. The selected target
survives a mode change; gesture history does not. Boot defaults are **Arduino
rules** and a four-second target (**CUSTOM**). Power cycling restores those defaults.

Both D9 and `LED_BUILTIN` blink with 150 ms on/off phases for three seconds after
a too-short detected interval. A first gesture starts the timer without warning.
Use an appropriate series resistor for an external LED. Mode indication uses
300 ms pulses: one for rules, two for PC. Overspeed alerts take priority.

## Standalone rules and PC inference

The same sketch contains both modes. No model or computer is needed for rules;
the board and IMU still need power. Sampling continues at 50 Hz regardless of
USB connection, `STREAM,0`, or whether a serial reader consumes the output.
USB output is buffered and may drop telemetry under backpressure, without
blocking detection or LED updates.

In rules mode, keep the neutral forearm pose still for 100 samples (about two
seconds). The detector estimates neutral gravity direction and gyro bias, rejects
moving calibration windows, and low-pass filters the accelerometer direction.
A candidate requires tilt above 25 degrees and corrected angular speed above
0.15 rad/s for 80 ms. Returning below 10 degrees for 150 ms rearms detection;
events less than 1.2 seconds apart are rejected. Strong linear acceleration is
rejected. A sensor read error or sample gap over 250 ms resets calibration and
timing. Thresholds live in `firmware/PaceLoop_UNO_R4/PaceLoopCore.h`.

This is a movement proxy, not a reliable eating classifier: yaw-only motion may
be missed, and scratching can trigger it. PC mode uses the bundled simulation-
trained classifier instead. Python performs inference and sends `ALERT`; the
board runs LED timing. In rules mode Python displays board counts and does not
send a second alert or run classifier inference. Orientation estimation and arm
following remain available while connected in either mode.

The desktop sends `HOST_PING` once per second in either mode. A heartbeat never
changes the selected mode, so it cannot undo a D4 press. PC mode falls back to
rules after three seconds without a heartbeat (with a three-second startup
grace after a button selection). Closing PaceLoop requests rules immediately.
Keep still again after a fallback or mode switch into rules. Disconnecting the
only power source turns the board off; USB-free operation requires another
supported power source.

## Serial protocol

The link uses 115200 baud and newline-terminated ASCII messages.

```text
IMU,micros,ax,ay,az,gx,gy,gz,temp_C
MODE,milliseconds
IMU_STATUS,state,address,device_id
CONTROL,PC
CONTROL,RULES
RULE_STATE,micros,READY_or_CALIBRATING,count,previous_ms,age_ms,result
```

Acceleration is in m/s^2 and includes gravity. Angular velocity is in rad/s. The device timestamp is a wrapping 32-bit microsecond counter. Python handles timestamp rollover and detects device restarts.

| Command | Response or effect |
| --- | --- |
| `GET_MODE` | Current `MODE` value |
| `GET_INFO` | Firmware, sensor interface, sampling rate, and ranges |
| `GET_IMU` | Current IMU status |
| `GET_CONTROL` | Current `CONTROL` and, in rules mode, a `RULE_STATE` snapshot |
| `SET_CONTROL,PC` / `SET_CONTROL,RULES` | Select mode, clear gesture history, report `CONTROL` |
| `HOST_PING` | Refresh the PC heartbeat; report `CONTROL` without changing it |
| `CALIBRATE` | Reset standalone neutral calibration and gesture history |
| `STREAM,1` / `STREAM,0` | Enable / disable raw sample output; sensing and rules continue |
| `ALERT` | In PC mode with a fresh heartbeat, flash LEDs and return `ACK,ALERT`; otherwise `ERR,ALERT_REQUIRES_PC` |

`MODE` still means the pace target, independent of `CONTROL`. `GET_INFO` identifies
v3 firmware with `DUAL_MODE`. Rule snapshots are emitted every 250 ms and on an
event; they share the IMU microsecond clock. `age_ms` is time since the last event,
and `result` is `NONE`, `FIRST`, `OK`, or `TOO_FAST`. `previous_ms` is zero until the
second event. Snapshots allow a monitor to recover current counts after dropped
telemetry. They are not new events every time they arrive.

Old firmware retains the original stream and alert protocol, but must be updated
for standalone detection and physical mode switching. The desktop labels unknown
firmware and disables its mode button until v3 control messages are received.

Read errors pause streaming while the firmware retries bus initialization. A stream timeout pauses following and timing. Reconnect USB and restart the application after a USB disconnection; calibration and lift teaching are repeated after a sensor restart or a long sample gap.
