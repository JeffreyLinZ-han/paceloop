#ifndef PACELOOP_CORE_H
#define PACELOOP_CORE_H

#include <math.h>
#include <stdint.h>

namespace paceloop {

enum Mode { RULES, PC };
enum Press { NONE, SHORT_PRESS, LONG_PRESS };

class Button {
 public:
  Press update(bool down, uint32_t now, bool allowLong = false) {
    if (down != raw_) { raw_ = down; changed_ = now; }
    if (stable_ != raw_ && now - changed_ >= 30) {
      stable_ = raw_;
      if (stable_) { pressed_ = now; held_ = false; }
      else if (!held_) return SHORT_PRESS;
    }
    if (allowLong && stable_ && raw_ && !held_ && now - pressed_ >= 1200) {
      held_ = true;
      return LONG_PRESS;
    }
    return NONE;
  }
 private:
  bool raw_ = false, stable_ = false, held_ = false;
  uint32_t changed_ = 0, pressed_ = 0;
};

class Control {
 public:
  Mode mode = RULES;
  void ping(uint32_t now) { lastHost_ = now; hostSeen_ = true; }
  bool hostLive(uint32_t now) const { return hostSeen_ && now - lastHost_ < 3000; }
  void select(Mode value, uint32_t now) { mode = value; selected_ = now; }
  bool expire(uint32_t now) {
    // Three-second connection grace period after selecting PC mode.
    if (mode == PC && !hostLive(now) && now - selected_ >= 3000) {
      select(RULES, now);
      return true;
    }
    return false;
  }
 private:
  bool hostSeen_ = false;
  uint32_t lastHost_ = 0, selected_ = 0;
};

class Led {
 public:
  void alert(uint32_t now) { alert_ = true; started_ = now; pulses_ = 0; }
  void indicate(Mode mode, uint32_t now) {
    // Overspeed alerts take priority over mode indication.
    if (alert_ && now - started_ < 3000) return;
    alert_ = false; started_ = now; pulses_ = mode == PC ? 2 : 1;
  }
  bool value(uint32_t now) {
    uint32_t elapsed = now - started_;
    if (alert_) {
      if (elapsed < 3000) return (elapsed / 150) % 2 == 0;
      alert_ = false;
    }
    return elapsed < pulses_ * 600U && elapsed % 600U < 300;
  }
 private:
  bool alert_ = false;
  unsigned pulses_ = 0;
  uint32_t started_ = 0;
};

enum Event { NO_EVENT, FIRST, OK, TOO_FAST };

// Neutral-relative tilt detector; eating and face gestures can overlap.
class Rules {
 public:
  bool ready = false;
  uint32_t count = 0, previousMs = 0, lastEventMs = 0;
  Event lastResult = NO_EVENT;
  float tilt = 0;

  void reset() { *this = Rules(); }

  Event update(uint32_t now, const float a[3], const float g[3], uint32_t target) {
    if (seen_ && now - lastSample_ > 250) reset();
    seen_ = true; lastSample_ = now;
    float norm2 = 0, speed2 = 0;
    for (int i = 0; i < 3; ++i) {
      if (!isfinite(a[i]) || !isfinite(g[i])) { reset(); return NO_EVENT; }
      norm2 += a[i] * a[i];
      float rate = g[i] - bias_[i];
      speed2 += rate * rate;
    }
    float norm = sqrtf(norm2);
    if (!ready) {
      if (norm < 9.0f || norm > 10.6f || speed2 > 0.04f) {
        clearCalibration(); return NO_EVENT;
      }
      ++calibration_;
      for (int i = 0; i < 3; ++i) {
        sumA_[i] += a[i]; sumG_[i] += g[i];
        squareA_[i] += a[i] * a[i]; squareG_[i] += g[i] * g[i];
      }
      if (calibration_ < 100) return NO_EVENT;
      float varianceA = 0, varianceG = 0, meanNorm2 = 0;
      for (int i = 0; i < 3; ++i) {
        float ma = sumA_[i] / 100, mg = sumG_[i] / 100;
        varianceA += fmaxf(0, squareA_[i] / 100 - ma * ma);
        varianceG += fmaxf(0, squareG_[i] / 100 - mg * mg);
        meanNorm2 += ma * ma;
      }
      if (varianceA > 0.12f || varianceG > 0.0025f) {
        clearCalibration(); return NO_EVENT;
      }
      float meanNorm = sqrtf(meanNorm2);
      for (int i = 0; i < 3; ++i) {
        neutral_[i] = filtered_[i] = sumA_[i] / (100 * meanNorm);
        bias_[i] = sumG_[i] / 100;
      }
      ready = true;
      return NO_EVENT;
    }
    // Reject acceleration magnitudes inconsistent with gravity.
    if (norm < 7.8f || norm > 11.8f) {
      high_ = low_ = false; return NO_EVENT;
    }
    float filteredNorm2 = 0, dot = 0;
    for (int i = 0; i < 3; ++i) {
      filtered_[i] += 0.2f * (a[i] / norm - filtered_[i]);
      filteredNorm2 += filtered_[i] * filtered_[i];
      dot += filtered_[i] * neutral_[i];
    }
    if (filteredNorm2 < 0.01f) { high_ = low_ = false; return NO_EVENT; }
    float cosine = fmaxf(-1, fminf(1, dot / sqrtf(filteredNorm2)));
    tilt = acosf(cosine) * 57.2957795f;
    if (tilt < 10) {
      if (!low_) { low_ = true; lowSince_ = now; }
      if (now - lowSince_ >= 150) armed_ = true;
    } else low_ = false;
    if (!(armed_ && tilt > 25 && speed2 > 0.0225f)) {
      high_ = false; return NO_EVENT;
    }
    if (!high_) { high_ = true; highSince_ = now; }
    if (now - highSince_ < 80) return NO_EVENT;
    armed_ = false; high_ = false;
    if (count && now - lastEventMs < 1200) return NO_EVENT;
    previousMs = count ? now - lastEventMs : 0;
    lastResult = count ? (previousMs < target ? TOO_FAST : OK) : FIRST;
    lastEventMs = now; ++count;
    return lastResult;
  }
 private:
  bool seen_ = false, armed_ = true, high_ = false, low_ = false;
  uint32_t lastSample_ = 0, highSince_ = 0, lowSince_ = 0;
  unsigned calibration_ = 0;
  float neutral_[3] = {}, filtered_[3] = {}, bias_[3] = {};
  float sumA_[3] = {}, sumG_[3] = {}, squareA_[3] = {}, squareG_[3] = {};
  void clearCalibration() {
    calibration_ = 0;
    for (int i = 0; i < 3; ++i) sumA_[i] = sumG_[i] = squareA_[i] = squareG_[i] = 0;
  }
};

}  // namespace paceloop
#endif
