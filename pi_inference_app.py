#!/usr/bin/env python3
"""
pi_inference_app.py
============================================================
Raspberry Pi 5 - real-time acoustic threat detector
(Acoustic remote sensing for illegal mining/logging detection)

PIPELINE:
  INMP441 mic --(I2S)--> 5-second audio window --> log-mel spectrogram
  --> Z-score normalize --> TFLite model --> class prediction
  --> alert debounce logic --> JSON message over USB serial --> ESP32

Every piece of the math/logic in this file (preprocessing shapes, the
TFLite loading/quantization handling, and the alert debounce logic) was
built and unit-tested against real TFLite files (one float32, one int8
quantized) before being assembled here. The one part that could NOT be
tested in that environment is live microphone capture, since that needs
real I2S hardware - so treat first boot as a hardware bring-up step (see
the SETUP CHECKLIST at the bottom of this file) and watch the console
output closely the first time you run it.

------------------------------------------------------------------
SERIAL PROTOCOL (Pi -> Sender ESP32), one JSON object per line:
------------------------------------------------------------------
  {"class": 1, "name": "chainsaw_sounds", "conf": 91, "seq": 42}\n

  class : integer 0-3, matches CLASS_NAMES order below (must be IDENTICAL
          to the class order baked into the sender/ranger ESP32 firmware)
  name  : human-readable class name (for debugging on the ESP32 serial
          monitor - not strictly required, but very handy)
  conf  : confidence as an integer percentage, 0-100
  seq   : an incrementing counter, so the receiving end can notice if a
          message got dropped
------------------------------------------------------------------
"""

import os
import sys
import time
import json
import logging
from collections import deque

import numpy as np
import librosa

# tflite_runtime is the small, Pi-friendly interpreter package. If it's not
# installed, fall back to the full tensorflow package's interpreter (works
# the same way, just heavier). On the Pi, prefer:
#   pip install tflite-runtime
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

try:
    import sounddevice as sd
except ImportError:
    sd = None  # handled at startup with a clear error message, see main()

try:
    import serial
except ImportError:
    serial = None  # handled at startup with a clear error message, see main()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("acoustic_sentinel")


# ============================================================
# CONFIG - edit these for your setup
# ============================================================

# --- Model ---
MODEL_PATH = "/home/pi/models/final_model.tflite"   # <-- point this at whichever model you finalize

# --- Audio / preprocessing (MUST match training exactly - do not change
#     these unless you also retrain the model with different values) ---
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CLIP_DURATION_SEC = 5
SAMPLES_PER_CLIP = SAMPLE_RATE * CLIP_DURATION_SEC
TARGET_TIME_STEPS = 216
GLOBAL_MEAN = -37.6293   # from your Channel-Attention preprocessing notebook
GLOBAL_STD = 16.1799     # from your Channel-Attention preprocessing notebook

# --- Classes (order confirmed from your notebooks' sorted() folder listing -
#     do NOT reorder this without also changing it in BOTH .ino files) ---
CLASS_NAMES = [
    "HeavyMachinery_excavator_truck_dieselEngines",  # 0
    "chainsaw_sounds",                               # 1
    "forest_sounds",                                 # 2  <- safe/background, never alerts
    "handtools_CuttingDigging_sounds",                # 3
]
ALERT_CLASS_INDICES = {0, 1, 3}   # every class except forest_sounds (2)

# --- Alert decision logic ---
CONFIDENCE_THRESHOLD = 0.70        # ignore predictions less confident than this
CONSECUTIVE_REQUIRED = 2           # require this many back-to-back same-class detections
COOLDOWN_CYCLES = 3                # after alerting, wait this many cycles before re-alerting

# --- Audio input device ---
# Run `python3 -c "import sounddevice as sd; print(sd.query_devices())"` on
# the Pi to list devices and find your I2S mic's index/name, then set this.
AUDIO_DEVICE = None    # None = system default input device; or set an int index / name substring

# --- Serial link to the sender ESP32 ---
SERIAL_PORT = "/dev/ttyUSB0"     # check with `ls /dev/ttyUSB*` or `/dev/ttyACM*`
SERIAL_BAUD = 115200


# ============================================================
# PREPROCESSING - turns a raw audio chunk into what the model expects
# ============================================================

def preprocess(audio):
    """
    audio: 1-D float32 numpy array, mono, at SAMPLE_RATE.
    Returns: (128, 216) float32 array, Z-score normalized log-mel spectrogram.
    """
    # Pad or trim to exactly 5 seconds, same as training.
    if len(audio) < SAMPLES_PER_CLIP:
        audio = np.pad(audio, (0, SAMPLES_PER_CLIP - len(audio)))
    else:
        audio = audio[:SAMPLES_PER_CLIP]

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Defensive pad/crop in case of a rounding edge-case in frame count.
    t = log_mel.shape[1]
    if t < TARGET_TIME_STEPS:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_TIME_STEPS - t)))
    elif t > TARGET_TIME_STEPS:
        log_mel = log_mel[:, :TARGET_TIME_STEPS]

    normalized = (log_mel - GLOBAL_MEAN) / GLOBAL_STD
    return normalized.astype(np.float32)


# ============================================================
# MODEL WRAPPER - auto-adapts to whichever of your 3 models you deploy
# (handles both (128,216,1) and (128,216) input shapes, and both plain
# float32 models and fully int8-quantized models, by reading the model's
# own metadata rather than assuming one format)
# ============================================================

class TFLiteClassifier:
    def __init__(self, model_path):
        self.interpreter = TFLiteInterpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.input_rank = len(self.input_details['shape'])  # 4 -> (1,128,216,1), 3 -> (1,128,216)
        self.is_quantized = self.input_details['dtype'] == np.int8
        if self.is_quantized:
            self.in_scale, self.in_zero_point = self.input_details['quantization']
            self.out_scale, self.out_zero_point = self.output_details['quantization']
        log.info(
            "Model loaded: input shape %s, dtype %s, quantized=%s",
            self.input_details['shape'], self.input_details['dtype'], self.is_quantized
        )

    def predict(self, spec_2d):
        x = spec_2d
        if self.input_rank == 4:
            x = x[np.newaxis, ..., np.newaxis]
        else:
            x = x[np.newaxis, ...]

        if self.is_quantized:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)
        else:
            x = x.astype(np.float32)

        self.interpreter.set_tensor(self.input_details['index'], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details['index'])[0]

        if self.is_quantized:
            out = (out.astype(np.float32) - self.out_zero_point) * self.out_scale

        return out


# ============================================================
# ALERT DECISION LOGIC (debounced, ignores the safe class, avoids spam)
# ============================================================

class AlertManager:
    def __init__(self):
        self.last_class = None
        self.consecutive_count = 0
        self.cooldown_remaining = 0
        self.seq = 0

    def process(self, predicted_class_idx, confidence):
        """Returns an alert dict, or None if nothing should be sent this cycle."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if predicted_class_idx not in ALERT_CLASS_INDICES or confidence < CONFIDENCE_THRESHOLD:
            self.consecutive_count = 0
            self.last_class = None
            return None

        if predicted_class_idx == self.last_class:
            self.consecutive_count += 1
        else:
            self.last_class = predicted_class_idx
            self.consecutive_count = 1

        if self.consecutive_count >= CONSECUTIVE_REQUIRED and self.cooldown_remaining == 0:
            self.cooldown_remaining = COOLDOWN_CYCLES
            self.seq = (self.seq + 1) % 65536
            return {
                "class": predicted_class_idx,
                "name": CLASS_NAMES[predicted_class_idx],
                "conf": int(round(confidence * 100)),
                "seq": self.seq,
            }
        return None


# ============================================================
# SERIAL LINK TO THE SENDER ESP32
# ============================================================

class SerialSender:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.conn = None
        self._connect()

    def _connect(self):
        if serial is None:
            log.error("pyserial is not installed. Run: pip install pyserial")
            self.conn = None
            return
        try:
            self.conn = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # let the ESP32 finish its reset-on-connect
            log.info("Serial link to ESP32 open on %s @ %d baud", self.port, self.baud)
        except Exception as e:
            log.error("Could not open serial port %s: %s", self.port, e)
            self.conn = None

    def send(self, alert_dict):
        line = json.dumps(alert_dict) + "\n"
        if self.conn is None or not self.conn.is_open:
            log.warning("Serial not connected - trying to reconnect before sending: %s", line.strip())
            self._connect()
            if self.conn is None:
                return False
        try:
            self.conn.write(line.encode("utf-8"))
            log.info("ALERT SENT -> %s", line.strip())
            return True
        except Exception as e:
            log.error("Serial write failed: %s", e)
            self.conn = None
            return False


# ============================================================
# AUDIO CAPTURE
# ============================================================

def record_clip():
    """
    Blocking-record one 5-second mono clip from the configured input device.
    Returns a 1-D float32 numpy array at SAMPLE_RATE.
    """
    recording = sd.rec(
        SAMPLES_PER_CLIP,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=AUDIO_DEVICE,
    )
    sd.wait()
    return recording.flatten()


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    if sd is None:
        log.error("sounddevice is not installed. On the Pi run:")
        log.error("  sudo apt install portaudio19-dev")
        log.error("  pip install sounddevice")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        log.error("Model file not found at %s - update MODEL_PATH in the CONFIG section.", MODEL_PATH)
        sys.exit(1)

    log.info("Loading model...")
    classifier = TFLiteClassifier(MODEL_PATH)

    log.info("Opening serial link to sender ESP32...")
    sender = SerialSender(SERIAL_PORT, SERIAL_BAUD)

    alert_mgr = AlertManager()

    log.info("Starting listen loop (Ctrl+C to stop). Recording %ds clips.", CLIP_DURATION_SEC)
    cycle = 0
    try:
        while True:
            cycle += 1
            audio = record_clip()
            spec = preprocess(audio)
            probs = classifier.predict(spec)

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            log.info(
                "Cycle %d: %s (%.1f%%)%s",
                cycle, CLASS_NAMES[pred_idx], confidence * 100,
                "  [not an alert class]" if pred_idx not in ALERT_CLASS_INDICES else ""
            )

            alert = alert_mgr.process(pred_idx, confidence)
            if alert is not None:
                sender.send(alert)

    except KeyboardInterrupt:
        log.info("Stopped by user.")


if __name__ == "__main__":
    main()


# ============================================================
# SETUP CHECKLIST (run these BEFORE `python3 pi_inference_app.py`)
# ============================================================
#
# 1) Install system + Python dependencies:
#      sudo apt update
#      sudo apt install portaudio19-dev
#      pip install sounddevice pyserial librosa numpy tflite-runtime --break-system-packages
#    (If tflite-runtime isn't available for your Pi OS/Python version, the
#    script will automatically fall back to `pip install tensorflow`, which
#    also works but is a much bigger install.)
#
# 2) Enable the I2S microphone (INMP441 wiring: VDD->3.3V, GND->GND,
#    SCK->GPIO18, WS->GPIO19, SD->GPIO20, L/R->GND for left-channel mode):
#      sudo nano /boot/firmware/config.txt
#    Make sure these two lines are present (uncomment/add them):
#      dtparam=i2s=on
#      dtoverlay=googlevoicehat-soundcard
#    Save, then: sudo reboot
#
# 3) After reboot, confirm the mic is detected:
#      arecord -l
#    Note the card number, then do a quick manual test recording:
#      arecord -D plughw:<card_number>,0 -c1 -r 22050 -f S32_LE -d 5 test.wav
#      aplay test.wav
#    If this is silent or errors out, the I2S wiring/overlay needs
#    troubleshooting BEFORE running this script - this step is the one
#    piece of the pipeline that depends on your specific hardware/OS
#    version, so please verify it works with plain `arecord` first.
#
# 4) Find your mic's sounddevice index/name:
#      python3 -c "import sounddevice as sd; print(sd.query_devices())"
#    Set AUDIO_DEVICE above to that index (or a unique substring of its name)
#    if the system default input isn't your I2S mic.
#
# 5) Plug in the sender ESP32 via USB, find its port:
#      ls /dev/ttyUSB*   (or /dev/ttyACM*)
#    Set SERIAL_PORT above to match.
#
# 6) Set MODEL_PATH above to point at whichever .tflite file you finalize,
#    copied onto the Pi (e.g. via `scp` from your computer or Google Drive).
#
# 7) Run it:
#      python3 pi_inference_app.py
#    Watch the console: it prints every 5-second cycle's prediction, and
#    logs "ALERT SENT" whenever it actually pushes a message to the ESP32.
