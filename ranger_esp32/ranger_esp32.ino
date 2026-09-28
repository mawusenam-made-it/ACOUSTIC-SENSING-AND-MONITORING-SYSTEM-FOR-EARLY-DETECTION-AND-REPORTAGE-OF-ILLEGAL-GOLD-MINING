/*
  ranger_esp32.ino
  ============================================================
  Acoustic Sentinel - RANGER'S HANDHELD unit
  (Updated for XFP1116-07AY OLED, which uses the SH1106 driver chip,
  NOT SSD1306 - using Adafruit_SH110X library instead)

  ------------------------------------------------------------
  REQUIRED ARDUINO LIBRARIES:
    1. "LoRa" by Sandeep Mistry
    2. "Adafruit SH110X" by Adafruit
       (installing this will prompt "Adafruit GFX Library" and
        "Adafruit BusIO" as dependencies - say YES to both)

  WIRING: ESP32 Dev Kit <-> RA-02 (SX1278) LoRa module
    RA-02 VCC  -> ESP32 3.3V    *** NEVER 5V - will damage the module ***
    RA-02 GND  -> ESP32 GND
    RA-02 SCK  -> ESP32 GPIO18
    RA-02 MISO -> ESP32 GPIO19
    RA-02 MOSI -> ESP32 GPIO23
    RA-02 NSS  -> ESP32 GPIO5
    RA-02 RST  -> ESP32 GPIO14
    RA-02 DIO0 -> ESP32 GPIO2

  WIRING: ESP32 <-> XFP1116-07AY OLED (SH1106, I2C)
    OLED VCC -> ESP32 3.3V
    OLED GND -> ESP32 GND
    OLED SDA -> ESP32 GPIO21
    OLED SCL -> ESP32 GPIO22

  *** LORA_FREQUENCY BELOW MUST EXACTLY MATCH sender_esp32.ino ***
*/

#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

// ---- LoRa radio pins ----
#define LORA_SCK   18
#define LORA_MISO  19
#define LORA_MOSI  23
#define LORA_NSS   5
#define LORA_RST   14
#define LORA_DIO0  2

// ---- Must match sender_esp32.ino EXACTLY ----
#define LORA_FREQUENCY 433E6
#define PACKET_MAGIC 0xA5

// ---- OLED display setup (SH1106 driver) ----
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SH1106G display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ---- Must be in the SAME ORDER as CLASS_NAMES in pi_inference_app.py ----
const char* CLASS_NAMES[] = {
  "HEAVY MACHINERY",
  "CHAINSAW",
  "Forest (safe)",
  "HAND TOOLS"
};

struct __attribute__((packed)) AlertPacket {
  uint8_t  magic;
  uint8_t  class_id;
  uint8_t  confidence;
  uint16_t seq;
  uint32_t timestamp;
  uint8_t  checksum;
};

bool haveSeenAnyPacket = false;
unsigned long lastPacketMillis = 0;
unsigned long alertDisplayUntil = 0;
const unsigned long ALERT_DISPLAY_DURATION_MS = 8000;  // keep an alert on screen for 8s

uint8_t computeChecksum(const AlertPacket &p) {
  const uint8_t* bytes = (const uint8_t*)&p;
  uint8_t chk = 0;
  for (size_t i = 0; i < sizeof(AlertPacket) - 1; i++) {
    chk ^= bytes[i];
  }
  return chk;
}

void showIdleScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SH110X_WHITE);
  display.setCursor(0, 0);
  display.println("Acoustic Sentinel");
  display.println("--------------------");
  display.println("Listening...");
  if (haveSeenAnyPacket) {
    display.print("Last alert: ");
    display.print((millis() - lastPacketMillis) / 1000);
    display.println("s ago");
  } else {
    display.println("No alerts yet.");
  }
  display.display();
}

void showAlertScreen(const AlertPacket &packet) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 0);
  display.println("ALERT!");

  display.setTextSize(1);
  display.setCursor(0, 20);
  display.println(CLASS_NAMES[packet.class_id]);

  display.setCursor(0, 35);
  display.print("Confidence: ");
  display.print(packet.confidence);
  display.println("%");

  display.setCursor(0, 50);
  display.print("Seq #");
  display.println(packet.seq);

  display.display();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  Serial.println("Acoustic Sentinel - Ranger unit booting...");

  Wire.begin(21, 22);  // SDA, SCL
  // 0x3C is the most common I2C address for these OLEDs; if begin()
  // fails, try 0x3D instead (printed on some modules' back side).
  if (!display.begin(0x3C, true)) {
    Serial.println("ERROR: OLED not found - check wiring, or try address 0x3D.");
    while (true) { delay(1000); }
  }
  display.clearDisplay();
  display.display();

  LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("ERROR: LoRa radio not found - check your wiring!");
    while (true) { delay(1000); }
  }

  Serial.println("LoRa radio ready. Listening for alerts...");
  showIdleScreen();
}

void loop() {
  int packetSize = LoRa.parsePacket();

  if (packetSize == sizeof(AlertPacket)) {
    AlertPacket packet;
    LoRa.readBytes((uint8_t*)&packet, sizeof(packet));

    if (packet.magic != PACKET_MAGIC) {
      Serial.println("Dropped packet: bad magic byte (corrupted, or not our protocol).");
    } else if (packet.checksum != computeChecksum(packet)) {
      Serial.println("Dropped packet: checksum mismatch (corrupted in transit).");
    } else if (packet.class_id > 3) {
      Serial.println("Dropped packet: invalid class id.");
    } else {
      Serial.print("ALERT RECEIVED: ");
      Serial.print(CLASS_NAMES[packet.class_id]);
      Serial.print(" (");
      Serial.print(packet.confidence);
      Serial.print("%), seq=");
      Serial.println(packet.seq);

      haveSeenAnyPacket = true;
      lastPacketMillis = millis();

      showAlertScreen(packet);
      alertDisplayUntil = millis() + ALERT_DISPLAY_DURATION_MS;

      // OPTIONAL: wire a buzzer/vibration motor to a spare GPIO pin and
      // trigger it here, so the ranger notices even without looking at
      // the screen.
    }
  } else if (packetSize > 0) {
    Serial.println("Dropped packet: unexpected size (not our protocol).");
  }

  static unsigned long lastIdleRedraw = 0;
  if (millis() > alertDisplayUntil && millis() - lastIdleRedraw > 1000) {
    showIdleScreen();
    lastIdleRedraw = millis();
  }
}