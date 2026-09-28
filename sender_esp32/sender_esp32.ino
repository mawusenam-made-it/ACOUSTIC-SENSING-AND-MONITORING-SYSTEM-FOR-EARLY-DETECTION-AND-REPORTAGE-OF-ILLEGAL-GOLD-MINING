/*
  sender_esp32.ino
  ============================================================
  Acoustic Sentinel - SENDER unit (lives with the Raspberry Pi 5)

  Reads one JSON alert message per line from the Raspberry Pi over USB
  serial, and relays it out over LoRa to the ranger's handheld unit.

  ------------------------------------------------------------
  REQUIRED ARDUINO LIBRARIES
  (Arduino IDE: Sketch > Include Library > Manage Libraries... and search
  the EXACT name below, then click Install):

    1. "LoRa" by Sandeep Mistry
    2. "ArduinoJson" by Benoit Blanchon  (this code uses the v7 API)

  BOARD SETUP (one-time):
    File > Preferences > "Additional Boards Manager URLs" - paste:
      https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
    Then: Tools > Board > Boards Manager > search "esp32" > Install.
    Then: Tools > Board > select your specific ESP32 Dev Module.

  ------------------------------------------------------------
  WIRING: ESP32 Dev Kit <-> RA-02 (SX1278) LoRa module
    RA-02 VCC  -> ESP32 3.3V    *** NEVER connect to 5V - this WILL
                                    damage the RA-02 module permanently ***
    RA-02 GND  -> ESP32 GND
    RA-02 SCK  -> ESP32 GPIO18
    RA-02 MISO -> ESP32 GPIO19
    RA-02 MOSI -> ESP32 GPIO23
    RA-02 NSS  -> ESP32 GPIO5
    RA-02 RST  -> ESP32 GPIO14
    RA-02 DIO0 -> ESP32 GPIO2

  WIRING: ESP32 <-> Raspberry Pi 5
    Just one USB cable, ESP32's USB port to a Pi 5 USB port.
    Carries both power and the serial data.

  ------------------------------------------------------------
  *** CHECK YOUR MODULE'S FREQUENCY BAND BEFORE UPLOADING ***
  RA-02/SX1278 modules ship in different frequency variants (433 MHz,
  868 MHz, 915 MHz) - it's usually printed on the module or in the
  listing you bought it from. Using the wrong one is against most
  countries' radio regulations, and the two ends won't hear each
  other anyway. This defaults to 433 MHz, the most common RA-02
  variant and the license-free ISM band across most of Africa,
  Europe and Asia. Change LORA_FREQUENCY below if yours differs -
  and it MUST exactly match ranger_esp32.ino's setting too.
  ------------------------------------------------------------

  NOTE: this firmware was written and carefully reviewed against the
  documented APIs of the LoRa and ArduinoJson libraries, but could not
  be compiled in the environment that generated it (no access to the
  Arduino library index there). Please compile it once in Arduino IDE
  before flashing to hardware, and let me know if it throws any
  errors so we can fix them together.
*/

#include <SPI.h>
#include <LoRa.h>
#include <ArduinoJson.h>

// ---- LoRa radio pins (see wiring above) ----
#define LORA_SCK   18
#define LORA_MISO  19
#define LORA_MOSI  23
#define LORA_NSS   5
#define LORA_RST   14
#define LORA_DIO0  2

// ---- CHECK YOUR MODULE! Options: 433E6, 868E6, 915E6 ----
#define LORA_FREQUENCY 433E6

// ---- Must be IDENTICAL to ranger_esp32.ino ----
#define PACKET_MAGIC 0xA5

// ---- Must be in the SAME ORDER as CLASS_NAMES in pi_inference_app.py ----
const char* CLASS_NAMES[] = {
  "HeavyMachinery",   // 0
  "Chainsaw",          // 1
  "Forest (safe)",     // 2 - the Pi should never actually send this one
  "HandTools"          // 3
};

// Compact binary packet sent over LoRa. Kept small on purpose: every byte
// costs airtime and battery over a radio link. Total size: 10 bytes.
struct __attribute__((packed)) AlertPacket {
  uint8_t  magic;       // always PACKET_MAGIC - lets the receiver sanity-check
  uint8_t  class_id;    // 0-3, see CLASS_NAMES
  uint8_t  confidence;  // 0-100
  uint16_t seq;         // increasing counter - lets the receiver spot drops
  uint32_t timestamp;   // millis() on the sender, since its own last boot
  uint8_t  checksum;    // XOR of all bytes above - cheap integrity check
};

uint8_t computeChecksum(const AlertPacket &p) {
  const uint8_t* bytes = (const uint8_t*)&p;
  uint8_t chk = 0;
  // XOR every byte EXCEPT the checksum field itself (the last byte)
  for (size_t i = 0; i < sizeof(AlertPacket) - 1; i++) {
    chk ^= bytes[i];
  }
  return chk;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  Serial.println("Acoustic Sentinel - Sender booting...");

  LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("ERROR: LoRa radio not found - check your wiring!");
    while (true) { delay(1000); }
  }
  Serial.println("LoRa radio ready.");
  Serial.println("Waiting for alerts from the Raspberry Pi over serial...");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    JsonDocument doc;   // ArduinoJson v7: auto-sized, no template argument needed
    DeserializationError err = deserializeJson(doc, line);

    if (err) {
      Serial.print("JSON parse failed: ");
      Serial.println(err.c_str());
      return;
    }

    int classId = doc["class"] | -1;
    int conf    = doc["conf"]  | 0;
    int seq     = doc["seq"]   | 0;

    if (classId < 0 || classId > 3) {
      Serial.println("Ignoring message with invalid/missing class id.");
      return;
    }

    AlertPacket packet;
    packet.magic      = PACKET_MAGIC;
    packet.class_id    = (uint8_t)classId;
    packet.confidence  = (uint8_t)constrain(conf, 0, 100);
    packet.seq         = (uint16_t)seq;
    packet.timestamp   = millis();
    packet.checksum    = computeChecksum(packet);

    LoRa.beginPacket();
    LoRa.write((uint8_t*)&packet, sizeof(packet));
    LoRa.endPacket();

    Serial.print("Relayed over LoRa -> ");
    Serial.print(CLASS_NAMES[packet.class_id]);
    Serial.print(" (");
    Serial.print(packet.confidence);
    Serial.print("%), seq=");
    Serial.println(packet.seq);
  }
}
