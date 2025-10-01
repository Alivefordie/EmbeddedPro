#include "BluetoothSerial.h"
#include <ESP32Servo.h>
#include <Arduino.h>
#include <LittleFS.h>
#include <AudioFileSourceLittleFS.h>
#include <AudioGeneratorWAV.h>
#include <AudioOutputI2S.h>

AudioGeneratorWAV*       wav;
AudioFileSourceLittleFS* file;
AudioOutputI2S*          out;
BluetoothSerial          SerialBT;

// ===== มอเตอร์ =====
#define MAX_PWM 255
int leftPins[3]  = { 18, 13, 14 };   // PWM, IN1, IN2
int rightPins[3] = { 19, 26, 27 };   // PWM, IN1, IN2

bool isPlaying = false;

Servo servo;
const int SERVO_PIN = 23;
const int LOCK_POS  = 0;
const int DROP_POS  = 90;

void motorDir(bool leftForward, int lpwm, bool rightForward, int rpwm) {
  lpwm = constrain(lpwm, 0, 255);
  rpwm = constrain(rpwm, 0, 255);

  digitalWrite(leftPins[1],  leftForward ? HIGH : LOW);
  digitalWrite(leftPins[2],  leftForward ? LOW  : HIGH);
  analogWrite(leftPins[0],   lpwm);

  digitalWrite(rightPins[1], rightForward ? HIGH : LOW);
  digitalWrite(rightPins[2], rightForward ? LOW  : HIGH);
  analogWrite(rightPins[0],  rpwm);
}

void motorStop() {
  analogWrite(leftPins[0],  0);
  analogWrite(rightPins[0], 0);
}

void play_sound() {
  if (isPlaying) return;                 // กันซ้ำ
  if (!LittleFS.exists("/help.wav")) {   // กันพลาด
    Serial.println("help.wav not found!");
    return;
  }
  file = new AudioFileSourceLittleFS("/help.wav");
  wav->begin(file, out);
  isPlaying = true;
}

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32_BT_BOAT");

  LittleFS.begin(true);

  out = new AudioOutputI2S(0, AudioOutputI2S::INTERNAL_DAC);
  out->SetPinout(25, -1, -1);   // บังคับใช้ DAC1 (GPIO25) อย่างเดียว
  out->SetGain(0.30f);
  out->SetRate(8000);           // ให้ตรงกับไฟล์ของคุณ (8k หรือ 16k)
  wav = new AudioGeneratorWAV();

  pinMode(leftPins[0],  OUTPUT);
  pinMode(leftPins[1],  OUTPUT);
  pinMode(leftPins[2],  OUTPUT);
  pinMode(rightPins[0], OUTPUT);
  pinMode(rightPins[1], OUTPUT);
  pinMode(rightPins[2], OUTPUT);

  // เซอร์โว
  servo.attach(SERVO_PIN);
  servo.write(LOCK_POS);

  motorStop();
}

void loop() {
  if (SerialBT.available()) {
    String cmd = SerialBT.readStringUntil('\n');
    cmd.trim();
    int dash = cmd.indexOf('-');

    if (dash > 0) {
      String action = cmd.substring(0, dash);
      int val = cmd.substring(dash + 1).toInt();
      val = constrain(val, 0, MAX_PWM);

      if      (action == "F")     motorDir(true,  val, true,  val);
      else if (action == "B")     motorDir(false, val, false, val);
      else if (action == "S")     motorStop();
      else if (action == "L")     motorDir(false, val, true,  val);
      else if (action == "R")     motorDir(true,  val, false, val);
      else if (action == "SOUND") play_sound();     
      else if (action == "DROP")  servo.write(DROP_POS);
      else if (action == "LOCK")  servo.write(LOCK_POS);
      else                        SerialBT.println("ERR: Unknown action");

      Serial.printf("ACTION %s=%d\n", action.c_str(), val);
      SerialBT.printf("OK %s=%d\n", action.c_str(), val);
    }
  }
  if (isPlaying && wav->isRunning()) {
    if (!wav->loop()) {
      wav->stop();
      delete file;       // cleanup source
      isPlaying = false; // เล่นจบแล้ว
    }
  }
}
