/*
  Pan servo controller for the person tracker.

  Protocol (from person_tracker.py), 115200 baud, one command per line:
      P,<angle>\n        e.g. "P,92\n"   (angle in degrees, integer)

  Behaviour:
    - Clamps the angle to ANGLE_MIN..ANGLE_MAX.
    - Limits how fast the servo moves (MAX_STEP_DEG per UPDATE_MS) so
      sudden commands do not slam the servo.
    - If no valid command arrives for CMD_TIMEOUT_MS, the servo holds its
      current position (or returns home if RETURN_HOME_ON_TIMEOUT is true).
    - Ignores malformed lines.

  Wiring (hobby servo):
    Servo signal (orange/yellow/white) -> D9
    Servo power (red)                  -> external 5V supply (NOT the Uno 5V pin
                                          for anything bigger than a micro servo)
    Servo ground (brown/black)         -> external supply GND AND Uno GND
                                          (grounds must be common)

  Quick test without Python: open the Arduino Serial Monitor at 115200 baud,
  set line ending to "Newline", and send e.g.  P,45   then   P,135
*/

#include <Servo.h>

// ---- Configuration ----
const uint8_t       SERVO_PIN              = 9;
const unsigned long BAUD                   = 115200;
const int           ANGLE_MIN              = 0;     // narrow these to your mount's safe range
const int           ANGLE_MAX              = 180;
const int           ANGLE_HOME             = 90;
const unsigned long CMD_TIMEOUT_MS         = 500;   // hold if no command for this long
const bool          RETURN_HOME_ON_TIMEOUT = false; // true: go to ANGLE_HOME instead of holding
const float         MAX_STEP_DEG           = 3.0;   // max movement per update tick
const unsigned long UPDATE_MS              = 15;    // servo update period

// ---- State ----
Servo panServo;
char lineBuf[16];
uint8_t lineLen = 0;
bool lineOverflow = false;

float currentAngle = ANGLE_HOME;
int   targetAngle  = ANGLE_HOME;

unsigned long lastCmdMs    = 0;
unsigned long lastUpdateMs = 0;
bool timedOut = true;

void handleLine(const char *line) {
  // Expect "P,<int>"
  if (line[0] != 'P' || line[1] != ',') return;

  char *endPtr = nullptr;
  long value = strtol(line + 2, &endPtr, 10);
  if (endPtr == line + 2) return;        // no digits parsed
  if (*endPtr != '\0') return;           // trailing junk

  if (value < ANGLE_MIN) value = ANGLE_MIN;
  if (value > ANGLE_MAX) value = ANGLE_MAX;

  targetAngle = (int)value;
  lastCmdMs = millis();
  timedOut = false;
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      if (!lineOverflow && lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
      }
      lineLen = 0;
      lineOverflow = false;
    } else if (c == '\r') {
      // ignore
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineOverflow = true;  // discard this line until the next newline
    }
  }
}

void checkTimeout() {
  if (!timedOut && (millis() - lastCmdMs) > CMD_TIMEOUT_MS) {
    timedOut = true;
    if (RETURN_HOME_ON_TIMEOUT) {
      targetAngle = ANGLE_HOME;
    } else {
      targetAngle = (int)round(currentAngle);  // hold where we are
    }
  }
}

void updateServo() {
  unsigned long now = millis();
  if (now - lastUpdateMs < UPDATE_MS) return;
  lastUpdateMs = now;

  float diff = (float)targetAngle - currentAngle;
  if (diff > MAX_STEP_DEG)       diff = MAX_STEP_DEG;
  else if (diff < -MAX_STEP_DEG) diff = -MAX_STEP_DEG;
  currentAngle += diff;

  panServo.write((int)round(currentAngle));
}

void setup() {
  Serial.begin(BAUD);
  panServo.attach(SERVO_PIN);
  panServo.write(ANGLE_HOME);
  lastUpdateMs = millis();
}

void loop() {
  readSerial();
  checkTimeout();
  updateServo();
}
