#include <Servo.h>
#include <Wire.h>

// Pin definitions
#define THUMB_PIN 7
#define INDEX_PIN 8
#define MIDDLE_PIN 9
#define RING_PIN 10
#define PINKY_PIN 11

// Create servo objects
Servo thumbServo;
Servo indexServo;
Servo middleServo;
Servo ringServo;
Servo pinkyServo;

// Tracking variables
bool isMoving = false;
int currentGesture = -1;

void setup() {
    // Initialize I2C as slave
    Wire.begin(8);
    Wire.onReceive(receiveEvent);
    
    // Initialize serial for debugging
    Serial.begin(9600);
    
    // Initialize all servos
    thumbServo.attach(THUMB_PIN);
    indexServo.attach(INDEX_PIN);
    middleServo.attach(MIDDLE_PIN);
    ringServo.attach(RING_PIN);
    pinkyServo.attach(PINKY_PIN);
    
    // Set all servos to neutral position
    resetServos();
    
    Serial.println("Uno initialized and ready");
}

void resetServos() {
    thumbServo.write(90);
    indexServo.write(90);
    middleServo.write(90);
    ringServo.write(90);
    pinkyServo.write(90);
}

void moveServo(Servo &servo) {
    // Forward motion
    servo.write(180);
    delay(2100);
    
    // Stop
    servo.write(90);
    delay(500);
    
    // Reverse motion
    servo.write(0);
    delay(2100);
    
    // Return to neutral
    servo.write(90);
}

void receiveEvent(int bytes) {
    int gesture = Wire.read();
    if (gesture != currentGesture) {
        currentGesture = gesture;
        isMoving = true;
        Serial.print("Received gesture: ");
        Serial.println(gesture);
    }
}

void loop() {
    if (isMoving && currentGesture >= 0) {
        switch(currentGesture) {
            case 0:  // Thumb
                moveServo(thumbServo);
                break;
            case 1:  // Index
                moveServo(indexServo);
                break;
            case 2:  // Middle
                moveServo(middleServo);
                break;
            case 3:  // Ring
                moveServo(ringServo);
                break;
            case 4:  // Pinky
                moveServo(pinkyServo);
                break;
            case 5:  // Static - no movement
                resetServos();
                break;
        }
        isMoving = false;
    }
}