# EMG-Controlled Robotic Hand

A robotic hand control system that uses EMG (electromyography) signals to detect finger movements and replicate them on a robotic hand. The system uses TinyML for gesture recognition and implements real-time control through Arduino-compatible boards.

## System Overview

The system consists of two main components:
1. A Seeed XIAO board that processes EMG signals and performs gesture recognition using TensorFlow Lite
2. An Arduino Uno that controls servo motors based on the recognized gestures

## Features

- Real-time EMG signal processing
- TensorFlow Lite-based gesture recognition
- Recognition of 6 different hand states:
  - Thumb flexion
  - Index finger flexion
  - Middle finger flexion
  - Ring finger flexion
  - Pinky finger flexion
  - Static (no movement)
- I2C communication between boards
- Servo motor control for individual finger movement

## Hardware Requirements

- Seeed XIAO board
- Arduino Uno
- EMG sensor
- 5 servo motors
- Robotic hand assembly
- Connecting wires

## Pin Connections

### XIAO Connections
- EMG sensor to A0
- SDA to Arduino Uno A4
- SCL to Arduino Uno A5
- Ground to Arduino Uno Ground

### Arduino Uno Connections
- Thumb servo to pin 7
- Index finger servo to pin 8
- Middle finger servo to pin 9
- Ring finger servo to pin 10
- Pinky finger servo to pin 11

## Software Requirements

- Arduino IDE
- TensorFlow Lite for Microcontrollers
- Required Arduino libraries:
  - Wire.h
  - Servo.h
  - TensorFlowLite_ESP32.h

## Installation

1. Clone this repository
2. Upload `EMG_Predictor_Xiao.ino` to your Seeed XIAO board
   - Ensure `model_data.h` and `normalization_params.h` are in the same directory
3. Upload `Servo_Controller_Uno.ino` to your Arduino Uno

## How It Works

1. The EMG sensor captures muscle signals from the user's arm
2. The XIAO processes these signals and uses a trained TensorFlow Lite model to classify the gesture
3. The classified gesture is sent to the Arduino Uno via I2C
4. The Uno controls the appropriate servo motors to move the robotic hand's fingers
