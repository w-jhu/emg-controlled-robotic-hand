#include <TensorFlowLite_ESP32.h>
#include <Arduino.h>
#include "model_data.h"
#include "normalization_params.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include <Wire.h>

// Constants
const int WINDOW_SIZE = 500;
const int SPEC_HEIGHT = 32;
const int SPEC_WIDTH = 32;
const int kTensorArenaSize = 128 * 1024;  // 64KB

// Static allocation
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
float emg_buffer[WINDOW_SIZE];
float features[SPEC_HEIGHT][SPEC_WIDTH];
int buffer_index = 0;

// TFLite globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
static tflite::MicroErrorReporter micro_error_reporter;

// Simple feature extraction
void computeFeatures(float* input_signal, float output_spec[SPEC_HEIGHT][SPEC_WIDTH]) {
    // Divide signal into segments
    int segment_size = WINDOW_SIZE / SPEC_WIDTH;
    
    for(int i = 0; i < SPEC_HEIGHT; i++) {
        for(int j = 0; j < SPEC_WIDTH; j++) {
            float sum = 0;
            float max_val = -1000000;
            float min_val = 1000000;
            float rms = 0;
            
            // Calculate features for this segment
            int start_idx = j * segment_size;
            int end_idx = start_idx + segment_size;
            for(int k = start_idx; k < end_idx && k < WINDOW_SIZE; k++) {
                sum += abs(input_signal[k]);
                max_val = max(max_val, input_signal[k]);
                min_val = min(min_val, input_signal[k]);
                rms += input_signal[k] * input_signal[k];
            }
            rms = sqrt(rms / segment_size);
            
            // Different features for different rows
            switch(i % 8) {
                case 0:
                    output_spec[i][j] = sum / segment_size;  // Mean absolute value
                    break;
                case 1:
                    output_spec[i][j] = max_val;  // Maximum
                    break;
                case 2:
                    output_spec[i][j] = min_val;  // Minimum
                    break;
                case 3:
                    output_spec[i][j] = max_val - min_val;  // Range
                    break;
                case 4:
                    output_spec[i][j] = rms;  // Root mean square
                    break;
                case 5:
                    output_spec[i][j] = variance(input_signal + start_idx, segment_size);  // Variance
                    break;
                case 6:
                    output_spec[i][j] = zero_crossings(input_signal + start_idx, segment_size);  // Zero crossings
                    break;
                case 7:
                    output_spec[i][j] = median(input_signal + start_idx, segment_size);  // Median
                    break;
            }
        }
    }
}

// Helper functions for feature extraction
float variance(float* data, int length) {
    float mean = 0;
    float sq_sum = 0;
    
    // Calculate mean
    for(int i = 0; i < length; i++) {
        mean += data[i];
    }
    mean /= length;
    
    // Calculate variance
    for(int i = 0; i < length; i++) {
        float diff = data[i] - mean;
        sq_sum += diff * diff;
    }
    
    return sq_sum / length;
}

int zero_crossings(float* data, int length) {
    int count = 0;
    for(int i = 1; i < length; i++) {
        if((data[i] >= 0 && data[i-1] < 0) || (data[i] < 0 && data[i-1] >= 0)) {
            count++;
        }
    }
    return count;
}

float median(float* data, int length) {
    // Simple bubble sort for median calculation
    float temp_array[16];  // Using smaller window for median
    int temp_length = min(length, 16);
    
    // Copy data to temporary array
    for(int i = 0; i < temp_length; i++) {
        temp_array[i] = data[i];
    }
    
    // Sort
    for(int i = 0; i < temp_length-1; i++) {
        for(int j = 0; j < temp_length-i-1; j++) {
            if(temp_array[j] > temp_array[j+1]) {
                float temp = temp_array[j];
                temp_array[j] = temp_array[j+1];
                temp_array[j+1] = temp;
            }
        }
    }
    
    return temp_array[temp_length/2];
}

void sendGestureToUno(int gesture) {
    Wire.beginTransmission(8);  // Address 8 for Uno
    Wire.write(gesture);        // Send gesture index (0-5)
    Wire.endTransmission();
}

void setup() {
    Serial.begin(115200);
    Wire.begin();  // Initialize I2C as master
    delay(3000);
    Serial.println("\nStarting setup...");
    
    analogReadResolution(12);
    
    // Load model
    Serial.println("Loading model...");
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        return;
    }

    // Add operations
    Serial.println("Adding operations...");
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();

    // Create interpreter
    Serial.println("Creating interpreter...");
    static tflite::MicroInterpreter static_interpreter(
        model, 
        micro_op_resolver, 
        tensor_arena, 
        kTensorArenaSize,
        &micro_error_reporter,
        nullptr,
        nullptr
    );
    interpreter = &static_interpreter;

    // Allocate tensors
    Serial.println("Allocating tensors...");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return;
    }

    Serial.println("Setup complete!");
}

void loop() {
    if (buffer_index < WINDOW_SIZE) {
        emg_buffer[buffer_index] = analogRead(A0);
        
        if (buffer_index % 100 == 0) {
            Serial.print("Collecting data: ");
            Serial.print(buffer_index);
            Serial.println("/500");
        }
        
        buffer_index++;
        delay(10);
        return;
    }
    
    Serial.println("\n--- Making Prediction ---");
    
    computeFeatures(emg_buffer, features);
    
    // Get input tensor
    TfLiteTensor* input = interpreter->input(0);
    int8_t* input_buffer = input->data.int8;
    TfLiteQuantizationParams params = input->params;
    
    // Process and quantize input data
    for (int i = 0; i < SPEC_HEIGHT; i++) {
        for (int j = 0; j < SPEC_WIDTH; j++) {
            float normalized = (features[i][j] - SIGNAL_MEAN) / SIGNAL_STD;
            float scaled = normalized / params.scale + params.zero_point;
            input_buffer[i * SPEC_WIDTH + j] = static_cast<int8_t>(min(max(scaled, -128.0f), 127.0f));
        }
    }
    
    Serial.println("Running inference...");
    
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed");
        buffer_index = 0;
        return;
    }
    
    // Get output tensor
    TfLiteTensor* output = interpreter->output(0);
    int8_t* results = output->data.int8;
    TfLiteQuantizationParams out_params = output->params;
    
    float max_prob = -1;
    int prediction = -1;
    const char* gestures[] = {"Thumb", "Index", "Middle", "Ring", "Pinky", "Static"};
    
    // Process results
    for (int i = 0; i < 6; i++) {
        float prob = (results[i] - out_params.zero_point) * out_params.scale;
        if (prob > max_prob) {
            max_prob = prob;
            prediction = i;
        }
        
        Serial.print(gestures[i]);
        Serial.print(": ");
        Serial.print(prob * 100);
        Serial.println("%");
    }
    
    // Only send gesture if probability is above threshold
    if (max_prob > 0.5) {
        sendGestureToUno(prediction);
        Serial.print("\nSent gesture to Uno: ");
        Serial.println(gestures[prediction]);
    } else {
        // If no clear gesture detected, send static class (5)
        sendGestureToUno(5);
        Serial.println("\nNo clear gesture detected, sending Static");
    }
    
    buffer_index = 0;
    delay(100); // Small delay before next prediction
}