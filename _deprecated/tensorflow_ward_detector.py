"""
TensorFlow Ward Detection Model for League of Legends
Deep Learning YOLO-based object detection for ward placement analysis
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any
import cv2
import json

class TensorFlowWardDetector:
    """
    TensorFlow-based deep learning model for ward detection in League of Legends gameplay
    Uses YOLO architecture for real-time object detection
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.class_names = ['Control Ward', 'Stealth Ward', 'Farsight Ward', 'Zombie Ward']
        self.input_size = (640, 640)
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.45
        
        if model_path:
            self.load_model(model_path)
    
    def create_model_architecture(self) -> tf.keras.Model:
        """
        Create YOLOv5-style architecture using TensorFlow/Keras
        This is a simplified version for demonstration
        """
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=(640, 640, 3), name='image_input')
        
        # Backbone (CSPDarknet-inspired)
        x = self._create_backbone(inputs)
        
        # Neck (PANet-inspired)
        x = self._create_neck(x)
        
        # Head (YOLO detection head)
        outputs = self._create_detection_head(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ward_detector_yolo')
        
        return model
    
    def _create_backbone(self, inputs):
        """Create feature extraction backbone"""
        
        # Initial conv block
        x = tf.keras.layers.Conv2D(32, 6, strides=2, padding='same', activation='swish')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Downsampling blocks
        x = self._conv_block(x, 64, 3, 2)   # 320x320
        x = self._csp_block(x, 64, 1)
        
        x = self._conv_block(x, 128, 3, 2)  # 160x160  
        x = self._csp_block(x, 128, 2)
        
        x = self._conv_block(x, 256, 3, 2)  # 80x80
        x = self._csp_block(x, 256, 3)
        
        x = self._conv_block(x, 512, 3, 2)  # 40x40
        x = self._csp_block(x, 512, 1)
        
        x = self._conv_block(x, 1024, 3, 2) # 20x20
        x = self._csp_block(x, 1024, 1)
        
        return x
    
    def _conv_block(self, x, filters, kernel_size, strides=1):
        """Convolution block with BatchNorm and Swish activation"""
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, 
            padding='same', use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        return x
    
    def _csp_block(self, x, filters, num_blocks):
        """Cross Stage Partial block"""
        # Simplified CSP implementation
        shortcut = tf.keras.layers.Conv2D(filters // 2, 1, padding='same')(x)
        
        x = tf.keras.layers.Conv2D(filters // 2, 1, padding='same')(x)
        
        for _ in range(num_blocks):
            residual = x
            x = self._conv_block(x, filters // 2, 1)
            x = self._conv_block(x, filters // 2, 3)
            x = tf.keras.layers.Add()([x, residual])
        
        x = tf.keras.layers.Concatenate()([x, shortcut])
        x = self._conv_block(x, filters, 1)
        
        return x
    
    def _create_neck(self, x):
        """Feature pyramid neck (simplified PANet)"""
        # This would include upsampling and feature fusion
        # For demo, just pass through
        return x
    
    def _create_detection_head(self, x):
        """YOLO detection head"""
        num_classes = len(self.class_names)
        num_anchors = 3
        
        # Output: [batch, grid_h, grid_w, anchors * (5 + num_classes)]
        # 5 = x, y, w, h, confidence
        output_channels = num_anchors * (5 + num_classes)
        
        outputs = tf.keras.layers.Conv2D(
            output_channels, 1, padding='same', name='detection_output'
        )(x)
        
        return outputs
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess video frame for ward detection"""
        
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_input = np.expand_dims(normalized, axis=0)
        
        return batch_input
    
    def postprocess_predictions(self, predictions: np.ndarray, 
                              original_shape: Tuple[int, int]) -> List[Dict]:
        """Convert model predictions to detected wards"""
        
        # This would include:
        # 1. Decode YOLO output format
        # 2. Apply confidence threshold
        # 3. Apply Non-Maximum Suppression (NMS)
        # 4. Scale boxes to original image size
        
        # For demo, return simulated detections
        detections = [
            {
                'type': 'Control Ward',
                'confidence': 0.92,
                'bbox': {'x1': 150, 'y1': 200, 'x2': 170, 'y2': 220},
                'position': {'x': 160, 'y': 210}
            },
            {
                'type': 'Stealth Ward', 
                'confidence': 0.87,
                'bbox': {'x1': 300, 'y1': 450, 'x2': 315, 'y2': 465},
                'position': {'x': 307, 'y': 457}
            }
        ]
        
        return detections
    
    def detect_wards_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect wards in a single video frame"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        model_input = self.preprocess_frame(frame)
        
        # Inference
        predictions = self.model(model_input, training=False)
        
        # Postprocess
        detections = self.postprocess_predictions(predictions, frame.shape[:2])
        
        return {
            'detections': detections,
            'total_wards': len(detections),
            'frame_shape': frame.shape,
            'inference_time_ms': 45.2  # Would measure actual time
        }
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile model with optimizer and loss function"""
        
        if self.model is None:
            self.model = self.create_model_architecture()
        
        # YOLO loss function (simplified)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=self._yolo_loss,
            metrics=['accuracy']
        )
        
        return self.model
    
    def _yolo_loss(self, y_true, y_pred):
        """YOLO loss function (simplified implementation)"""
        
        # This would include:
        # - Box regression loss (IoU or GIoU)
        # - Objectness loss (binary cross-entropy)
        # - Classification loss (categorical cross-entropy)
        
        # Placeholder implementation
        box_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        return box_loss
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(model_path)
    
    def save_model(self, save_path: str):
        """Save trained model"""
        if self.model is not None:
            self.model.save(save_path)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            self.model = self.create_model_architecture()
        
        return self.model.summary()

# Training utilities
class WardDetectionTrainer:
    """Utilities for training the ward detection model"""
    
    def __init__(self, model: TensorFlowWardDetector):
        self.model = model
    
    def prepare_training_data(self, dataset_path: str):
        """Prepare training data from annotated League gameplay frames"""
        
        # This would:
        # 1. Load annotated frames with ward bounding boxes
        # 2. Convert annotations to YOLO format
        # 3. Apply data augmentation
        # 4. Create tf.data.Dataset
        
        pass
    
    def train_model(self, train_dataset, val_dataset, epochs: int = 100):
        """Train the ward detection model"""
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'ward_detector_best.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                monitor='val_loss'
            )
        ]
        
        history = self.model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history

# Example usage for SageMaker deployment
def create_sagemaker_inference_handler():
    """
    Create inference handler for SageMaker deployment
    This would be in a separate inference.py file
    """
    
    model = TensorFlowWardDetector()
    
    def model_fn(model_dir):
        """Load model for SageMaker inference"""
        model.load_model(f"{model_dir}/ward_detector.h5")
        return model
    
    def input_fn(request_body, request_content_type):
        """Parse input data"""
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            # Convert base64 image to numpy array
            return input_data
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    
    def predict_fn(input_data, model):
        """Run inference"""
        # Convert input to frame
        frame = np.array(input_data['frame'])
        
        # Detect wards
        results = model.detect_wards_in_frame(frame)
        
        return results
    
    def output_fn(prediction, accept):
        """Format output"""
        if accept == 'application/json':
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

if __name__ == "__main__":
    # Demo usage
    detector = TensorFlowWardDetector()
    model = detector.create_model_architecture()
    
    print("🤖 TensorFlow Ward Detection Model")
    print(f"📊 Model Parameters: {model.count_params():,}")
    print(f"🎯 Classes: {detector.class_names}")
    print(f"📐 Input Size: {detector.input_size}")
    print("✅ Model architecture created successfully!")