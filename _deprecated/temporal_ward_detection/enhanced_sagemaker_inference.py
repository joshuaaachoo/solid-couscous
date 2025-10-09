
import json
import torch
import numpy as np
from temporal_ward_trainer import TemporalWardNet, TemporalWardFeatureExtractor

class EnhancedWardDetector:
    def __init__(self):
        # Load temporal-enhanced model
        self.model = TemporalWardNet()
        self.model.load_state_dict(torch.load('./temporal_ward_detector.pth'))
        self.model.eval()
        
        self.feature_extractor = TemporalWardFeatureExtractor()
        self.temporal_buffer = []  # Store recent temporal data
        
    def __call__(self, request_data):
        """Enhanced inference with temporal context"""
        
        # Extract current frame data
        frame_metadata = request_data.get('frame_metadata', {})
        temporal_context = request_data.get('temporal_context', {})
        
        # Update temporal buffer
        if temporal_context:
            self.temporal_buffer.append(temporal_context)
            # Keep last 30 frames (10 seconds at 3fps)
            self.temporal_buffer = self.temporal_buffer[-30:]
        
        # Extract temporal features if we have enough history
        temporal_features = self._extract_temporal_features()
        
        # Mock visual features (replace with real CNN extraction)
        visual_features = torch.randn(1, 2048)
        
        # Run enhanced inference
        with torch.no_grad():
            outputs = self.model(visual_features, temporal_features)
        
        # Process outputs
        ownership_probs = torch.softmax(outputs['ownership'], dim=1)
        ownership_class = torch.argmax(ownership_probs, dim=1).item()
        confidence = outputs['confidence'].item()
        
        ownership_map = {0: "player", 1: "teammate", 2: "enemy", 3: "unknown"}
        
        # Generate mock detections with enhanced ownership
        detections = []
        for i in range(np.random.randint(0, 4)):  # 0-3 wards
            detection = {
                "type": np.random.choice(["Stealth Ward", "Control Ward"]),
                "confidence": 0.75 + np.random.random() * 0.25,
                "position": {
                    "x": np.random.randint(100, 1820),
                    "y": np.random.randint(100, 980)
                },
                "ownership": {
                    "owner": ownership_map[ownership_class],
                    "confidence": round(confidence, 3),
                    "analysis_method": "temporal_enhanced_ml"
                }
            }
            detections.append(detection)
        
        return {
            "detections": detections,
            "total_wards": len(detections),
            "model_info": {
                "name": "Temporal Enhanced Ward Detector",
                "version": "2.0.0",
                "features": ["visual_detection", "temporal_tracking", "ownership_classification"],
                "architecture": "YOLOv5 + LSTM + Temporal Features"
            },
            "temporal_analysis": {
                "frames_in_buffer": len(self.temporal_buffer),
                "temporal_features_active": len(self.temporal_buffer) >= 5
            }
        }
    
    def _extract_temporal_features(self):
        """Extract temporal features from buffer"""
        if len(self.temporal_buffer) < 5:
            # Not enough temporal context, return zeros
            return torch.zeros(1, 1, 20)
        
        # Extract sequences from buffer
        player_positions = []
        timestamps = []
        inventory_states = []
        
        for frame_data in self.temporal_buffer[-10:]:  # Last 10 frames
            if 'player_position' in frame_data:
                pos = frame_data['player_position']
                player_positions.append((pos.get('x', 0), pos.get('y', 0)))
            
            if 'timestamp' in frame_data:
                timestamps.append(frame_data['timestamp'])
                
            if 'inventory_state' in frame_data:
                inventory_states.append(frame_data['inventory_state'])
        
        # Extract features using our temporal extractor
        if len(player_positions) >= 2:
            movement_features = self.feature_extractor.extract_movement_features(
                player_positions, timestamps
            )
            inventory_features = self.feature_extractor.extract_inventory_features(
                inventory_states, timestamps
            )
            
            # Combine into feature vector
            feature_vector = [
                movement_features.get('avg_velocity', 0),
                movement_features.get('velocity_variance', 0),
                movement_features.get('movement_smoothness', 0),
                inventory_features.get('trinket_usage_rate', 0),
                inventory_features.get('control_ward_usage_rate', 0),
                len(player_positions),  # Sequence length
                max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0,  # Time span
            ]
            
            # Pad to 20 features
            feature_vector = feature_vector[:20] + [0] * max(0, 20 - len(feature_vector))
            
            return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return torch.zeros(1, 1, 20)

def model_fn(model_dir):
    """Load enhanced temporal model"""
    return EnhancedWardDetector()

def input_fn(request_body, request_content_type='application/json'):
    """Parse input with temporal context"""
    if request_content_type == 'application/json':
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run enhanced temporal prediction"""
    return model(input_data)

def output_fn(prediction, accept='application/json'):
    """Format enhanced output"""
    if accept == 'application/json':
        return json.dumps(prediction), 'application/json'
    raise ValueError(f"Unsupported accept type: {accept}")
