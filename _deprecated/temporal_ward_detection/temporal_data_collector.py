#!/usr/bin/env python3
"""
Temporal Ward Data Collection Script
Collects both visual and temporal data for enhanced ward detection training
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

class TemporalWardDataCollector:
    """Collect temporal data during League gameplay"""
    
    def __init__(self, output_dir: str = "temporal_ward_detection/dataset/temporal"):
        self.output_dir = Path(output_dir)
        self.current_sequence = []
        self.recording = False
        
    def start_sequence_recording(self, sequence_id: str):
        """Start recording a new ward placement sequence"""
        self.sequence_id = sequence_id
        self.current_sequence = []
        self.recording = True
        print(f"🎥 Started recording sequence: {sequence_id}")
    
    def add_frame_data(self, frame_number: int, player_pos: Tuple[float, float], 
                      inventory_state: Dict, timestamp: float):
        """Add frame data to current sequence"""
        if not self.recording:
            return
            
        frame_data = {
            "frame": frame_number,
            "player_position": {"x": player_pos[0], "y": player_pos[1]},
            "inventory_state": inventory_state,
            "timestamp": timestamp
        }
        
        self.current_sequence.append(frame_data)
    
    def mark_ward_placement(self, frame_number: int, ward_pos: Tuple[float, float], 
                           ward_type: str, ownership: str):
        """Mark when a ward is placed"""
        if not self.recording:
            return
            
        # Save sequence with ward placement info
        sequence_data = {
            "sequence_id": self.sequence_id,
            "frames": self.current_sequence,
            "ward_placement": {
                "frame": frame_number,
                "position": {"x": ward_pos[0], "y": ward_pos[1]},
                "type": ward_type,
                "ownership": ownership,
                "timestamp": time.time()
            }
        }
        
        # Extract temporal features
        from temporal_ward_trainer import TemporalWardFeatureExtractor
        extractor = TemporalWardFeatureExtractor()
        
        player_positions = [(f["player_position"]["x"], f["player_position"]["y"]) 
                           for f in self.current_sequence]
        timestamps = [f["timestamp"] for f in self.current_sequence]
        inventory_states = [f["inventory_state"] for f in self.current_sequence]
        
        features = {
            "movement": extractor.extract_movement_features(player_positions, timestamps),
            "inventory": extractor.extract_inventory_features(inventory_states, timestamps),
            "spatial": extractor.extract_spatial_features(player_positions, ward_pos),
            "timing": extractor.extract_timing_features(timestamps, sequence_data["ward_placement"]["timestamp"])
        }
        
        sequence_data["extracted_features"] = features
        
        # Save to file
        output_file = self.output_dir / "sequences" / f"{self.sequence_id}.json"
        with open(output_file, "w") as f:
            json.dump(sequence_data, f, indent=2)
        
        print(f"💾 Saved sequence: {output_file}")
        self.recording = False
    
    def create_training_interface(self):
        """Create simple interface for data collection during gameplay"""
        print("""
🎮 Temporal Ward Data Collection Interface

Controls:
- Press 'S' to start recording a ward placement sequence
- Move your champion around (data will be automatically tracked)
- Press 'W' when you place a ward
- Enter ward details (type, position)
- Sequence automatically saves

Usage:
1. Start League game
2. Run this script
3. Press 'S' when you're about to place a ward
4. Move normally and place the ward
5. Press 'W' and enter details
6. Repeat for each ward placement

This will build your temporal training dataset!
        """)

def main():
    collector = TemporalWardDataCollector()
    collector.create_training_interface()

if __name__ == "__main__":
    main()
