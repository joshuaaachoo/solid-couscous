#!/usr/bin/env python3
"""
Enhanced Ward Detection Training with Temporal Features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from pathlib import Path
from temporal_ward_trainer import TemporalWardNet

class TemporalWardDataset(Dataset):
    """Dataset combining visual and temporal features"""
    
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequences = self.load_sequences()
    
    def load_sequences(self):
        """Load temporal sequence files"""
        sequence_dir = self.data_dir / "temporal" / "sequences"
        sequences = []
        
        for sequence_file in sequence_dir.glob("*.json"):
            with open(sequence_file) as f:
                sequence_data = json.load(f)
                sequences.append(sequence_data)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Extract temporal features (movement, inventory, spatial, timing)
        features = sequence.get("extracted_features", {})
        
        temporal_vector = [
            features.get("movement", {}).get("avg_velocity", 0),
            features.get("movement", {}).get("velocity_variance", 0),
            features.get("movement", {}).get("movement_smoothness", 0),
            features.get("inventory", {}).get("trinket_usage_rate", 0),
            features.get("inventory", {}).get("control_ward_usage_rate", 0),
            features.get("spatial", {}).get("min_distance", 0),
            features.get("spatial", {}).get("placement_distance", 0),
            features.get("timing", {}).get("time_before_placement", 0),
            features.get("timing", {}).get("dwell_time_score", 0)
        ]
        
        # Pad/truncate to fixed size
        temporal_vector = temporal_vector[:20] + [0] * max(0, 20 - len(temporal_vector))
        
        # Create visual features (mock for now - replace with real CNN features)
        visual_features = torch.randn(2048)  # Replace with real visual encoding
        
        # Create labels
        ownership_map = {"player": 0, "teammate": 1, "enemy": 2, "unknown": 3}
        ownership_label = ownership_map.get(
            sequence.get("ward_placement", {}).get("ownership", "unknown"), 3
        )
        
        return {
            "visual_features": visual_features,
            "temporal_features": torch.tensor(temporal_vector, dtype=torch.float32),
            "ownership_label": torch.tensor(ownership_label, dtype=torch.long),
            "ward_position": torch.tensor([
                sequence.get("ward_placement", {}).get("position", {}).get("x", 0),
                sequence.get("ward_placement", {}).get("position", {}).get("y", 0)
            ], dtype=torch.float32)
        }

def train_temporal_ward_model():
    """Train enhanced ward detection model with temporal features"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalWardNet().to(device)
    
    # Dataset and loader
    dataset = TemporalWardDataset("temporal_ward_detection/dataset")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ownership_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    print("🚀 Training temporal ward detection model...")
    print(f"Device: {device}")
    print(f"Sequences: {len(dataset)}")
    
    # Training loop
    model.train()
    for epoch in range(50):
        total_loss = 0
        
        for batch in dataloader:
            visual = batch["visual_features"].to(device)
            temporal = batch["temporal_features"].unsqueeze(1).to(device)  # Add sequence dim
            ownership_labels = batch["ownership_label"].to(device)
            
            # Forward pass
            outputs = model(visual, temporal)
            
            # Calculate losses
            ownership_loss = ownership_criterion(outputs["ownership"], ownership_labels)
            confidence_loss = regression_criterion(
                outputs["confidence"].squeeze(), 
                torch.ones_like(ownership_labels, dtype=torch.float32)
            )
            
            total_loss = ownership_loss + 0.1 * confidence_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss += total_loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")
    
    # Save model
    model_path = Path("temporal_ward_detection/models") / "temporal_ward_detector.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved: {model_path}")

if __name__ == "__main__":
    train_temporal_ward_model()
