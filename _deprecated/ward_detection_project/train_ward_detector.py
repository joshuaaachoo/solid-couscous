#!/usr/bin/env python3
"""
Ward Detection Training Script
"""

import subprocess
import sys
from pathlib import Path

def train_ward_detector():
    yolo_dir = Path("ward_detection_project/yolov5")
    dataset_yaml = Path("ward_detection_project/dataset/ward_dataset.yaml")
    
    # Install requirements first
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "ultralytics", "opencv-python"
    ])
    
    cmd = [
        sys.executable, str(yolo_dir / "train.py"),
        "--img", "640",
        "--batch", "16", 
        "--epochs", "100",
        "--data", str(dataset_yaml),
        "--weights", "yolov5s.pt",
        "--project", str(Path("ward_detection_project/models")),
        "--name", "ward_detector"
    ]
    
    print("🚀 Starting YOLOv5 training...")
    subprocess.run(cmd, check=True)
    print("✅ Training completed!")

if __name__ == "__main__":
    train_ward_detector()
