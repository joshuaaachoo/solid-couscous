#!/usr/bin/env python3
"""
Real Ward Detection Setup
Creates everything needed to train actual computer vision models
"""

import os
import yaml
from pathlib import Path
import subprocess

class WardDetectionTrainer:
    """Set up real ward detection training"""
    
    def __init__(self, project_dir: str = "ward_detection_project"):
        self.project_dir = Path(project_dir)
        self.dataset_dir = self.project_dir / "dataset"
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels" 
        self.models_dir = self.project_dir / "models"
        
        # Ward classes for detection
        self.ward_classes = {
            0: "stealth_ward",     # Regular wards (green)
            1: "control_ward",     # Pink wards 
            2: "zombie_ward",      # Zombie wards
            3: "ward_debris"       # Destroyed ward debris
        }
        
        self.setup_directories()
    
    def setup_directories(self):
        """Create project directory structure"""
        for dir_path in [self.images_dir, self.labels_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create train/val splits
        for split in ["train", "val"]:
            (self.images_dir / split).mkdir(exist_ok=True)
            (self.labels_dir / split).mkdir(exist_ok=True)
    
    def create_dataset_yaml(self):
        """Create YOLOv5 dataset configuration"""
        
        dataset_config = {
            "train": str(self.images_dir / "train"),
            "val": str(self.images_dir / "val"),
            "nc": len(self.ward_classes),
            "names": list(self.ward_classes.values())
        }
        
        yaml_path = self.dataset_dir / "ward_dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"📋 Dataset config created: {yaml_path}")
        return yaml_path
    
    def install_yolov5(self):
        """Install YOLOv5 for training"""
        
        yolo_dir = self.project_dir / "yolov5"
        
        if not yolo_dir.exists():
            print("📦 Installing YOLOv5...")
            subprocess.run([
                "git", "clone", "https://github.com/ultralytics/yolov5", 
                str(yolo_dir)
            ], check=True)
            print("✅ YOLOv5 installed successfully!")
        else:
            print("✅ YOLOv5 already installed")
            
        return yolo_dir
    
    def create_data_collection_guide(self):
        """Create guide for collecting training data"""
        
        guide = """# Ward Detection Dataset Collection Guide

## 📸 **Step 1: Collect Screenshots**

### What You Need:
- League of Legends client
- Screenshot tool (built-in or OBS)  
- Various game situations with wards

### Screenshots to Capture:
1. **Stealth Wards (Green):** River bushes, lane bushes, jungle entrances
2. **Control Wards (Pink):** Common pink ward spots, both revealed and hidden  
3. **Zombie Wards:** From support items, different stages
4. **Ward Debris:** Recently destroyed wards, particle effects

### Tips:
- Take 200+ screenshots per ward type (800+ total minimum)
- Include false positives (things that look like wards but aren't)
- Various game times (day/night cycle affects visibility)
- Different champions/skins near wards
- Multiple camera angles and zoom levels

## 🏷️ **Step 2: Annotate Images**

### Option A: LabelImg (Free)
```bash
pip install labelImg
labelImg
```
1. Open your screenshots folder
2. Draw bounding boxes around each ward
3. Save in YOLO format
4. Classes: stealth_ward=0, control_ward=1, zombie_ward=2, ward_debris=3

### Option B: Roboflow (Web-based)
1. Upload images to roboflow.com
2. Draw bounding boxes
3. Export in YOLOv5 format
4. Download and place in dataset folder

## 📁 **Step 3: Organize Dataset**

```
ward_detection_project/
└── dataset/
    ├── images/
    │   ├── train/          # 80% of images
    │   └── val/            # 20% of images
    └── labels/
        ├── train/          # Corresponding .txt files
        └── val/            # Corresponding .txt files
```

## 🚀 **Step 4: Train Model**

```bash
python train_ward_detector.py
```

Training will take 2-6 hours depending on dataset size and GPU power.

## ✅ **Expected Results:**

With good data (~1000+ images):
- **mAP@0.5**: 0.7-0.9 (70-90% accuracy)
- **Inference Speed**: 10-30ms per image
- **False Positives**: <5% with proper training

## 🎯 **Integration:**

Once trained, replace the mock model in your SageMaker endpoint with real weights!
"""
        
        guide_path = self.project_dir / "DATA_COLLECTION_GUIDE.md"
        with open(guide_path, "w") as f:
            f.write(guide)
        
        print(f"📖 Data collection guide created: {guide_path}")
        return guide_path
    
    def create_training_script(self):
        """Create script to train the ward detection model"""
        
        training_script = f'''#!/usr/bin/env python3
"""
Ward Detection Training Script
"""

import subprocess
import sys
from pathlib import Path

def train_ward_detector():
    yolo_dir = Path("{self.project_dir}/yolov5")
    dataset_yaml = Path("{self.dataset_dir}/ward_dataset.yaml")
    
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
        "--project", str(Path("{self.models_dir}")),
        "--name", "ward_detector"
    ]
    
    print("🚀 Starting YOLOv5 training...")
    subprocess.run(cmd, check=True)
    print("✅ Training completed!")

if __name__ == "__main__":
    train_ward_detector()
'''
        
        script_path = self.project_dir / "train_ward_detector.py"
        with open(script_path, "w") as f:
            f.write(training_script)
        
        os.chmod(script_path, 0o755)
        print(f"🎯 Training script created: {script_path}")
        return script_path
    
    def setup_complete_project(self):
        """Set up complete ward detection training project"""
        
        print("🎯 Setting up Real Ward Detection Training Project...")
        print("=" * 60)
        
        # Create all necessary files
        yaml_path = self.create_dataset_yaml()
        yolo_dir = self.install_yolov5()
        training_script = self.create_training_script()
        guide_path = self.create_data_collection_guide()
        
        print("\\n🎉 Project setup complete!")
        print(f"📁 Project directory: {self.project_dir}")
        print("\\n📋 Next Steps:")
        print("1. Read the data collection guide:")
        print(f"   open {guide_path}")
        print("\\n2. Collect and annotate League screenshots")
        print(f"   - Place images in: {self.images_dir}/train/ and {self.images_dir}/val/")
        print(f"   - Place labels in: {self.labels_dir}/train/ and {self.labels_dir}/val/")
        print("\\n3. Train the model:")
        print(f"   python {training_script}")
        print("\\n🚀 Then you'll have REAL ward detection!")
        
        return {
            'project_dir': self.project_dir,
            'guide': guide_path,
            'training_script': training_script
        }

def main():
    trainer = WardDetectionTrainer()
    result = trainer.setup_complete_project()
    return result

if __name__ == "__main__":
    main()