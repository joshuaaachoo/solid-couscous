# Ward Detection Dataset Collection Guide

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

### 🎯 **CRITICAL: Ward Ownership Training**
For each ward type, capture examples showing:
- **Your Wards**: Bright, clear visibility with subtle outlines
- **Teammate Wards**: Moderate visibility, same team color
- **Enemy Wards**: Dimmed, less particle effects, no outlines
- **Blue vs Red Team**: Different color tints based on team side

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
4. **Enhanced Classes** for ownership detection:
   - **0**: my_stealth_ward (your green wards)
   - **1**: teammate_stealth_ward (ally green wards)  
   - **2**: enemy_stealth_ward (enemy green wards)
   - **3**: my_control_ward (your pink wards)
   - **4**: teammate_control_ward (ally pink wards)
   - **5**: enemy_control_ward (enemy pink wards)
   - **6**: zombie_ward (support item wards)
   - **7**: ward_debris (destroyed wards)

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
