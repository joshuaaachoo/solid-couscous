# Temporal-Enhanced Ward Detection Training Guide

## 🧠 **Concept: ML Model with Temporal Features**

Instead of just visual detection, the ML model now learns from:
- **Visual Features**: What wards look like (traditional computer vision)
- **Temporal Features**: Movement patterns, inventory changes, timing (NEW!)

This creates a **hybrid model** that understands both:
1. "This looks like a ward" (visual)
2. "This was likely placed by the player" (temporal behavioral patterns)

## 🏗️ **Architecture Overview**

```
Input Frame + Temporal Context
           │
    ┌──────┴──────┐
    │             │
Visual CNN    Temporal LSTM
(YOLOv5)      (Movement/Inventory)
    │             │
    └──────┬──────┘
           │
    Fusion Network
           │
  ┌────────┼────────┐
  │        │        │
Ownership Ward    Confidence
Classifier Detection  Score
```

## 📊 **Temporal Features Used for Training**

### Movement Patterns:
- **Velocity**: How fast player moves toward ward spot
- **Acceleration**: Changes in movement speed
- **Smoothness**: Consistent vs erratic movement
- **Approach Pattern**: Direct approach vs wandering

### Inventory Tracking:
- **Trinket Usage**: When player uses ward charges
- **Control Ward Usage**: Pink ward placement patterns
- **Cooldown States**: Ward availability timing
- **Usage Rates**: How frequently player wards

### Spatial Relationships:
- **Distance to Ward**: How close player was when placed
- **Proximity Over Time**: Movement toward/away from ward
- **Placement Context**: Where player was relative to ward

### Timing Analysis:
- **Dwell Time**: How long player spent in area
- **Placement Timing**: When in sequence ward appeared
- **Temporal Correlation**: Timing relative to inventory changes

## 🎯 **Training Data Collection Process**

### 1. **Record Gameplay Sessions**
```python
# Use temporal_data_collector.py
collector = TemporalWardDataCollector()

# For each ward placement:
collector.start_sequence_recording("ward_001")
# ... player moves around ...
collector.add_frame_data(frame_num, player_pos, inventory_state, timestamp)
# ... ward gets placed ...
collector.mark_ward_placement(frame_num, ward_pos, "stealth_ward", "player")
```

### 2. **Automatic Feature Extraction**
```python
# Features automatically computed:
{
    "movement": {
        "avg_velocity": 45.2,      # Pixels per second
        "movement_smoothness": 0.85, # 0-1 smoothness score
        "approach_pattern": "direct"
    },
    "inventory": {
        "trinket_usage_rate": 0.5,  # Uses per second
        "trinket_usage_count": 1     # Total uses in sequence
    },
    "spatial": {
        "placement_distance": 0.0,   # Distance when placed (0 = exactly at player)
        "min_distance": 0.0          # Closest approach to final position
    },
    "timing": {
        "time_before_placement": 3.5, # Seconds of buildup
        "dwell_time_score": 0.35      # How long in area (0-1)
    }
}
```

### 3. **Enhanced Model Training**
```python
# Model learns correlations:
# High movement smoothness + trinket usage + close placement = "player" ward
# Erratic movement + no inventory change + distant = "enemy" ward
# Smooth approach + support item usage = "teammate" ward
```

## 🚀 **Training Steps**

1. **Collect Temporal Data**:
   ```bash
   python temporal_data_collector.py
   ```
   - Record 50+ ward placement sequences
   - Include player, teammate, and enemy ward examples
   - Vary contexts (laning, teamfights, roaming)

2. **Train Enhanced Model**:
   ```bash
   python train_temporal_model.py
   ```
   - Combines visual CNN with temporal LSTM
   - Learns behavioral patterns for ownership
   - 50-100 epochs training

3. **Deploy to SageMaker**:
   ```bash
   python deploy_enhanced_model.py
   ```
   - Updates endpoint with temporal-enhanced inference
   - Maintains temporal buffer for context
   - Real-time ownership classification

## 📈 **Expected Improvements**

### Traditional Visual-Only Model:
- **Accuracy**: ~70% (can see wards, can't determine ownership)
- **Ownership**: Random guessing (25% for 4-class problem)

### Temporal-Enhanced Model:
- **Accuracy**: ~85%+ (visual + behavioral patterns)
- **Ownership**: ~80%+ (learns player behavior patterns)
- **False Positives**: Reduced (temporal context filters noise)

## 💡 **Key Insights**

### Player Wards:
- **Smooth approach** to placement location
- **Inventory usage** exactly when ward appears
- **Short placement distance** (ward appears at player)
- **Deliberate timing** (approaches then places)

### Enemy Wards:
- **No movement correlation** (player elsewhere when placed)
- **No inventory changes** (enemy's inventory, not yours)
- **Ward appears without approach** (sudden appearance)
- **Irregular timing** (no buildup pattern)

### Teammate Wards:
- **Moderate correlation** (may be near teammate)
- **Support item patterns** (different from trinkets)
- **Team coordination timing** (during grouped movement)

## 🔄 **Integration with RiftRewind**

The enhanced model provides:
1. **Visual Detection**: "There's a ward at (x,y)"
2. **Ownership Classification**: "It's probably yours with 85% confidence"
3. **Behavioral Analysis**: "Based on your approach pattern and trinket usage"

This creates **intelligent ward analysis** that understands not just what wards look like, but how they get placed!
