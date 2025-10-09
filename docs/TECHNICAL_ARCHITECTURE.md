# RiftRewind Technical Architecture

## 🏗️ **System Overview**

RiftRewind is a multi-tier AI platform that processes League of Legends video replays to provide intelligent performance analysis and coaching insights.

```
┌─────────────────────────────────────────────────────────────────┐
│                          RiftRewind Platform                    │
├─────────────────────────────────────────────────────────────────┤
│  Frontend Layer                                                 │
│  ├─ Demo Interface (Python/Streamlit)                          │
│  ├─ Visualization Dashboard                                     │
│  └─ Configuration Management                                    │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                              │
│  ├─ VOD Analysis Engine                                         │
│  ├─ Performance Tracker                                         │
│  ├─ Coaching Insights Generator                                 │
│  └─ Quality Configuration                                       │
├─────────────────────────────────────────────────────────────────┤
│  ML Processing Layer                                            │
│  ├─ Visual Detection (YOLOv5/TensorFlow)                      │
│  ├─ Temporal Analysis (LSTM Networks)                          │
│  ├─ Behavioral Pattern Recognition                             │
│  └─ Fusion Network (Multi-modal)                               │
├─────────────────────────────────────────────────────────────────┤
│  Cloud Infrastructure (AWS)                                    │
│  ├─ SageMaker Endpoints                                        │
│  ├─ S3 Storage                                                 │
│  ├─ CloudWatch Monitoring                                      │
│  └─ IAM Security                                               │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                     │
│  ├─ Video Processing (OpenCV)                                  │
│  ├─ Frame Analysis Cache                                       │
│  ├─ Model Storage                                              │
│  └─ Results Database                                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 **Machine Learning Architecture**

### Multi-Modal Neural Network Design

```
Input: Game Frame + Temporal Context
            │
    ┌───────┴───────┐
    │               │
Visual CNN      Temporal LSTM
(YOLOv5)        (Behavioral)
    │               │
  Visual         Temporal
 Features        Features
    │               │
    └───────┬───────┘
            │
      Fusion Network
            │
    ┌───────┼───────┐
    │       │       │
  Ward    Ownership Confidence
Detection  Class    Score
    │       │       │
    └───────┼───────┘
            │
        Final Output
```

### Component Details

#### 1. Visual Processing Pipeline
```python
# YOLOv5-based Ward Detection
Input: 1920x1080 RGB Frame
├─ Preprocessing: Resize, Normalize
├─ Backbone: CSPDarknet53
├─ Neck: PANet
├─ Head: Detection + Classification
└─ Output: Bounding boxes + Classes
```

#### 2. Temporal Analysis Pipeline
```python
# LSTM-based Behavioral Analysis
Input: Sequence of Player States
├─ Movement Features: Velocity, acceleration
├─ Inventory Features: Usage patterns, timing
├─ Spatial Features: Distance correlation
├─ Timing Features: Placement sequences
└─ Output: Ownership probabilities
```

#### 3. Fusion Network
```python
# Multi-modal Feature Fusion
Visual Features (256d) + Temporal Features (64d)
├─ Concatenation Layer
├─ Dense Layer (128d) + ReLU + Dropout
├─ Dense Layer (64d) + ReLU
└─ Output Heads:
   ├─ Ward Detection (bbox + class)
   ├─ Ownership Classification (4 classes)
   └─ Confidence Regression (0-1)
```

## ☁️ **AWS Cloud Architecture**

### Infrastructure Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Cloud Platform                       │
├─────────────────────────────────────────────────────────────────┤
│  Compute Layer                                                  │
│  ├─ SageMaker Inference Endpoints                              │
│  │  ├─ ml.m5.large instances                                   │
│  │  ├─ Auto-scaling (1-10 instances)                           │
│  │  └─ PyTorch containers                                      │
│  └─ EC2 Development Environment                                 │
├─────────────────────────────────────────────────────────────────┤
│  Storage Layer                                                  │
│  ├─ S3 Buckets                                                 │
│  │  ├─ Model Artifacts                                         │
│  │  ├─ Training Data                                           │
│  │  └─ Analysis Results                                        │
│  └─ EFS (shared file system)                                   │
├─────────────────────────────────────────────────────────────────┤
│  Security Layer                                                │
│  ├─ IAM Roles & Policies                                       │
│  ├─ VPC & Security Groups                                      │
│  └─ Encryption (S3, SageMaker)                                 │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring Layer                                              │
│  ├─ CloudWatch Metrics                                         │
│  ├─ CloudWatch Logs                                            │
│  ├─ CloudWatch Alarms                                          │
│  └─ Cost Monitoring                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Pipeline

```
Development → Testing → Staging → Production

1. Model Training:
   ├─ Local development
   ├─ SageMaker training jobs
   ├─ Model validation
   └─ Artifact storage (S3)

2. Endpoint Deployment:
   ├─ Model registration
   ├─ Container build
   ├─ Endpoint configuration
   └─ Auto-scaling setup

3. Monitoring & Maintenance:
   ├─ Performance metrics
   ├─ Cost optimization
   ├─ Error handling
   └─ Automated rollback
```

## 📊 **Data Flow Architecture**

### Processing Pipeline

```
Video Input → Frame Extraction → ML Processing → Analysis → Results

1. Video Processing:
   VOD File (.mp4/.avi)
   ├─ Frame extraction (3fps)
   ├─ Quality validation
   ├─ Metadata extraction
   └─ Preprocessing

2. ML Processing:
   Frame Batch
   ├─ Visual analysis (SageMaker)
   ├─ Temporal context gathering
   ├─ Feature fusion
   └─ Prediction generation

3. Results Processing:
   Raw Predictions
   ├─ Post-processing
   ├─ Confidence filtering
   ├─ Aggregation
   └─ Insight generation

4. Output Generation:
   Analysis Results
   ├─ Visualization creation
   ├─ Report generation
   ├─ Coaching recommendations
   └─ Storage/caching
```

### Real-time Processing Flow

```
Input Frame → Preprocessing → Inference → Post-processing → Output
     │             │            │            │            │
    3ms          2ms         25ms         2ms          1ms
                            
Total Latency: ~33ms (well under 50ms target)
```

## 🔧 **Performance Optimization**

### Model Optimization
- **Quantization**: INT8 inference for 2x speed improvement
- **Pruning**: Remove redundant network parameters
- **Batch Processing**: Optimize GPU utilization
- **Caching**: Store frequent computations

### Infrastructure Optimization  
- **Auto-scaling**: Dynamic resource allocation
- **Load Balancing**: Distribute traffic efficiently
- **CDN**: Cache static assets
- **Compression**: Reduce data transfer

### Cost Optimization
- **Spot Instances**: Use for training workloads
- **Reserved Instances**: Long-term capacity planning
- **S3 Lifecycle**: Automatic data archiving
- **Monitoring**: Continuous cost tracking

## 🔒 **Security Architecture**

### Access Control
```
IAM Structure:
├─ RiftRewind-Admin (Full access)
├─ RiftRewind-Developer (Development resources)
├─ RiftRewind-ML (Model training/inference)
└─ RiftRewind-ReadOnly (Monitoring access)
```

### Data Protection
- **Encryption at Rest**: S3, EBS volumes
- **Encryption in Transit**: HTTPS, TLS
- **Network Security**: VPC, Security Groups
- **Access Logging**: CloudTrail, access logs

### Compliance
- **Data Privacy**: No personal data storage
- **Audit Trail**: Complete action logging
- **Backup Strategy**: Multi-region replication
- **Disaster Recovery**: Automated failover

## 📈 **Scalability Design**

### Horizontal Scaling
- **SageMaker Auto-scaling**: 1-10 instances based on load
- **Microservices**: Independent scaling per component
- **Load Distribution**: Geographic load balancing
- **Caching Strategy**: Multi-level caching

### Performance Targets
- **Latency**: <50ms inference time
- **Throughput**: 1000+ concurrent requests
- **Availability**: 99.5% uptime SLA
- **Accuracy**: 85%+ detection, 80%+ ownership

### Growth Planning
- **Multi-region Deployment**: Global user base
- **Feature Expansion**: Additional game support
- **API Scaling**: External integrations
- **Mobile Support**: Cross-platform access

This architecture supports the current system while providing a foundation for future expansion and optimization.