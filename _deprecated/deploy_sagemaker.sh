#!/bin/bash
# RiftRewind SageMaker Deployment Runner
# Run this after adding S3 permissions to deploy your YOLOv5-TensorFlow ward detector

echo "🎯 RiftRewind SageMaker Deployment"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "setup_sagemaker.py" ]; then
    echo "❌ Error: Please run this from the RiftRewindClean directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "🐍 Activating Python virtual environment..."
    source .venv/bin/activate
fi

# Install required packages
echo "📦 Installing required packages..."
pip install boto3 numpy tensorflow

# Run the deployment
echo "🚀 Starting SageMaker deployment..."
echo "⏳ This will take 10-15 minutes to complete..."
echo ""

python3 setup_sagemaker.py

echo ""
echo "✅ Deployment script completed!"
echo "Check the output above for results."