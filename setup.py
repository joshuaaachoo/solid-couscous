#!/usr/bin/env python3
"""
RiftRewind Setup Script
Initializes the project environment and AWS infrastructure
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def setup_aws():
    """Setup AWS infrastructure"""
    print("☁️ Setting up AWS infrastructure...")
    subprocess.run([sys.executable, "src/aws/ultra_simple_deploy.py"], check=False)

def create_directories():
    """Create necessary project directories"""
    dirs = ["logs", "models", "cache", "outputs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("📁 Project directories created")

def main():
    print("🚀 RiftRewind Setup")
    print("=" * 40)
    
    try:
        create_directories()
        install_dependencies()
        setup_aws()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Configure AWS credentials: aws configure")
        print("2. Run demo: python src/core/demo_riftrewind.py")
        print("3. Check documentation in docs/")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()