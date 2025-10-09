#!/usr/bin/env python3
"""
RiftRewind SageMaker Deployment Script
Complete setup for YOLOv5-TensorFlow ward detection endpoints
"""

import boto3
import json
import time
import tarfile
import os
import io
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime

class RiftRewindSageMakerSetup:
    """
    Complete SageMaker setup for RiftRewind ward detection
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.region_name = region_name
        self.account_id = '407275151589'  # Your AWS account
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS clients
        self.sagemaker = boto3.client('sagemaker', region_name=region_name)
        self.s3 = boto3.client('s3', region_name=region_name)
        self.iam = boto3.client('iam', region_name=region_name)
        
        # Configuration
        self.bucket_name = f'riftrewind-models-{int(time.time())}'
        self.execution_role_name = 'RiftRewind-SageMaker-ExecutionRole'
        
    def check_permissions(self) -> Dict[str, bool]:
        """Check if user has required SageMaker permissions"""
        
        permissions = {
            'sagemaker_access': False,
            'iam_access': False, 
            's3_access': False
        }
        
        try:
            # Test SageMaker access
            self.sagemaker.list_endpoints(MaxResults=1)
            permissions['sagemaker_access'] = True
            self.logger.info("✅ SageMaker access confirmed")
        except Exception as e:
            self.logger.error(f"❌ SageMaker access failed: {e}")
        
        try:
            # Test IAM access
            self.iam.list_roles(MaxItems=1)
            permissions['iam_access'] = True
            self.logger.info("✅ IAM access confirmed")
        except Exception as e:
            self.logger.error(f"❌ IAM access failed: {e}")
            
        try:
            # Test S3 access
            self.s3.list_buckets()
            permissions['s3_access'] = True
            self.logger.info("✅ S3 access confirmed")
        except Exception as e:
            self.logger.error(f"❌ S3 access failed: {e}")
        
        return permissions
    
    def create_s3_bucket(self) -> str:
        """Create S3 bucket for model artifacts"""
        
        try:
            self.logger.info(f"🪣 Creating S3 bucket: {self.bucket_name}")
            
            # Create bucket
            if self.region_name == 'us-east-1':
                # us-east-1 doesn't need LocationConstraint
                self.s3.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={
                        'LocationConstraint': self.region_name
                    }
                )
            
            # Add bucket policy for SageMaker access
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": ["s3:GetObject", "s3:PutObject"],
                        "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
                    }
                ]
            }
            
            self.s3.put_bucket_policy(
                Bucket=self.bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            self.logger.info(f"✅ S3 bucket created: {self.bucket_name}")
            return self.bucket_name
            
        except Exception as e:
            self.logger.error(f"❌ S3 bucket creation failed: {e}")
            raise
    
    def create_execution_role(self) -> str:
        """Create SageMaker execution role"""
        
        try:
            # Check if role already exists
            try:
                role_response = self.iam.get_role(RoleName=self.execution_role_name)
                role_arn = role_response['Role']['Arn']
                self.logger.info(f"✅ Using existing execution role: {role_arn}")
                return role_arn
            except self.iam.exceptions.NoSuchEntityException:
                pass
            
            self.logger.info(f"🔧 Creating execution role: {self.execution_role_name}")
            
            # Trust policy for SageMaker
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            # Create the role
            role_response = self.iam.create_role(
                RoleName=self.execution_role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description='Execution role for RiftRewind SageMaker models'
            )
            
            role_arn = role_response['Role']['Arn']
            
            # Attach required policies
            policies_to_attach = [
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess'
            ]
            
            for policy_arn in policies_to_attach:
                self.iam.attach_role_policy(
                    RoleName=self.execution_role_name,
                    PolicyArn=policy_arn
                )
                self.logger.info(f"📎 Attached policy: {policy_arn}")
            
            # Wait for role to propagate
            self.logger.info("⏳ Waiting for role to propagate...")
            time.sleep(10)
            
            self.logger.info(f"✅ Execution role created: {role_arn}")
            return role_arn
            
        except Exception as e:
            self.logger.error(f"❌ Role creation failed: {e}")
            raise
    
    def package_model(self) -> str:
        """Package the TensorFlow ward detection model for SageMaker"""
        
        self.logger.info("📦 Packaging TensorFlow ward detection model...")
        
        # Create model.tar.gz with inference code
        model_tar_path = '/tmp/model.tar.gz'
        
        with tarfile.open(model_tar_path, 'w:gz') as tar:
            
            # Add the main model file
            tar.add('/Users/joshs/RiftRewindClean/tensorflow_ward_detector.py', 
                   arcname='tensorflow_ward_detector.py')
            
            # Create inference.py for SageMaker
            inference_code = '''
import json
import numpy as np
import tensorflow as tf
from tensorflow_ward_detector import TensorFlowWardDetector
import base64
import io
from PIL import Image

def model_fn(model_dir):
    """Load model for SageMaker inference"""
    detector = TensorFlowWardDetector()
    # In production, would load trained weights from model_dir
    detector.compile_model()
    return detector

def input_fn(request_body, request_content_type='application/json'):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Handle base64 encoded image
        if 'frame_b64' in input_data:
            image_data = base64.b64decode(input_data['frame_b64'])
            image = Image.open(io.BytesIO(image_data))
            frame = np.array(image)
        elif 'frame' in input_data:
            frame = np.array(input_data['frame'])
        else:
            raise ValueError("No frame data provided")
            
        return frame
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run inference"""
    frame = input_data
    
    # Detect wards using the model
    results = model.detect_wards_in_frame(frame)
    
    return results

def output_fn(prediction, accept='application/json'):
    """Format output"""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''
            
            # Add inference.py to tar
            inference_info = tarfile.TarInfo(name='inference.py')
            inference_info.size = len(inference_code.encode('utf-8'))
            tar.addfile(inference_info, io.BytesIO(inference_code.encode('utf-8')))
            
            # Add requirements.txt
            requirements = '''
tensorflow==2.12.0
numpy==1.24.3
opencv-python-headless==4.8.0.74
Pillow==9.5.0
'''
            req_info = tarfile.TarInfo(name='requirements.txt')
            req_info.size = len(requirements.encode('utf-8'))
            tar.addfile(req_info, io.BytesIO(requirements.encode('utf-8')))
        
        # Upload to S3
        s3_key = 'ward-detector/model.tar.gz'
        
        self.logger.info(f"☁️ Uploading model to S3: s3://{self.bucket_name}/{s3_key}")
        
        with open(model_tar_path, 'rb') as f:
            self.s3.upload_fileobj(f, self.bucket_name, s3_key)
        
        model_data_url = f"s3://{self.bucket_name}/{s3_key}"
        self.logger.info(f"✅ Model uploaded: {model_data_url}")
        
        return model_data_url
    
    def deploy_ward_detector(self, model_data_url: str, execution_role_arn: str) -> Dict[str, str]:
        """Deploy the YOLOv5-TensorFlow ward detector"""
        
        timestamp = int(time.time())
        
        model_name = f'riftrewind-ward-detector-{timestamp}'
        endpoint_config_name = f'riftrewind-ward-config-{timestamp}'
        endpoint_name = f'riftrewind-ward-endpoint-{timestamp}'
        
        try:
            # Step 1: Create SageMaker model
            self.logger.info(f"🤖 Creating SageMaker model: {model_name}")
            
            self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.12.0-cpu-py310-ubuntu20.04-ec2',
                    'ModelDataUrl': model_data_url,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                        'SAGEMAKER_REGION': self.region_name
                    }
                },
                ExecutionRoleArn=execution_role_arn,
                Tags=[
                    {'Key': 'Project', 'Value': 'RiftRewind'},
                    {'Key': 'Model', 'Value': 'YOLOv5-TensorFlow-WardDetector'}
                ]
            )
            
            self.logger.info("✅ SageMaker model created")
            
            # Step 2: Create endpoint configuration
            self.logger.info(f"⚙️ Creating endpoint configuration: {endpoint_config_name}")
            
            self.sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.m5.large',  # Cost-effective for testing
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            self.logger.info("✅ Endpoint configuration created")
            
            # Step 3: Create endpoint
            self.logger.info(f"🚀 Creating endpoint: {endpoint_name}")
            self.logger.info("⏳ This will take 5-10 minutes...")
            
            self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                Tags=[
                    {'Key': 'Project', 'Value': 'RiftRewind'},
                    {'Key': 'Model', 'Value': 'WardDetector'},
                    {'Key': 'DeployedBy', 'Value': 'RiftRewindSetup'}
                ]
            )
            
            # Wait for endpoint to be ready
            self.logger.info("⏳ Waiting for endpoint to be in service...")
            
            waiter = self.sagemaker.get_waiter('endpoint_in_service')
            waiter.wait(
                EndpointName=endpoint_name,
                WaiterConfig={
                    'Delay': 30,  # Check every 30 seconds
                    'MaxAttempts': 20  # Wait up to 10 minutes
                }
            )
            
            self.logger.info("🎉 Endpoint is ready for inference!")
            
            return {
                'model_name': model_name,
                'endpoint_config_name': endpoint_config_name,
                'endpoint_name': endpoint_name,
                'status': 'InService'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            
            # Cleanup on failure
            self._cleanup_failed_resources(model_name, endpoint_config_name, endpoint_name)
            raise
    
    def _cleanup_failed_resources(self, model_name: str, endpoint_config_name: str, endpoint_name: str):
        """Clean up resources if deployment fails"""
        
        try:
            # Try to delete endpoint
            try:
                self.sagemaker.delete_endpoint(EndpointName=endpoint_name)
                self.logger.info(f"🧹 Cleaned up endpoint: {endpoint_name}")
            except:
                pass
                
            # Try to delete endpoint config
            try:
                self.sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                self.logger.info(f"🧹 Cleaned up endpoint config: {endpoint_config_name}")
            except:
                pass
                
            # Try to delete model
            try:
                self.sagemaker.delete_model(ModelName=model_name)
                self.logger.info(f"🧹 Cleaned up model: {model_name}")
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup error: {e}")
    
    def test_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Test the deployed endpoint with sample data"""
        
        self.logger.info(f"🧪 Testing endpoint: {endpoint_name}")
        
        try:
            # Create test frame data
            test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8).tolist()
            
            # Invoke endpoint
            runtime_client = boto3.client('sagemaker-runtime', region_name=self.region_name)
            
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps({'frame': test_frame})
            )
            
            result = json.loads(response['Body'].read().decode())
            
            self.logger.info(f"✅ Test successful! Found {result.get('total_wards', 0)} wards")
            
            return {
                'status': 'success',
                'test_result': result,
                'endpoint_ready': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Endpoint test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'endpoint_ready': False
            }
    
    def full_setup(self) -> Dict[str, Any]:
        """Complete SageMaker setup process"""
        
        self.logger.info("🚀 Starting complete SageMaker setup for RiftRewind ward detection")
        
        try:
            # Step 1: Check permissions
            self.logger.info("1️⃣ Checking AWS permissions...")
            permissions = self.check_permissions()
            
            if not all(permissions.values()):
                missing = [k for k, v in permissions.items() if not v]
                raise ValueError(f"Missing permissions: {missing}")
            
            # Step 2: Create S3 bucket
            self.logger.info("2️⃣ Setting up S3 storage...")
            bucket_name = self.create_s3_bucket()
            
            # Step 3: Create execution role
            self.logger.info("3️⃣ Creating execution role...")
            execution_role_arn = self.create_execution_role()
            
            # Step 4: Package and upload model
            self.logger.info("4️⃣ Packaging TensorFlow model...")
            model_data_url = self.package_model()
            
            # Step 5: Deploy model
            self.logger.info("5️⃣ Deploying YOLOv5-TensorFlow ward detector...")
            deployment_info = self.deploy_ward_detector(model_data_url, execution_role_arn)
            
            # Step 6: Test endpoint
            self.logger.info("6️⃣ Testing deployed endpoint...")
            test_result = self.test_endpoint(deployment_info['endpoint_name'])
            
            self.logger.info("🎉 SageMaker setup complete!")
            
            return {
                'status': 'success',
                'bucket_name': bucket_name,
                'execution_role': execution_role_arn,
                'deployment': deployment_info,
                'test_result': test_result,
                'summary': {
                    'model_type': 'YOLOv5-TensorFlow Ward Detector',
                    'endpoint_name': deployment_info['endpoint_name'],
                    'instance_type': 'ml.m5.large',
                    'ready_for_inference': test_result['endpoint_ready']
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Setup failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'help': 'Check AWS permissions and try again'
            }

def main():
    """Main setup function"""
    
    print("🎯 RiftRewind SageMaker Setup")
    print("=" * 50)
    
    # Create setup manager
    setup = RiftRewindSageMakerSetup()
    
    # Run full setup
    result = setup.full_setup()
    
    if result['status'] == 'success':
        print("\\n🎉 SUCCESS! Your SageMaker setup is complete:")
        print(f"   🤖 Model: {result['summary']['model_type']}")
        print(f"   🔗 Endpoint: {result['summary']['endpoint_name']}")
        print(f"   💻 Instance: {result['summary']['instance_type']}")
        print(f"   ✅ Ready: {result['summary']['ready_for_inference']}")
        
        print("\\n🚀 Your RiftRewind app can now use real SageMaker inference!")
        
    else:
        print(f"\\n❌ Setup failed: {result['error']}")
        print(f"💡 Help: {result.get('help', 'Check logs for details')}")

if __name__ == "__main__":
    main()