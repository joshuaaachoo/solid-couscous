#!/usr/bin/env python3
"""
RiftRewind SageMaker Deployment Script - No Tags Version
Complete setup for YOLOv5-TensorFlow ward detection endpoints without tagging
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

class RiftRewindSageMakerSetupNoTags:
    """
    Complete SageMaker setup for RiftRewind ward detection - without tags to avoid permission issues
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
        
        # Use existing resources from previous run
        self.bucket_name = 'riftrewind-models-1759421611'  # From previous run
        self.execution_role_arn = 'arn:aws:iam::407275151589:role/RiftRewind-SageMaker-ExecutionRole'
        
    def deploy_ward_detector_no_tags(self) -> Dict[str, str]:
        """Deploy the YOLOv5-TensorFlow ward detector without tags"""
        
        timestamp = int(time.time())
        
        model_name = f'riftrewind-ward-detector-{timestamp}'
        endpoint_config_name = f'riftrewind-ward-config-{timestamp}'
        endpoint_name = f'riftrewind-ward-endpoint-{timestamp}'
        
        model_data_url = f"s3://{self.bucket_name}/ward-detector/model.tar.gz"
        
        try:
            # Step 1: Create SageMaker model (NO TAGS)
            self.logger.info(f"🤖 Creating SageMaker model: {model_name}")
            
            self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.12.0-cpu-py39-ubuntu20.04-sagemaker',
                    'ModelDataUrl': model_data_url,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                        'SAGEMAKER_REGION': self.region_name
                    }
                },
                ExecutionRoleArn=self.execution_role_arn
                # NO TAGS - This was causing the permission error
            )
            
            self.logger.info("✅ SageMaker model created (no tags)")
            
            # Step 2: Create endpoint configuration (NO TAGS)
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
                # NO TAGS
            )
            
            self.logger.info("✅ Endpoint configuration created (no tags)")
            
            # Step 3: Create endpoint (NO TAGS)
            self.logger.info(f"🚀 Creating endpoint: {endpoint_name}")
            self.logger.info("⏳ This will take 5-10 minutes...")
            
            self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
                # NO TAGS
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
    
    def quick_deploy(self) -> Dict[str, Any]:
        """Quick deployment using existing resources"""
        
        self.logger.info("🚀 Quick SageMaker deployment for RiftRewind ward detection")
        
        try:
            # Deploy model without tags
            self.logger.info("🤖 Deploying YOLOv5-TensorFlow ward detector...")
            deployment_info = self.deploy_ward_detector_no_tags()
            
            # Test endpoint
            self.logger.info("🧪 Testing deployed endpoint...")
            test_result = self.test_endpoint(deployment_info['endpoint_name'])
            
            self.logger.info("🎉 Quick deployment complete!")
            
            return {
                'status': 'success',
                'deployment': deployment_info,
                'test_result': test_result,
                'summary': {
                    'model_type': 'YOLOv5-TensorFlow Ward Detector',
                    'endpoint_name': deployment_info['endpoint_name'],
                    'instance_type': 'ml.m5.large',
                    'ready_for_inference': test_result['endpoint_ready'],
                    'bucket_used': self.bucket_name,
                    'execution_role': self.execution_role_arn
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Quick deployment failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'help': 'Check AWS permissions and try again'
            }

def main():
    """Main setup function"""
    
    print("🎯 RiftRewind SageMaker Quick Deploy (No Tags)")
    print("=" * 50)
    
    # Create setup manager
    setup = RiftRewindSageMakerSetupNoTags()
    
    # Run quick deployment
    result = setup.quick_deploy()
    
    if result['status'] == 'success':
        print("\n🎉 SUCCESS! Your SageMaker endpoint is deployed:")
        print(f"   🤖 Model: {result['summary']['model_type']}")
        print(f"   🔗 Endpoint: {result['summary']['endpoint_name']}")
        print(f"   💻 Instance: {result['summary']['instance_type']}")
        print(f"   ✅ Ready: {result['summary']['ready_for_inference']}")
        print(f"   🪣 S3 Bucket: {result['summary']['bucket_used']}")
        
        print("\n🚀 Your RiftRewind app can now use real SageMaker inference!")
        print("💰 Cost: ~$0.10/hour for ml.m5.large instance")
        print("🔗 Update your app to use endpoint:", result['summary']['endpoint_name'])
        
    else:
        print(f"\n❌ Deployment failed: {result['error']}")
        print(f"💡 Help: {result.get('help', 'Check logs for details')}")

if __name__ == "__main__":
    main()