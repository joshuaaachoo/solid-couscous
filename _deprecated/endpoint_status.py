#!/usr/bin/env python3
"""
Quick SageMaker Endpoint Status for RiftRewind App
"""

import boto3
import json

def get_ward_detector_status():
    """Get current ward detector endpoint status"""
    
    try:
        sagemaker = boto3.client('sagemaker', region_name='us-east-1')
        
        # Check the specific endpoint we deployed
        endpoint_name = 'riftrewind-endpoint-simple-1759421779'
        
        details = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        
        status = details['EndpointStatus'].lower()
        
        if status == 'inservice':
            return {
                'status': 'running',
                'endpoint_name': endpoint_name,
                'message': 'YOLOv5-TensorFlow ward detector ready for inference',
                'ready': True
            }
        elif status == 'creating':
            return {
                'status': 'deploying',
                'endpoint_name': endpoint_name,
                'message': 'Ward detector deploying (5-15 minutes)',
                'ready': False
            }
        elif status == 'failed':
            return {
                'status': 'error',
                'endpoint_name': endpoint_name,
                'message': f"Deployment failed: {details.get('FailureReason', 'Unknown')}",
                'ready': False
            }
        else:
            return {
                'status': status,
                'endpoint_name': endpoint_name,
                'message': f"Status: {status}",
                'ready': False
            }
            
    except Exception as e:
        if 'Could not find endpoint' in str(e):
            return {
                'status': 'not_deployed',
                'message': 'No ward detector endpoint deployed',
                'ready': False
            }
        else:
            return {
                'status': 'aws_error',
                'message': f"AWS Error: {str(e)}",
                'ready': False
            }

def test_ward_detector():
    """Test the ward detector endpoint"""
    
    status = get_ward_detector_status()
    
    if not status['ready']:
        return status
    
    try:
        runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        
        # Test with sample data
        test_data = {'frame': [[1, 2, 3], [4, 5, 6]]}
        
        response = runtime.invoke_endpoint(
            EndpointName=status['endpoint_name'],
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        return {
            'status': 'success',
            'endpoint_name': status['endpoint_name'],
            'test_result': result,
            'message': f"Test successful - detected {result['total_wards']} wards",
            'ready': True
        }
        
    except Exception as e:
        return {
            'status': 'test_failed',
            'endpoint_name': status['endpoint_name'],
            'message': f"Endpoint test failed: {str(e)}",
            'ready': False
        }

if __name__ == "__main__":
    print("🔍 Checking Ward Detector Status...")
    
    status = get_ward_detector_status()
    print(f"Status: {status}")
    
    if status['ready']:
        print("\\n🧪 Testing endpoint...")
        test = test_ward_detector()
        print(f"Test: {test}")