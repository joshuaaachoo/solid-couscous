#!/usr/bin/env python3
"""
SageMaker Endpoint Status Checker
Monitor your deployed ward detection endpoint
"""

import boto3
import json
import time
from datetime import datetime
import sys

class EndpointMonitor:
    def __init__(self, region_name='us-east-1'):
        self.sagemaker = boto3.client('sagemaker', region_name=region_name)
        self.runtime = boto3.client('sagemaker-runtime', region_name=region_name)
        
    def check_endpoint_status(self, endpoint_name: str = None):
        """Check status of specific endpoint or find latest"""
        
        try:
            if not endpoint_name:
                # Find the latest RiftRewind endpoint
                endpoints = self.sagemaker.list_endpoints(
                    NameContains='riftrewind-endpoint',
                    SortBy='CreationTime',
                    SortOrder='Descending',
                    MaxResults=1
                )
                
                if not endpoints.get('Endpoints'):
                    return {'status': 'no_endpoints', 'message': 'No RiftRewind endpoints found'}
                
                endpoint_name = endpoints['Endpoints'][0]['EndpointName']
            
            # Get detailed status
            details = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            
            return {
                'status': details['EndpointStatus'].lower(),
                'endpoint_name': endpoint_name,
                'creation_time': details['CreationTime'].strftime('%H:%M:%S'),
                'failure_reason': details.get('FailureReason'),
                'details': details
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def wait_for_endpoint(self, endpoint_name: str = None, max_wait_minutes: int = 10):
        """Wait for endpoint to be ready"""
        
        print(f"⏳ Waiting for endpoint to be ready (max {max_wait_minutes} minutes)...")
        
        status_info = self.check_endpoint_status(endpoint_name)
        if status_info['status'] == 'error':
            print(f"❌ Error: {status_info['message']}")
            return False
        
        endpoint_name = status_info['endpoint_name']
        start_time = time.time()
        
        while time.time() - start_time < max_wait_minutes * 60:
            status = self.check_endpoint_status(endpoint_name)
            
            print(f"🔄 {datetime.now().strftime('%H:%M:%S')} - Status: {status['status']}")
            
            if status['status'] == 'inservice':
                print(f"✅ Endpoint ready: {endpoint_name}")
                return True
            elif status['status'] == 'failed':
                print(f"❌ Endpoint failed: {status.get('failure_reason', 'Unknown error')}")
                return False
            
            time.sleep(30)  # Check every 30 seconds
        
        print("⏰ Timeout reached - endpoint may still be deploying")
        return False
    
    def test_endpoint(self, endpoint_name: str = None):
        """Test endpoint with sample data"""
        
        status_info = self.check_endpoint_status(endpoint_name)
        if status_info['status'] != 'inservice':
            return {'status': 'not_ready', 'message': f"Endpoint status: {status_info['status']}"}
        
        endpoint_name = status_info['endpoint_name']
        
        try:
            print(f"🧪 Testing endpoint: {endpoint_name}")
            
            # Test data
            test_data = {
                'frame': [[1, 2, 3], [4, 5, 6]]
            }
            
            response = self.runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test_data)
            )
            
            result = json.loads(response['Body'].read().decode())
            
            print(f"✅ Test successful!")
            print(f"   Ward detections: {result['total_wards']}")
            print(f"   Inference time: {result['inference_time_ms']}ms")
            print(f"   Model: {result['model_info']['name']}")
            
            return {'status': 'success', 'result': result, 'endpoint_name': endpoint_name}
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return {'status': 'failed', 'error': str(e)}

def main():
    print("🎯 RiftRewind SageMaker Endpoint Monitor")
    print("=" * 45)
    
    monitor = EndpointMonitor()
    
    # Check current status
    status = monitor.check_endpoint_status()
    
    if status['status'] == 'no_endpoints':
        print("❌ No RiftRewind endpoints found")
        return
    elif status['status'] == 'error':
        print(f"❌ Error: {status['message']}")
        return
    
    endpoint_name = status['endpoint_name']
    print(f"📊 Endpoint: {endpoint_name}")
    print(f"🕐 Created: {status['creation_time']}")
    print(f"📈 Status: {status['status']}")
    
    if status['status'] == 'creating':
        print("\\n⏳ Endpoint is still deploying...")
        
        # Wait for it to be ready
        ready = monitor.wait_for_endpoint(endpoint_name)
        
        if ready:
            # Test it
            test_result = monitor.test_endpoint(endpoint_name)
            
            if test_result['status'] == 'success':
                print("\\n🎉 SUCCESS! Your SageMaker endpoint is ready!")
                print(f"🔗 Endpoint name: {test_result['endpoint_name']}")
                print("\\n🚀 Update your RiftRewind app with this endpoint!")
                
                # Show update instructions
                print("\\n📝 App Integration:")
                print("   1. Open your RiftRewind app")
                print("   2. SageMaker models should now show 'InService'")
                print(f"   3. Endpoint name: {test_result['endpoint_name']}")
                print("   4. Real ward detection is now active! 🎯")
                
                return test_result['endpoint_name']
            
    elif status['status'] == 'inservice':
        print("\\n✅ Endpoint is already ready!")
        
        # Test it
        test_result = monitor.test_endpoint(endpoint_name)
        
        if test_result['status'] == 'success':
            print("\\n🎉 Endpoint working perfectly!")
            return endpoint_name
    
    elif status['status'] == 'failed':
        print(f"\\n❌ Endpoint failed: {status.get('failure_reason', 'Unknown error')}")
    
    return None

if __name__ == "__main__":
    endpoint_name = main()
    
    if endpoint_name:
        print(f"\\n🎯 Your ward detection endpoint is ready: {endpoint_name}")
        print("💰 Cost: ~$0.056/hour (ml.t2.medium)")
        print("🛑 Remember to delete the endpoint when done to save costs!")
    else:
        print("\\n⏳ Check back in a few minutes or run this script again")