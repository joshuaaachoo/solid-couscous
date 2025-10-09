#!/usr/bin/env python3
"""
Quick SageMaker Cost Checker
Run this anytime to see if you're being charged
"""

import boto3
from datetime import datetime

def check_sagemaker_costs():
    try:
        client = boto3.client('sagemaker', region_name='us-east-1')
        endpoints = client.list_endpoints()['Endpoints']
        
        print("💰 SAGEMAKER COST CHECK")
        print("=" * 40)
        print(f"🕐 Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if not endpoints:
            print("✅ NO ENDPOINTS RUNNING")
            print("💰 Current cost: $0.00/hour")
            print("🎉 You're not being charged!")
        else:
            total_cost = 0
            print(f"⚠️  {len(endpoints)} ACTIVE ENDPOINTS:")
            print()
            
            for ep in endpoints:
                name = ep['EndpointName']
                status = ep['EndpointStatus']
                created = ep['CreationTime'].strftime('%Y-%m-%d %H:%M')
                
                # Estimate cost (ml.m5.large ≈ $0.096/hour)
                estimated_cost = 0.096
                total_cost += estimated_cost
                
                print(f"📊 {name}")
                print(f"   Status: {status}")
                print(f"   Created: {created}")
                print(f"   Cost: ~${estimated_cost:.3f}/hour")
                print()
            
            print(f"💸 TOTAL ESTIMATED COST: ${total_cost:.3f}/hour")
            print(f"💸 Daily cost: ~${total_cost * 24:.2f}")
            print(f"💸 Monthly cost: ~${total_cost * 24 * 30:.2f}")
            print()
            print("🛑 TO STOP CHARGES:")
            print("   python3 -c \"import boto3; [boto3.client('sagemaker').delete_endpoint(EndpointName=ep['EndpointName']) for ep in boto3.client('sagemaker').list_endpoints()['Endpoints']]\"")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure AWS credentials are configured")

if __name__ == "__main__":
    check_sagemaker_costs()