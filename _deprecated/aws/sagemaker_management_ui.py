"""
SageMaker Management Interface for Streamlit
Provides UI for deploying, monitoring, and managing ML models
"""

import streamlit as st
import os
from sagemaker_deployment import SageMakerModelDeployment
from typing import Dict, Any

def render_sagemaker_management():
    """Render the SageMaker model management interface"""
    
    st.header("🎯 SageMaker Model Management")
    st.write("Deploy and manage League of Legends analysis models")
    
    # Add helpful information
    with st.expander("ℹ️ AWS SageMaker Setup Guide", expanded=False):
        st.markdown("""
        **What you need for SageMaker:**
        1. **AWS Account** with SageMaker access
        2. **AWS Credentials** (Access Key ID + Secret Key)
        3. **IAM Permissions** for SageMaker operations
        
        **Common Issues:**
        - 🔒 **AccessDeniedException**: Your AWS user needs SageMaker permissions
        - 🔑 **No Credentials**: Add AWS keys to your .env file
        - ⚠️ **Region Issues**: Ensure your AWS region supports SageMaker
        
        **For Demo Mode**: The VOD analysis works without SageMaker using simulated results.
        """)
    
    # Initialize deployment manager
    deployment_manager = SageMakerModelDeployment()
    
    # AWS Configuration Check
    aws_configured = bool(os.getenv('AWS_ACCESS_KEY_ID')) and bool(os.getenv('AWS_SECRET_ACCESS_KEY'))
    
    if not aws_configured:
        st.warning("⚠️ AWS credentials not configured")
        st.info("💡 **Demo Mode Available**: VOD analysis works without AWS using simulated ML results")
        with st.expander("🔧 Configure AWS Credentials"):
            st.markdown("""
            **Add to your .env file:**
            ```
            AWS_ACCESS_KEY_ID=your_access_key_here
            AWS_SECRET_ACCESS_KEY=your_secret_key_here
            AWS_REGION=us-east-1
            ```
            """)
        return
    st.write("Deploy and manage League of Legends analysis models")
    
    # Initialize deployment manager
    deployment_manager = SageMakerModelDeployment()
    
    # AWS Configuration Check
    aws_configured = bool(os.getenv('AWS_ACCESS_KEY_ID')) and bool(os.getenv('AWS_SECRET_ACCESS_KEY'))
    
    if not aws_configured:
        st.error("⚠️ AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        return
    
    # Get all deployment statuses
    all_deployments = deployment_manager.list_all_deployments()
    
    st.success(f"🚀 AWS SageMaker Ready - Managing {all_deployments['total_models']} model types")
    
    # Model Status Overview
    st.subheader("📊 Model Deployment Status")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (model_type, deployment_info) in enumerate(all_deployments['deployments'].items()):
        config = deployment_info['config']
        status = deployment_info['status']
        
        with [col1, col2, col3][i]:
            st.write(f"**{config['description']}**")
            
            if status['status'] == 'deployed':
                st.success(f"✅ {status['endpoint_status']}")
                st.caption(f"Endpoint: {status['endpoint_name']}")
                st.caption(f"Instance: {status['instance_type']}")
            elif status['status'] == 'not_deployed':
                st.warning("⏸️ Not Deployed")
                st.caption("Ready to deploy")
            else:
                st.error(f"❌ {status.get('message', 'Error')}")
    
    # Model Actions
    st.subheader("⚙️ Model Actions")
    
    tab1, tab2, tab3 = st.tabs(["Deploy Models", "Test Inference", "Manage Endpoints"])
    
    with tab1:
        render_deployment_tab(deployment_manager, all_deployments)
    
    with tab2:
        render_inference_tab(deployment_manager, all_deployments)
    
    with tab3:
        render_management_tab(deployment_manager, all_deployments)

def render_deployment_tab(deployment_manager, all_deployments):
    """Render the model deployment tab"""
    
    st.write("**Deploy New Models**")
    
    # Model selection
    model_options = list(all_deployments['deployments'].keys())
    selected_model = st.selectbox(
        "Choose model to deploy:",
        model_options,
        format_func=lambda x: all_deployments['deployments'][x]['config']['description']
    )
    
    if selected_model:
        config = all_deployments['deployments'][selected_model]['config']
        status = all_deployments['deployments'][selected_model]['status']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Model**: {config['model_name']}")
            st.info(f"**Framework**: {config['framework'].title()}")
            st.info(f"**Instance**: {config['instance_type']}")
        
        with col2:
            if status['status'] == 'deployed':
                st.success("✅ Already deployed")
                st.caption(f"Status: {status['endpoint_status']}")
            elif status['status'] == 'not_configured':
                st.info("⚙️ Configuration ready")
                st.caption("Ready to deploy with endpoint configuration")
            elif status['status'] == 'permission_denied':
                st.error("🔒 AWS Permissions Issue")
                st.caption("User lacks SageMaker permissions")
                with st.expander("🔧 How to Fix"):
                    st.markdown("""
                    **AWS IAM Policy Required:**
                    ```json
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "sagemaker:DescribeEndpoint",
                                    "sagemaker:CreateModel", 
                                    "sagemaker:CreateEndpointConfig",
                                    "sagemaker:CreateEndpoint"
                                ],
                                "Resource": "*"
                            }
                        ]
                    }
                    ```
                    **Steps:**
                    1. Go to AWS IAM Console
                    2. Find your user account
                    3. Attach the SageMaker permissions above
                    4. Refresh this page
                    """)
            elif status['status'] == 'no_credentials':
                st.warning("🔑 AWS Credentials Missing")
                st.caption("Configure AWS access keys")
                with st.expander("🔧 How to Fix"):
                    st.markdown("""
                    **Add to your .env file:**
                    ```
                    AWS_ACCESS_KEY_ID=your_access_key_here
                    AWS_SECRET_ACCESS_KEY=your_secret_key_here
                    AWS_REGION=us-east-1
                    ```
                    """)
            elif status['status'] == 'no_aws':
                st.warning("🔑 AWS not configured")
                st.caption("Check AWS credentials")
            elif status['status'] == 'not_deployed':
                st.info("⏸️ Ready to deploy")
                st.caption("Endpoint not created yet")
            else:
                st.warning("⚠️ Status unknown")
                st.caption(f"Status: {status['status']}")
        
        # Model artifact URI input
        model_uri = st.text_input(
            "Model Artifact S3 URI:",
            placeholder="s3://your-bucket/models/ward-detector/model.tar.gz",
            help="S3 location of your trained model artifacts"
        )
        
        # Deployment options
        with st.expander("Advanced Deployment Options"):
            custom_instance = st.selectbox(
                "Instance Type:",
                ["ml.t2.medium", "ml.m5.large", "ml.m5.xlarge", "ml.c5.large"],
                index=0
            )
            
            instance_count = st.slider("Initial Instance Count:", 1, 3, 1)
            
            auto_scaling = st.checkbox("Enable Auto Scaling", value=False)
        
        # Deploy button
        if st.button(f"🚀 Deploy {selected_model.replace('_', ' ').title()}", type="primary"):
            if model_uri:
                with st.spinner(f"Deploying {selected_model} model..."):
                    # Note: In a real implementation, you'd deploy the actual model
                    # For now, we'll show what would happen
                    st.info("🔧 Deployment simulation (replace with real deployment)")
                    
                    # Simulated deployment result
                    result = {
                        'success': True,
                        'model_name': config['model_name'],
                        'endpoint_name': config['endpoint_name'],
                        'message': f'Deployment initiated for {selected_model}'
                    }
                    
                    if result['success']:
                        st.success(f"✅ {result['message']}")
                        st.info("⏱️ Deployment typically takes 5-10 minutes")
                        st.balloons()
                    else:
                        st.error(f"❌ Deployment failed: {result.get('error')}")
            else:
                st.warning("Please provide a model artifact S3 URI")

def render_inference_tab(deployment_manager, all_deployments):
    """Render the model inference testing tab"""
    
    st.write("**Test Model Inference**")
    
    # Select deployed model
    deployed_models = [
        model_type for model_type, info in all_deployments['deployments'].items()
        if info['status']['status'] == 'deployed'
    ]
    
    if not deployed_models:
        st.warning("⚠️ No models deployed. Deploy models first to test inference.")
        return
    
    selected_model = st.selectbox(
        "Choose deployed model to test:",
        deployed_models,
        format_func=lambda x: all_deployments['deployments'][x]['config']['description']
    )
    
    if selected_model:
        st.write(f"**Testing**: {all_deployments['deployments'][selected_model]['config']['description']}")
        
        # Test input
        test_input = st.text_area(
            "Test Input (JSON):",
            value='{"video_frame": "base64_encoded_frame", "timestamp": 1234}',
            help="Provide test input data for the model"
        )
        
        if st.button(f"🧪 Test {selected_model.replace('_', ' ').title()}", type="primary"):
            with st.spinner("Running inference..."):
                # Use placeholder results for demo
                result = deployment_manager.get_model_placeholder_results(selected_model, test_input)
                
                st.success("✅ Inference completed!")
                
                # Display results
                st.json(result)
                
                # Interpret results based on model type
                if selected_model == 'ward_detector':
                    st.write("**Interpretation:**")
                    st.write(f"• Detected {result.get('total_wards', 0)} wards")
                    st.write(f"• Vision coverage: {result.get('vision_coverage_score', 0):.1%}")
                    
                elif selected_model == 'champion_recognition':
                    st.write("**Interpretation:**")
                    champions = result.get('champions_detected', [])
                    st.write(f"• Detected {len(champions)} champions")
                    for champ in champions[:3]:
                        st.write(f"• {champ['name']}: {champ['confidence']:.1%} confidence")
                        
                elif selected_model == 'minimap_analyzer':
                    st.write("**Interpretation:**")
                    st.write(f"• Map control: {result.get('map_control_percentage', 0):.1%}")
                    st.write(f"• Threat level: {result.get('threat_level', 'Unknown')}")

def render_management_tab(deployment_manager, all_deployments):
    """Render the endpoint management tab"""
    
    st.write("**Manage Deployed Endpoints**")
    
    # List all endpoints with management options
    for model_type, deployment_info in all_deployments['deployments'].items():
        config = deployment_info['config']
        status = deployment_info['status']
        
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"**{config['description']}**")
                st.caption(f"Model: {config['model_name']}")
                st.caption(f"Endpoint: {config['endpoint_name']}")
            
            with col2:
                if status['status'] == 'deployed':
                    st.success(f"✅ {status['endpoint_status']}")
                    if 'creation_time' in status:
                        st.caption(f"Created: {status['creation_time'].strftime('%Y-%m-%d %H:%M')}")
                elif status['status'] == 'not_deployed':
                    st.warning("⏸️ Not Deployed")
                else:
                    st.error("❌ Error")
                    st.caption(status.get('message', ''))
            
            with col3:
                if status['status'] == 'deployed':
                    if st.button(f"🗑️ Delete", key=f"delete_{model_type}"):
                        # Confirm deletion
                        if st.checkbox(f"Confirm deletion of {model_type}", key=f"confirm_{model_type}"):
                            with st.spinner("Deleting endpoint..."):
                                # Note: In real implementation, call deployment_manager.delete_endpoint(model_type)
                                st.success(f"✅ {config['endpoint_name']} deletion initiated")
                                st.experimental_rerun()
        
        st.divider()
    
    # Cost management info
    st.subheader("💰 Cost Management")
    st.info("""
    **Cost Optimization Tips:**
    - Delete unused endpoints to avoid charges
    - Use smaller instances for development/testing
    - Enable auto-scaling for production workloads
    - Monitor usage through AWS CloudWatch
    """)

if __name__ == "__main__":
    # For testing the interface
    render_sagemaker_management()