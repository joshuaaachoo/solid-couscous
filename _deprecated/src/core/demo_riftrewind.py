#!/usr/bin/env python3
"""
RiftRewind Demo Script
Demonstrates the complete ML + AI pipeline for League of Legends coaching
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def demo_tensorflow_model():
    """Demonstrate TensorFlow ward detection model"""
    print("🤖 TensorFlow Ward Detection Model Demo")
    print("=" * 50)
    
    try:
        from tensorflow_ward_detector import TensorFlowWardDetector
        
        # Create model instance
        detector = TensorFlowWardDetector()
        
        print(f"📊 Model Classes: {detector.class_names}")
        print(f"📐 Input Size: {detector.input_size}")
        print(f"🎯 Confidence Threshold: {detector.confidence_threshold}")
        
        # Create model architecture
        print("\n🏗️ Creating model architecture...")
        model = detector.create_model_architecture()
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        print(f"🔧 Model input shape: {model.input_shape}")
        
        # Simulate ward detection
        print("\n🔍 Simulating ward detection...")
        import numpy as np
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # This would normally detect wards, but for demo we'll simulate
        print("📹 Processing frame shape:", dummy_frame.shape)
        print("✅ Ward detection simulation complete")
        
    except ImportError as e:
        print(f"⚠️ TensorFlow not installed: {e}")
        print("💡 Run: pip install tensorflow")
    except Exception as e:
        print(f"❌ Demo error: {e}")

def demo_sagemaker_deployment():
    """Demonstrate SageMaker model deployment"""
    print("\n🚀 AWS SageMaker Deployment Demo")
    print("=" * 50)
    
    try:
        from sagemaker_deployment import SageMakerModelDeployment
        
        # Initialize deployment manager
        deployer = SageMakerModelDeployment()
        
        print(f"📋 Available models: {list(deployer.model_configs.keys())}")
        
        # Show model configurations
        for model_name, config in deployer.model_configs.items():
            print(f"\n🔧 {model_name.upper()} Configuration:")
            print(f"   Framework: {config['framework']} {config['framework_version']}")
            print(f"   Instance: {config['instance_type']}")
            print(f"   Description: {config['description']}")
        
        # Get placeholder results for each model
        print("\n🎯 Generating sample analysis results...")
        
        sample_input = {"frame_data": "sample_video_frame"}
        
        for model_type in ['ward_detector', 'champion_recognition', 'minimap_analyzer']:
            result = deployer.get_model_placeholder_results(model_type, sample_input)
            print(f"\n📊 {model_type.upper()} Results:")
            
            if model_type == 'ward_detector':
                print(f"   Total wards: {result.get('total_wards', 0)}")
                print(f"   Vision coverage: {result.get('vision_coverage_score', 0):.0%}")
            elif model_type == 'champion_recognition':
                champions = result.get('champions_detected', [])
                print(f"   Champions detected: {len(champions)}")
            elif model_type == 'minimap_analyzer':
                control = result.get('map_control_percentage', 0)
                print(f"   Map control: {control:.0%}")
        
        print("\n✅ SageMaker deployment demo complete")
        
    except Exception as e:
        print(f"❌ SageMaker demo error: {e}")
        print("💡 Tip: Check AWS credentials configuration")

def demo_bedrock_coaching():
    """Demonstrate Bedrock AI coaching insights"""
    print("\n🧠 AWS Bedrock AI Coaching Demo")
    print("=" * 50)
    
    try:
        from bedrock_coaching_insights import BedrockCoachingInsights
        
        # Initialize Bedrock coach
        coach = BedrockCoachingInsights()
        
        print(f"🤖 AI Model: {coach.model_id}")
        
        # Sample ward analysis data
        sample_ward_data = {
            'total_wards': 6,
            'vision_coverage_score': 0.78,
            'strategic_rating': 'Good',
            'ward_type_distribution': {
                'Control Ward': 2,
                'Stealth Ward': 3,
                'Farsight Ward': 1,
                'Zombie Ward': 0
            },
            'wards_detected': [
                {'type': 'Control Ward', 'confidence': 0.92, 'position': {'x': 150, 'y': 200}},
                {'type': 'Stealth Ward', 'confidence': 0.87, 'position': {'x': 300, 'y': 450}},
                {'type': 'Stealth Ward', 'confidence': 0.84, 'position': {'x': 180, 'y': 320}}
            ]
        }
        
        print("\n📊 Sample ward analysis data:")
        print(json.dumps(sample_ward_data, indent=2))
        
        print("\n🎯 Generating coaching insights...")
        insights_result = coach.generate_ward_coaching_insights(sample_ward_data)
        
        print(f"\n📝 AI Coaching Insights:")
        print(f"Analysis Type: {insights_result['analysis_type']}")
        print(f"Confidence: {insights_result['confidence_score']:.0%}")
        
        # Show first part of insights
        insights_text = insights_result['insights']
        preview = insights_text[:300] + "..." if len(insights_text) > 300 else insights_text
        print(f"\nInsights Preview:\n{preview}")
        
        print("\n✅ Bedrock coaching demo complete")
        
    except Exception as e:
        print(f"❌ Bedrock demo error: {e}")
        print("💡 Tip: Check AWS Bedrock configuration and permissions")

def demo_complete_pipeline():
    """Demonstrate complete ML + AI pipeline"""
    print("\n🔗 Complete ML + AI Pipeline Demo")
    print("=" * 50)
    
    try:
        from bedrock_coaching_insights import MLCoachingPipeline
        from sagemaker_deployment import SageMakerModelDeployment
        from bedrock_coaching_insights import BedrockCoachingInsights
        
        # Initialize components
        sagemaker_deployment = SageMakerModelDeployment()
        bedrock_insights = BedrockCoachingInsights()
        
        # Create pipeline
        pipeline = MLCoachingPipeline(sagemaker_deployment, bedrock_insights)
        
        print("🔧 Pipeline components initialized")
        print("   ✅ SageMaker ML models")
        print("   ✅ Bedrock AI coaching")
        print("   ✅ ML + AI pipeline")
        
        # Sample input data (would be actual video/game data)
        sample_input = {
            "video_url": "https://youtube.com/watch?v=sample",
            "player_name": "Faker",
            "game_duration": 1800,  # 30 minutes
            "rank": "Challenger"
        }
        
        print(f"\n📥 Sample input: {json.dumps(sample_input, indent=2)}")
        
        print("\n⚙️ Running complete analysis pipeline...")
        results = pipeline.analyze_and_coach(sample_input)
        
        if 'ml_analysis' in results:
            print("   ✅ ML Analysis complete")
            ward_count = results['ml_analysis']['ward_analysis'].get('total_wards', 0)
            print(f"      Ward Analysis: {ward_count} wards detected")
            
            champions = results['ml_analysis']['champion_analysis'].get('champions_detected', [])
            print(f"      Champion Analysis: {len(champions)} champions detected")
            
            map_control = results['ml_analysis']['map_analysis'].get('map_control_percentage', 0)
            print(f"      Map Analysis: {map_control:.0%} control")
        
        if 'ai_coaching' in results:
            print("   ✅ AI Coaching complete")
            print("      Ward coaching insights generated")
            print("      Comprehensive gameplay coaching generated")
        
        if 'pipeline_summary' in results:
            summary = results['pipeline_summary']
            print(f"\n📋 Pipeline Summary:")
            print(f"   Processing time: {summary.get('total_analysis_time')}")
            print(f"   Models used: {summary.get('models_used')}")
            print(f"   Confidence: {summary.get('confidence_score', 0):.0%}")
        
        print("\n✅ Complete pipeline demo successful!")
        
    except Exception as e:
        print(f"❌ Pipeline demo error: {e}")

def show_system_overview():
    """Show complete system architecture overview"""
    print("🎮 RiftRewind: League of Legends AI Coach")
    print("=" * 60)
    print()
    print("🏗️  SYSTEM ARCHITECTURE:")
    print("   📹 Video Input (YouTube/Upload)")
    print("   ⬇️")
    print("   🤖 AWS SageMaker ML Models:")
    print("      • TensorFlow Ward Detection (YOLOv5)")
    print("      • Champion Recognition")  
    print("      • Minimap Analysis")
    print("   ⬇️")
    print("   🧠 AWS Bedrock AI Coaching:")
    print("      • Claude 3 Sonnet")
    print("      • Natural language insights")
    print("      • Personalized recommendations")
    print("   ⬇️")
    print("   🎯 Coaching Output")
    print()
    print("🔧 TECHNOLOGIES:")
    print("   • TensorFlow 2.12 (Deep Learning)")
    print("   • AWS SageMaker (ML Deployment)")
    print("   • AWS Bedrock (AI Coaching)")
    print("   • AWS S3 (Video Storage)")
    print("   • Streamlit (Web Interface)")
    print("   • YOLOv5 (Object Detection)")
    print()
    print("🎯 FEATURES:")
    print("   • Real-time ward detection")
    print("   • Professional gameplay analysis")
    print("   • AI-powered coaching insights")
    print("   • VOD upload and analysis")
    print("   • Rank improvement plans")

def main():
    """Run complete demo"""
    show_system_overview()
    
    print("\n" + "=" * 60)
    print("🚀 RUNNING COMPONENT DEMOS")
    print("=" * 60)
    
    # Run individual component demos
    demo_tensorflow_model()
    demo_sagemaker_deployment()
    demo_bedrock_coaching()
    demo_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("=" * 60)
    print()
    print("🚀 To start the web interface:")
    print("   streamlit run app.py")
    print()
    print("📚 Next steps:")
    print("   1. Configure AWS credentials")
    print("   2. Set up .env file with API keys")
    print("   3. Deploy SageMaker models")
    print("   4. Enable AWS Bedrock access")
    print("   5. Upload training data for custom models")

if __name__ == "__main__":
    main()