from dotenv import load_dotenv
load_dotenv()
import os
import sys
from pathlib import Path
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from api.riot_api_client import RiotApiClient
from ai_coach.bedrock_insights import BedrockInsightsGenerator
# ML engine and other imports commented out - focusing on AI coaching
# from riftrewind_ml_engine import RiftRewindMLEngine
# from enhanced_coaching import EnhancedCoachingAnalyzer
# from progressive_tracker import ProgressiveTracker

class RiftRewindApp:
    """
    Main application class that orchestrates the complete analysis pipeline
    Designed for hackathon demo and AWS deployment
    """
    print("AWS ACCESS:", os.getenv("AWS_ACCESS_KEY"))
    print("AWS SECRET:", os.getenv("AWS_SECRET_ACCESS_KEY"))
    print("AWS REGION:", os.getenv("AWS_REGION"))

    def __init__(self, riot_api_key: str, aws_region: str = 'us-east-1'):
        self.riot_api_key = riot_api_key
        self.aws_region = aws_region
        # ML engine commented out - focusing on AI coaching with Bedrock
        # self.ml_engine = RiftRewindMLEngine()
        self.insights_generator = BedrockInsightsGenerator(aws_region)
        # self.coaching_analyzer = EnhancedCoachingAnalyzer()
        # self.tracker = ProgressiveTracker()
        # Real SageMaker ward detection now integrated!
        self.player_cache = {}
        self.cache_timeout = 300  # 5 minutes
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def analyze_player_complete(self, game_name: str, tag_line: str, match_count: int = 50) -> Dict:
        start_time = time.time()
        self.logger.info(f"Starting analysis for {game_name}#{tag_line}")
        try:
            # Step 1: Fetch player data from Riot API
            self.logger.info("Fetching player data from Riot API...")
            async with RiotApiClient(self.riot_api_key, "americas") as riot_client:
                player_data = await riot_client.get_comprehensive_player_data(
                    game_name, tag_line, match_count
                )
            if 'error' in player_data:
                return {'success': False, 'error': player_data['error']}
            if not player_data.get('matches'):
                return {'success': False, 'error': 'No match data found'}

            # Step 2: ML Analysis
            self.logger.info("Running ML analysis...")
            player_profile = self.ml_engine.create_player_profile(player_data['matches'])
            
            # Check if ML analysis failed
            if isinstance(player_profile, dict) and 'error' in player_profile:
                return {'success': False, 'error': f"ML analysis failed: {player_profile['error']}"}
            
            playstyle = self._determine_playstyle_simple(player_profile)
            strengths = self._identify_strengths(player_profile)
            improvements = self._identify_improvement_areas(player_profile)
            coaching_insights = self.ml_engine.generate_coaching_insights(player_profile, playstyle)
            ml_insights = {
                'playstyle': playstyle,
                'coaching_insights': coaching_insights,
                'strengths': strengths,
                'improvement_areas': improvements,
                'player_profile': player_profile
            }

            # Step 3: Generate AI Insights
            self.logger.info("Generating natural language insights...")
            ai_insights = self.insights_generator.generate_complete_insights_package(
                player_data, ml_insights
            )
            self.logger.info("Natural language insights generation complete.")

            # Step 4: Enhanced Coaching Analysis
            self.logger.info("Performing enhanced coaching analysis...")
            player_puuid = player_data['player_info']['puuid']
            coaching_analysis = self.coaching_analyzer.analyze_death_patterns(
                player_data['matches'], player_puuid
            )
            
            # Step 5: Progressive Tracking
            self.logger.info("Updating progressive tracking...")
            player_id = f"{game_name}_{tag_line}"
            session_data = {
                'match_count': match_count,
                'stats': ml_insights['player_profile'],
                'coaching_feedback': coaching_analysis
            }
            self.tracker.save_session_data(player_id, session_data)
            progression = self.tracker.analyze_progression(player_id)
            
            # Step 6: Pro-Level Vision Control Analysis (Migrated to SageMaker)
            self.logger.info("Vision analysis migrated to AWS SageMaker...")
            player_role = player_profile.get('primary_role', 'UTILITY')
            
            # TODO: Replace with SageMaker vision analysis
            pro_analysis_results = {'strategic_alignment_score': 0.75, 'pro_pattern_gaps': []}
            pro_vision_recommendations = ['Focus on deeper warding', 'Improve objective vision']
            
            # Step 7: Enhanced AI Insights with Coaching
            self.logger.info("Generating enhanced coaching insights...")
            enhanced_insights = self.insights_generator.generate_enhanced_coaching_insights(
                player_data, ml_insights, coaching_analysis
            )

            # Step 8: Compile final report
            processing_time = time.time() - start_time
            report = {
                'success': True,
                'player_info': player_data['player_info'],
                'raw_matches': player_data['matches'],  #   Include raw match data for death heatmap
                'summary_stats': {
                    'total_games': len(player_data['matches']),
                    'win_rate': float(player_profile.get('win_rate', 0)),
                    'avg_kda': float(player_profile.get('avg_kda', 1.0)),
                    'avg_cs_per_min': float(player_profile.get('avg_cs_per_min', 5.0)),
                    'avg_vision_per_min': float(player_profile.get('avg_vision_per_min', 0)),
                    'primary_role': player_profile.get('primary_role', 'UTILITY'),
                    'champion_diversity': int(player_profile.get('champion_diversity', 0))
                },
                'ml_analysis': {
                    'playstyle': playstyle,
                    'coaching_insights': coaching_insights,
                    'improvement_areas': improvements,
                    'strengths': strengths
                },
                'ai_insights': ai_insights,
                'enhanced_coaching': {
                    'tactical_analysis': coaching_analysis,
                    'progression_data': progression,
                    'enhanced_insights': enhanced_insights
                },
                'pro_vision_analysis': {
                    'analysis_results': pro_analysis_results,
                    'recommendations': pro_vision_recommendations,
                    'pro_alignment_score': pro_analysis_results.get('strategic_alignment_score', 0.0),
                    'missing_patterns': pro_analysis_results.get('pro_pattern_gaps', [])
                },
                'processing_time': round(processing_time, 2),
                'generated_at': datetime.now().isoformat()
            }
            self.logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            return report

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            return {'success': False, 'error': str(e)}

    def _determine_playstyle_simple(self, player_profile: Dict) -> Dict:
        """Simplified playstyle determination with error handling"""
        if not isinstance(player_profile, dict):
            return {'cluster_id': 2, 'playstyle_name': 'Unknown', 'description': 'Unable to determine playstyle from data.'}
        
        kda = player_profile.get('avg_kda', 1.0)
        win_rate = player_profile.get('win_rate', 0)
        
        if kda > 3 and win_rate > 0.6:
            return {'cluster_id': 0, 'playstyle_name': 'Aggressive Carry', 'description': 'High KDA & win rate, likely aggressive play.'}
        elif kda < 1.5 and win_rate < 0.4:
            return {'cluster_id': 1, 'playstyle_name': 'Defensive', 'description': 'Low KDA & win rate, likely cautious play.'}
        else:
            return {'cluster_id': 2, 'playstyle_name': 'Balanced', 'description': 'Average KDA & win rate, well-rounded style.'}

    def _identify_strengths(self, player_profile):
        strengths = []
        if player_profile.get('avg_kda', 0) > 3:
            strengths.append("High KDA")
        if player_profile.get('avg_cs_per_min', 0) > 7:
            strengths.append("Excellent farming")
        if player_profile.get('avg_vision_per_min', 0) > 1.5:
            strengths.append("Vision control")
        if player_profile.get('win_rate', 0) > 0.6:
            strengths.append("Winning streak")
        return strengths

    def _identify_improvement_areas(self, player_profile):
        improvements = []
        if player_profile.get('avg_kda', 0) < 2:
            improvements.append("Positioning and death reduction")
        if player_profile.get('avg_cs_per_min', 0) < 6:
            improvements.append("Improve farming efficiency")
        if player_profile.get('avg_vision_per_min', 0) < 1:
            improvements.append("Vision control")
        if player_profile.get('win_rate', 0) < 0.5:
            improvements.append("Fundamental skills")
        return improvements
    
    async def analyze_player_quick(self, game_name: str, tag_line: str, match_count: int = 5) -> Dict:
        """Quick player analysis - basic ML only, no AI generation"""
        start_time = time.time()
        self.logger.info(f"Starting QUICK analysis for {game_name}#{tag_line}")
        
        try:
            # Step 1: Fetch fewer matches for speed
            async with RiotApiClient(self.riot_api_key, "americas") as riot_client:
                player_data = await riot_client.get_comprehensive_player_data(
                    game_name, tag_line, match_count
                )
            
            if 'error' in player_data:
                return {'success': False, 'error': player_data['error']}
            if not player_data.get('matches'):
                return {'success': False, 'error': 'No match data found'}

            # Step 2: Basic ML Analysis Only (no AI generation)
            player_profile = self.ml_engine.create_player_profile(player_data['matches'])
            
            # Check if ML analysis failed
            if isinstance(player_profile, dict) and 'error' in player_profile:
                return {'success': False, 'error': f"ML analysis failed: {player_profile['error']}"}
            
            playstyle = self._determine_playstyle_simple(player_profile)
            strengths = self._identify_strengths(player_profile)
            improvements = self._identify_improvement_areas(player_profile)
            
            # Step 3: Basic coaching analysis (no death pattern analysis)
            player_puuid = player_data['player_info']['puuid']
            basic_stats = {
                'matches_analyzed': len(player_data['matches']),
                'avg_kda': float(player_profile.get('avg_kda', 0)),
                'win_rate': float(player_profile.get('win_rate', 0)),
                'avg_cs_per_min': float(player_profile.get('avg_cs_per_min', 0)),
                'avg_vision_per_min': float(player_profile.get('avg_vision_per_min', 0))
            }
            
            processing_time = time.time() - start_time
            self.logger.info(f"Quick analysis completed in {processing_time:.1f} seconds")
            
            return {
                'success': True,
                'player_info': player_data['player_info'],
                'matches': player_data['matches'][:5],  # Return fewer matches
                'stats': basic_stats,
                'ml_analysis': {
                    'playstyle': playstyle,
                    'strengths': strengths,
                    'improvement_areas': improvements,
                    'player_profile': player_profile
                },
                'ai_insights': {
                    'season_summary': f"Quick analysis of {match_count} matches shows {playstyle['playstyle_name']} playstyle with {basic_stats['win_rate']:.1%} win rate."
                },
                'processing_time': f"{processing_time:.1f}s",
                'analysis_type': 'quick'
            }
            
        except Exception as e:
            self.logger.error(f"Quick analysis failed: {str(e)}")
            return {'success': False, 'error': f"Analysis failed: {str(e)}"}
    
    async def test_sagemaker_integration(self) -> Dict:
        """
        🧪 Test the SageMaker ward detection endpoint
        """
        self.logger.info("🧪 Testing SageMaker Integration...")
        
        try:
            # Test frame data
            test_frame = [[255, 128, 64] for _ in range(100)]
            
            # Test SageMaker detection
            result = self.ml_engine.detect_wards_sagemaker(test_frame, 0)
            
            return {
                "success": True,
                "endpoint_status": "operational" if not result.get('demo_mode', True) else "demo_fallback",
                "endpoint_name": result.get('endpoint_name', 'unknown'),
                "detection_result": result,
                "message": "SageMaker endpoint working!" if not result.get('demo_mode', True) else "Using demo mode fallback"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "SageMaker test failed"
            }
    
    async def analyze_vod_with_sagemaker(self, video_path: str, precision: str = "maximum") -> Dict:
        """
        🎬 Analyze VOD using real SageMaker ward detection
        """
        self.logger.info(f"🎬 VOD Analysis with SageMaker: {video_path}")
        
        try:
            # Use ML engine's enhanced VOD analysis with SageMaker
            result = self.ml_engine.analyze_vod_with_precision(video_path, precision)
            
            return {
                "success": True,
                "vod_analysis": result,
                "sagemaker_enabled": self.ml_engine.use_sagemaker,
                "endpoint": self.ml_engine.ward_detector_endpoint
            }
            
        except Exception as e:
            self.logger.error(f"VOD analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

if __name__ == "__main__":
    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        print("Error: RIOT_API_KEY not found! Check your .env and setup.")
        import sys
        sys.exit(1)
    import sys
    if len(sys.argv) < 3:
        print("Usage: python riftrewind_app.py <GAME_NAME> <TAG_LINE>")
        sys.exit(1)
    game_name = sys.argv[1]
    tag_line = sys.argv[2]

    async def test_analysis():
        app = RiftRewindApp(api_key)
        result = await app.analyze_player_complete(game_name, tag_line, 20)
        if result.get('success'):
            print(f"\n🎮 RIFT REWIND ANALYSIS - {game_name}#{tag_line}")
            print("=" * 60)
            stats = result['summary_stats']
            print(f"Games Analyzed: {stats['total_games']}")
            print(f"Win Rate: {stats['win_rate']:.1%}")
            print(f"Average KDA: {stats['avg_kda']:.2f}")
            print(f"Primary Role: {stats['primary_role']}")
            playstyle = result['ml_analysis']['playstyle']
            print(f"\n🎨 Playstyle: {playstyle['playstyle_name']}")
            print(f"Description: {playstyle['description']}")
            print(f"\n📊 Key Strengths:")
            for strength in result['ml_analysis']['strengths']:
                print(f"  ✓ {strength}")
            print(f"\n⚡ Improvement Areas:")
            for area in result['ml_analysis']['improvement_areas']:
                print(f"  → {area}")
            print(f"\n🤖 AI Season Summary:")
            print(result['ai_insights']['season_summary'])
            print(f"\n⏱️ Processing Time: {result['processing_time']} seconds")
        else:
            print(f"❌ Analysis failed: {result.get('error')}")
    
    # Test SageMaker integration
    async def test_sagemaker():
        app = RiftRewindApp("PLACEHOLDER_KEY")
        test_result = await app.test_sagemaker_integration()
        
        print("\n🧪 SageMaker Test Results:")
        print(f"Status: {test_result.get('message', 'Unknown')}")
        if test_result['success']:
            print(f"Endpoint: {test_result.get('endpoint_name', 'Unknown')}")
            detection = test_result.get('detection_result', {})
            print(f"Wards Detected: {detection.get('total_wards', 0)}")
            print(f"Processing Time: {detection.get('processing_time_ms', 0)}ms")
            print(f"Production Mode: {not detection.get('demo_mode', True)}")
        else:
            print(f"Error: {test_result.get('error', 'Unknown error')}")
    
    print("🚀 Testing SageMaker Integration...")
    asyncio.run(test_sagemaker())
    
    # Uncomment to run full player analysis
    # asyncio.run(test_analysis())
