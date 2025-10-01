from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
from riot_api_client import RiotApiClient
from riftrewind_ml_engine import RiftRewindMLEngine
from bedrock_insights import BedrockInsightsGenerator
from enhanced_coaching import EnhancedCoachingAnalyzer
from progressive_tracker import ProgressiveTracker

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
        self.ml_engine = RiftRewindMLEngine()
        self.insights_generator = BedrockInsightsGenerator(aws_region)
        self.coaching_analyzer = EnhancedCoachingAnalyzer()
        self.tracker = ProgressiveTracker()
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
            
            # Step 6: Enhanced AI Insights with Coaching
            self.logger.info("Generating enhanced coaching insights...")
            enhanced_insights = self.insights_generator.generate_enhanced_coaching_insights(
                player_data, ml_insights, coaching_analysis
            )

            # Step 7: Compile final report
            processing_time = time.time() - start_time
            report = {
                'success': True,
                'player_info': player_data['player_info'],
                'raw_matches': player_data['matches'],  #   Include raw match data for death heatmap
                'summary_stats': {
                    'total_games': len(player_data['matches']),
                    'win_rate': player_profile.get('win_rate', 0),
                    'avg_kda': player_profile.get('avg_kda', 1.0),
                    'avg_cs_per_min': player_profile.get('avg_cs_per_min', 5.0),
                    'primary_role': player_profile.get('primary_role', 'UTILITY'),
                    'champion_diversity': player_profile.get('champion_diversity', 0)
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
                'processing_time': round(processing_time, 2),
                'generated_at': datetime.now().isoformat()
            }
            self.logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            return report

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            return {'success': False, 'error': str(e)}

    def _determine_playstyle_simple(self, player_profile: Dict) -> Dict:
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
    asyncio.run(test_analysis())
