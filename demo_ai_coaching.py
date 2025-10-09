"""
Demo: AI Coaching System with Riot API Integration
Shows how Bedrock AI generates personalized coaching from match data
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.riot_api_client import RiotApiClient
from ai_coach.bedrock_insights import BedrockInsightsGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def demo_ai_coaching():
    """Demonstrate AI coaching with real Riot API data"""
    
    # Load environment
    load_dotenv()
    riot_api_key = os.getenv('RIOT_API_KEY')
    
    if not riot_api_key:
        logger.error("❌ RIOT_API_KEY not found in .env file")
        return
    
    # Initialize services
    logger.info("🚀 Initializing AI Coaching Demo...")
    bedrock = BedrockInsightsGenerator()
    
    # Fetch recent match data
    logger.info("\n📊 Fetching recent match data from Riot API...")
    
    async with RiotApiClient(riot_api_key, region="americas") as client:
        # Get player account
        player_name = "radpoles"
        tag_line = "chill"
        
        logger.info(f"🔍 Looking up player: {player_name}#{tag_line}")
        
        try:
            account = await client.get_player_info(player_name, tag_line)
            if not account:
                logger.error(f"❌ Player {player_name}#{tag_line} not found")
                return
            
            puuid = account.puuid
            logger.info(f"✅ Found player: {account.game_name}#{account.tag_line}")
            
            # Get recent matches (ranked solo/duo only - queue 420)
            logger.info(f"\n🎮 Fetching recent ranked solo/duo matches...")
            match_ids = await client.get_match_history(puuid, count=5)
            
            if not match_ids:
                logger.error("❌ No matches found")
                return
            
            logger.info(f"✅ Found {len(match_ids)} recent matches")
            
            # Fetch all matches first for accurate stats
            logger.info(f"\n� Fetching match history for accurate analysis...")
            
            all_matches = []
            for match_id in match_ids[:5]:  # Get last 5 matches
                match = await client.get_match_details(match_id)
                if match:
                    for participant in match.get('info', {}).get('participants', []):
                        if participant.get('puuid') == puuid:
                            all_matches.append(participant)
                            break
            
            logger.info(f"✅ Collected {len(all_matches)} matches for analysis")
            
            # Display what matches were fetched
            logger.info("\n" + "="*60)
            logger.info("RANKED SOLO/DUO MATCHES (Last 5)")
            logger.info("="*60)
            for i, match in enumerate(all_matches, 1):
                result = "✅ WIN" if match.get('win') else "❌ LOSS"
                kda = (match.get('kills', 0) + match.get('assists', 0)) / max(match.get('deaths', 1), 1)
                logger.info(f"{i}. {match.get('championName')} - {match.get('kills')}/{match.get('deaths')}/{match.get('assists')} ({kda:.1f} KDA) - {result}")
            logger.info("="*60)
            
            # Get most recent match for detailed display
            logger.info(f"\n📈 Analyzing most recent match...")
            player_stats = all_matches[0] if all_matches else None
            
            if not player_stats:
                logger.error("❌ Could not find player in match data")
                return
            
            # Display match summary
            logger.info("\n" + "="*60)
            logger.info("MATCH SUMMARY")
            logger.info("="*60)
            logger.info(f"Champion: {player_stats.get('championName')}")
            logger.info(f"KDA: {player_stats.get('kills')}/{player_stats.get('deaths')}/{player_stats.get('assists')}")
            logger.info(f"CS: {player_stats.get('totalMinionsKilled')}")
            logger.info(f"Vision Score: {player_stats.get('visionScore')}")
            logger.info(f"Result: {'Victory' if player_stats.get('win') else 'Defeat'}")
            logger.info(f"Duration: {player_stats.get('gameDuration', 0) // 60} minutes")
            logger.info("="*60)
            
            # Generate AI coaching insights with ALL matches for accurate stats
            logger.info("\n🤖 Generating AI Coaching Insights with Claude 3...")
            logger.info("(This may take a few seconds...)")
            
            # Prepare match data with ALL matches
            player_data = {
                'player_info': {
                    'name': f"{account.game_name}#{account.tag_line}",
                    'summoner_level': account.summoner_level
                },
                'matches': all_matches  # Use all matches here
            }
            
            ml_insights = {
                'playstyle': {'playstyle_name': 'Performance Analysis', 'description': 'AI-powered insights'},
                'coaching_insights': [
                    f"Recent performance across {len(all_matches)} games",
                    f"Win rate: {sum(1 for m in all_matches if m.get('win')) / len(all_matches):.0%}",
                    f"Main champion: {player_stats.get('championName')}"
                ]
            }
            
            coaching_result = bedrock.generate_season_summary(player_data, ml_insights)
            
            # Display coaching insights
            logger.info("\n" + "="*60)
            logger.info("AI COACHING INSIGHTS")
            logger.info("="*60)
            logger.info(f"Generated at: {datetime.now().isoformat()}")
            logger.info("\n" + "-"*60)
            logger.info(coaching_result)
            logger.info("="*60)
            
            # Generate champion mastery analysis
            logger.info("\n🎮 Analyzing Champion Pool...")
            
            comprehensive_player_data = {
                'player_info': {
                    'name': f"{account.game_name}#{account.tag_line}",
                    'summoner_level': account.summoner_level
                },
                'matches': all_matches
            }
            
            comprehensive = bedrock.generate_champion_mastery_insights(comprehensive_player_data)
            
            logger.info("\n" + "="*60)
            logger.info("CHAMPION MASTERY ANALYSIS")
            logger.info("="*60)
            logger.info(f"Matches Analyzed: {len(all_matches)}")
            logger.info("\n" + "-"*60)
            logger.info(comprehensive)
            logger.info("="*60)
            
            # Generate improvement roadmap
            logger.info("\n🎯 Creating Personalized Improvement Roadmap...")
            
            roadmap = bedrock.generate_improvement_roadmap(comprehensive_player_data, ml_insights)
            
            logger.info("\n" + "="*60)
            logger.info("30-DAY IMPROVEMENT ROADMAP")
            logger.info("="*60)
            logger.info("\n" + "-"*60)
            logger.info(roadmap)
            logger.info("="*60)
            
            logger.info("\n✅ AI Coaching Demo Complete!")
            
        except Exception as e:
            logger.error(f"❌ Error during demo: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("RIFT REWIND - AI COACHING DEMO")
    logger.info("="*60)
    logger.info("This demo shows how AWS Bedrock AI (Claude 3) generates")
    logger.info("personalized coaching insights from Riot API match data.")
    logger.info("")
    
    asyncio.run(demo_ai_coaching())
