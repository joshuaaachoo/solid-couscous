#!/usr/bin/env python3
"""
Test script to validate timeline data processing for death heatmaps
"""

import asyncio
import os
from dotenv import load_dotenv
from riot_api_client import RiotApiClient
from visualization import validate_timeline_data_availability, debug_match_structure

load_dotenv()

async def test_timeline_data():
    """Test timeline data availability and structure"""
    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        print("❌ RIOT_API_KEY not found in environment variables")
        return
    
    # Test with a sample player - using a more common example
    game_name = "radpoles"  # You can change this
    tag_line = "chill"            # You can change this
    
    print(f"🔍 Testing timeline data for {game_name}#{tag_line}")
    print("=" * 50)
    
    try:
        async with RiotApiClient(api_key, "americas") as riot_client:
            print("📡 Fetching player data with timeline...")
            player_data = await riot_client.get_comprehensive_player_data(
                game_name, tag_line, 5  # Just 5 matches for testing
            )
            
            if 'error' in player_data:
                print(f"❌ Error fetching player data: {player_data['error']}")
                return
            
            if not player_data.get('matches'):
                print("❌ No matches found")
                return
            
            matches = player_data['matches']
            player_puuid = player_data['player_info']['puuid']
            
            print(f"✅ Found {len(matches)} matches")
            print(f"📋 Player PUUID: {player_puuid}")
            
            # Debug first match structure
            if matches:
                print("\n🔍 Debugging first match structure:")
                debug_match_structure(matches[0], 0)
            
            # Validate timeline data
            print("\n📊 Validating timeline data availability:")
            timeline_summary = validate_timeline_data_availability(
                matches, player_puuid, debug=True
            )
            
            print("\n📈 Timeline Summary:")
            for key, value in timeline_summary.items():
                print(f"  {key}: {value}")
            
            # Test death heatmap generation
            print("\n🎯 Testing death heatmap generation...")
            from visualization import create_combined_death_heatmap
            heatmap = create_combined_death_heatmap(matches, player_puuid)
            
            if heatmap:
                print("✅ Death heatmap generated successfully!")
            else:
                print("❌ Failed to generate death heatmap")
                
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_timeline_data())
