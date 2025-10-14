"""
RiftRewind - Streamlit App for AWS App Runner
Enhanced with DynamoDB caching and AWS AI services
"""
import streamlit as st
import os
import sys
import asyncio
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import boto3
from botocore.exceptions import ClientError

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from api.riot_api_client import RiotApiClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
RIOT_API_KEY = os.getenv('RIOT_API_KEY')

# DynamoDB Tables
MATCHES_TABLE = os.getenv('MATCHES_TABLE', 'riftrewind-matches')
SEARCHES_TABLE = os.getenv('SEARCHES_TABLE', 'riftrewind-searches')

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
matches_table = dynamodb.Table(MATCHES_TABLE)
searches_table = dynamodb.Table(SEARCHES_TABLE)

class DynamoDBCache:
    """DynamoDB caching layer for match data and searches"""
    
    @staticmethod
    def get_match(match_id: str) -> Optional[Dict]:
        """Get match from cache"""
        try:
            response = matches_table.get_item(Key={'match_id': match_id})
            if 'Item' in response:
                item = response['Item']
                # Check if expired
                if int(item.get('ttl', 0)) > int(time.time()):
                    logger.info(f"✅ Cache HIT: {match_id}")
                    return json.loads(item['match_data'])
                logger.info(f"⏰ Cache EXPIRED: {match_id}")
            else:
                logger.info(f"❌ Cache MISS: {match_id}")
            return None
        except ClientError as e:
            logger.error(f"DynamoDB get_match error: {e}")
            return None
    
    @staticmethod
    def put_match(match_id: str, match_data: Dict):
        """Store match in cache (7 days TTL)"""
        try:
            ttl = int(time.time()) + (7 * 24 * 60 * 60)  # 7 days
            matches_table.put_item(
                Item={
                    'match_id': match_id,
                    'match_data': json.dumps(match_data),
                    'ttl': ttl,
                    'cached_at': datetime.utcnow().isoformat()
                }
            )
            logger.info(f"💾 Cached match: {match_id}")
        except ClientError as e:
            logger.error(f"DynamoDB put_match error: {e}")
    
    @staticmethod
    def get_search(search_key: str) -> Optional[Dict]:
        """Get search result from cache"""
        try:
            response = searches_table.get_item(Key={'search_key': search_key})
            if 'Item' in response:
                item = response['Item']
                # Check if expired
                if int(item.get('ttl', 0)) > int(time.time()):
                    logger.info(f"✅ Search Cache HIT: {search_key}")
                    return json.loads(item['result_data'])
                logger.info(f"⏰ Search Cache EXPIRED: {search_key}")
            else:
                logger.info(f"❌ Search Cache MISS: {search_key}")
            return None
        except ClientError as e:
            logger.error(f"DynamoDB get_search error: {e}")
            return None
    
    @staticmethod
    def put_search(search_key: str, result_data: Dict):
        """Store search result (24 hours TTL)"""
        try:
            ttl = int(time.time()) + (24 * 60 * 60)  # 24 hours
            searches_table.put_item(
                Item={
                    'search_key': search_key,
                    'result_data': json.dumps(result_data),
                    'ttl': ttl,
                    'cached_at': datetime.utcnow().isoformat()
                }
            )
            logger.info(f"💾 Cached search: {search_key}")
        except ClientError as e:
            logger.error(f"DynamoDB put_search error: {e}")


class PlayerConnectionFinder:
    """Find connection between two players using cached BFS search"""
    
    def __init__(self, riot_api_key: str):
        self.riot_api_key = riot_api_key
        self.cache = DynamoDBCache()
    
    async def find_connection(self, player1: Tuple[str, str], player2: Tuple[str, str], 
                             max_depth: int = 3) -> Dict:
        """
        Find shortest path between two players
        Returns: {
            'found': bool,
            'path': List[dict] with player info,
            'matches': List[str] match IDs,
            'stats': search statistics
        }
        """
        # Create cache key
        search_key = f"{player1[0]}#{player1[1]}__{player2[0]}#{player2[1]}__{max_depth}"
        
        # Check cache first
        cached_result = self.cache.get_search(search_key)
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        try:
            async with RiotApiClient(self.riot_api_key, "americas") as riot_client:
                # Get PUUIDs
                p1_data = await riot_client.get_account_by_riot_id(player1[0], player1[1])
                p2_data = await riot_client.get_account_by_riot_id(player2[0], player2[1])
                
                if not p1_data or not p2_data:
                    return {'found': False, 'error': 'Player not found'}
                
                p1_puuid = p1_data['puuid']
                p2_puuid = p2_data['puuid']
                
                # Bidirectional BFS
                result = await self._bidirectional_bfs(
                    riot_client, p1_puuid, p2_puuid, player1, player2, max_depth
                )
                
                result['search_time'] = time.time() - start_time
                
                # Cache result
                if result.get('found'):
                    self.cache.put_search(search_key, result)
                
                return result
                
        except Exception as e:
            logger.error(f"Connection search failed: {e}")
            return {'found': False, 'error': str(e)}
    
    async def _bidirectional_bfs(self, riot_client: RiotApiClient, 
                                 p1_puuid: str, p2_puuid: str,
                                 p1_name: Tuple[str, str], p2_name: Tuple[str, str],
                                 max_depth: int) -> Dict:
        """Bidirectional BFS with DynamoDB caching"""
        
        # Initialize queues
        queue1 = [(p1_puuid, [{'puuid': p1_puuid, 'name': f"{p1_name[0]}#{p1_name[1]}"}], [])]
        queue2 = [(p2_puuid, [{'puuid': p2_puuid, 'name': f"{p2_name[0]}#{p2_name[1]}"}], [])]
        
        visited1 = {p1_puuid: ([], [])}
        visited2 = {p2_puuid: ([], [])}
        
        stats = {
            'players_explored': 0,
            'matches_checked': 0,
            'cache_hits': 0,
            'api_calls': 0
        }
        
        for depth in range(max_depth):
            # Search from player 1
            if queue1:
                new_queue1 = []
                for puuid, path, match_path in queue1:
                    # Get matches
                    match_ids = await self._get_player_matches(riot_client, puuid, stats)
                    
                    for match_id in match_ids[:20]:  # Limit matches per player
                        # Get match details with caching
                        match_data = await self._get_match_cached(riot_client, match_id, stats)
                        
                        if not match_data:
                            continue
                        
                        # Get other players
                        for participant in match_data.get('info', {}).get('participants', []):
                            other_puuid = participant['puuid']
                            
                            if other_puuid == puuid:
                                continue
                            
                            # Check if we found connection
                            if other_puuid in visited2:
                                # Found connection!
                                path2, match_path2 = visited2[other_puuid]
                                full_path = path + [{'puuid': other_puuid, 'name': participant['riotIdGameName']}] + path2[::-1]
                                full_matches = match_path + [match_id] + match_path2[::-1]
                                
                                return {
                                    'found': True,
                                    'path': full_path,
                                    'matches': full_matches,
                                    'depth': depth + 1,
                                    'stats': stats
                                }
                            
                            if other_puuid not in visited1:
                                visited1[other_puuid] = (
                                    path + [{'puuid': other_puuid, 'name': participant['riotIdGameName']}],
                                    match_path + [match_id]
                                )
                                new_queue1.append((
                                    other_puuid,
                                    path + [{'puuid': other_puuid, 'name': participant['riotIdGameName']}],
                                    match_path + [match_id]
                                ))
                                stats['players_explored'] += 1
                
                queue1 = new_queue1[:50]  # Limit queue size
            
            # Search from player 2 (similar logic)
            if queue2:
                new_queue2 = []
                for puuid, path, match_path in queue2:
                    match_ids = await self._get_player_matches(riot_client, puuid, stats)
                    
                    for match_id in match_ids[:20]:
                        match_data = await self._get_match_cached(riot_client, match_id, stats)
                        
                        if not match_data:
                            continue
                        
                        for participant in match_data.get('info', {}).get('participants', []):
                            other_puuid = participant['puuid']
                            
                            if other_puuid == puuid:
                                continue
                            
                            if other_puuid in visited1:
                                path1, match_path1 = visited1[other_puuid]
                                full_path = path1 + [{'puuid': other_puuid, 'name': participant['riotIdGameName']}] + path[::-1]
                                full_matches = match_path1 + [match_id] + match_path[::-1]
                                
                                return {
                                    'found': True,
                                    'path': full_path,
                                    'matches': full_matches,
                                    'depth': depth + 1,
                                    'stats': stats
                                }
                            
                            if other_puuid not in visited2:
                                visited2[other_puuid] = (
                                    path + [{'puuid': other_puuid, 'name': participant['riotIdGameName']}],
                                    match_path + [match_id]
                                )
                                new_queue2.append((
                                    other_puuid,
                                    path + [{'puuid': other_puuid, 'name': participant['riotIdGameName']}],
                                    match_path + [match_id]
                                ))
                                stats['players_explored'] += 1
                
                queue2 = new_queue2[:50]
        
        return {
            'found': False,
            'message': f'No connection found within {max_depth} degrees',
            'stats': stats
        }
    
    async def _get_player_matches(self, riot_client: RiotApiClient, puuid: str, stats: Dict) -> List[str]:
        """Get recent match IDs for player"""
        try:
            stats['api_calls'] += 1
            matches = await riot_client.get_match_history(puuid, count=20)
            return matches
        except Exception as e:
            logger.error(f"Failed to get matches for {puuid}: {e}")
            return []
    
    async def _get_match_cached(self, riot_client: RiotApiClient, match_id: str, stats: Dict) -> Optional[Dict]:
        """Get match data with DynamoDB caching"""
        # Check cache
        cached = self.cache.get_match(match_id)
        if cached:
            stats['cache_hits'] += 1
            return cached
        
        # Fetch from API
        try:
            stats['api_calls'] += 1
            stats['matches_checked'] += 1
            match_data = await riot_client.get_match_details(match_id)
            
            # Cache it
            if match_data:
                self.cache.put_match(match_id, match_data)
            
            return match_data
        except Exception as e:
            logger.error(f"Failed to get match {match_id}: {e}")
            return None


# Streamlit UI
def main():
    st.set_page_config(
        page_title="RiftRewind - Player Connection Finder",
        page_icon="🎮",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #151933 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #5b8fff 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 8px;
    }
    .connection-path {
        background: rgba(91, 143, 255, 0.1);
        border-left: 4px solid #5b8fff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎮 RiftRewind - Player Connection Finder")
    st.markdown("### Find how League of Legends players are connected through matches!")
    
    # Check AWS connection
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Check DynamoDB
        try:
            matches_table.table_status
            st.success("✅ DynamoDB Connected")
        except Exception as e:
            st.error(f"❌ DynamoDB Error: {str(e)}")
        
        # Check Riot API
        if RIOT_API_KEY:
            st.success("✅ Riot API Key Found")
        else:
            st.error("❌ Missing Riot API Key")
        
        st.markdown("---")
        st.markdown("**Powered by AWS:**")
        st.markdown("- 📊 DynamoDB (Caching)")
        st.markdown("- 🚀 App Runner (Hosting)")
        st.markdown("- 🤖 Bedrock (Coming Soon)")
        st.markdown("- 🔍 Comprehend (Coming Soon)")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Player 1")
        p1_name = st.text_input("Game Name", key="p1_name", placeholder="Apricity")
        p1_tag = st.text_input("Tag Line", key="p1_tag", placeholder="Meoww")
    
    with col2:
        st.markdown("#### Player 2")
        p2_name = st.text_input("Game Name", key="p2_name", placeholder="Pepperonji")
        p2_tag = st.text_input("Tag Line", key="p2_tag", placeholder="5543")
    
    max_depth = st.slider("Max Degrees of Separation", 1, 4, 3)
    
    if st.button("🔍 Find Connection", type="primary"):
        if not all([p1_name, p1_tag, p2_name, p2_tag]):
            st.error("Please fill in all player information!")
            return
        
        if not RIOT_API_KEY:
            st.error("Riot API key not configured!")
            return
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Searching for connection...")
        progress_bar.progress(10)
        
        # Run search
        finder = PlayerConnectionFinder(RIOT_API_KEY)
        
        async def run_search():
            return await finder.find_connection(
                (p1_name, p1_tag),
                (p2_name, p2_tag),
                max_depth
            )
        
        result = asyncio.run(run_search())
        
        progress_bar.progress(100)
        status_text.text("✅ Search complete!")
        
        # Display results
        st.markdown("---")
        
        if result.get('found'):
            st.success(f"🎉 Connection Found! ({result.get('depth', 0)} degrees of separation)")
            
            # Display path
            st.markdown("### 🌳 Connection Path:")
            path = result.get('path', [])
            matches = result.get('matches', [])
            
            for i, player in enumerate(path):
                st.markdown(f"""
                <div class="connection-path">
                    <strong>Player {i+1}:</strong> {player.get('name', 'Unknown')}
                </div>
                """, unsafe_allow_html=True)
                
                if i < len(matches):
                    st.caption(f"⚔️ Played together in match: `{matches[i]}`")
            
            # Display stats
            st.markdown("### 📊 Search Statistics:")
            stats = result.get('stats', {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Players Explored", stats.get('players_explored', 0))
            with col2:
                st.metric("Matches Checked", stats.get('matches_checked', 0))
            with col3:
                st.metric("Cache Hits", stats.get('cache_hits', 0))
            with col4:
                st.metric("API Calls", stats.get('api_calls', 0))
            
            st.info(f"⏱️ Search completed in {result.get('search_time', 0):.2f} seconds")
            
        else:
            st.warning(f"😔 No connection found within {max_depth} degrees")
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            
            # Show stats anyway
            if 'stats' in result:
                st.markdown("### 📊 Search Statistics:")
                stats = result['stats']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Players Explored", stats.get('players_explored', 0))
                with col2:
                    st.metric("Matches Checked", stats.get('matches_checked', 0))
                with col3:
                    st.metric("Cache Hits", stats.get('cache_hits', 0))


if __name__ == "__main__":
    main()
