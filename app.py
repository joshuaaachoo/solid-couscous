"""
RiftRewind AI Coach - Streamlit Interface
Get personalized League of Legends coaching powered by AWS Bedrock
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from api.riot_api_client import RiotApiClient
from ai_coach.bedrock_insights import BedrockInsightsGenerator
from core.visualization import create_map_overlay_heatmap
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RiftRewind AI Coach",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Custom CSS for premium gaming aesthetic with video background
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Rajdhani', 'Inter', sans-serif !important;
    }
    
    /* Headings use bold Rajdhani for LoL feel */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    
    /* Hero section with video background */
    .hero-section {
        position: relative;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        margin-top: -5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin-bottom: 0;
    }
    
    .hero-video {
        position: absolute;
        top: 50%;
        left: 50%;
        min-width: 100%;
        min-height: 100%;
        width: 100%;
        height: 100%;
        transform: translate(-50%, -50%) scale(1.5);
        z-index: 0;
        opacity: 0.6;
        pointer-events: none;
    }
    
    .hero-video iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border: none;
    }
    
    .hero-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.2) 0%, rgba(10, 14, 39, 0.6) 60%, rgba(10, 14, 39, 1) 95%);
        z-index: 1;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
        text-align: center;
        color: white;
        padding: 2rem;
    }
    
    .scroll-indicator {
        position: absolute;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        z-index: 2;
        animation: bounce 2s infinite;
        color: rgba(255, 255, 255, 0.6);
        font-size: 2rem;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateX(-50%) translateY(0); }
        40% { transform: translateX(-50%) translateY(-10px); }
        60% { transform: translateX(-50%) translateY(-5px); }
    }
    
    /* Content section with dark background */
    .content-section {
        background: #0a0e27;
        padding-top: 0;
        margin-top: 0;
        min-height: 100vh;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        padding-left: calc(50vw - 50%);
        padding-right: calc(50vw - 50%);
        position: relative;
    }
    
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Remove sidebar collapse button */
    button[kind="header"] {
        display: none !important;
    }
    
    /* Hide bottom toolbar */
    footer {
        display: none !important;
    }
    
    .stApp > footer {
        display: none !important;
    }
    
    /* Remove padding that creates white bar */
    .main .block-container {
        padding-bottom: 0 !important;
        max-width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Ensure full width */
    .main {
        overflow-x: hidden;
    }
    
    .stApp {
        overflow-x: hidden;
    }
    
    /* Dark theme background with subtle pattern */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        background-attachment: fixed;
    }
    
    /* Main title with animated gradient - LoL style */
    .main-title {
        text-align: center;
        font-size: 5.5rem;
        font-weight: 900;
        font-family: 'Rajdhani', sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradientShift 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 0 0 80px rgba(102, 126, 234, 0.5);
        filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.4));
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Subtitle with glow effect */
    .subtitle {
        text-align: center;
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    
    /* Glowing card styling */
    .stats-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3);
    }
    
    /* Premium champion cards */
    .champion-card {
        text-align: center;
        padding: 0.75rem;
        border-radius: 12px;
        background: rgba(30, 30, 46, 0.6);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .champion-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent 0%, rgba(255, 255, 255, 0.05) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .champion-card:hover::before {
        opacity: 1;
    }
    
    .champion-card:hover {
        transform: scale(1.05);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .champion-card.win {
        border-color: #4caf50;
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(76, 175, 80, 0.05) 100%);
        box-shadow: 0 4px 16px rgba(76, 175, 80, 0.2);
    }
    
    .champion-card.loss {
        border-color: #f44336;
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.15) 0%, rgba(244, 67, 54, 0.05) 100%);
        box-shadow: 0 4px 16px rgba(244, 67, 54, 0.2);
    }
    
    /* Section headers with glow */
    .section-header {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        letter-spacing: -1px;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, transparent 100%);
    }
    
    /* Champion icon with ring effect */
    .champion-icon {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        border: 3px solid #ffd700;
        margin: 0 auto 0.5rem auto;
        display: block;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .champion-icon:hover {
        transform: scale(1.1) rotate(5deg);
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.6);
    }
    
    /* Footer with premium styling */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling with LoL aesthetic */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4), inset 0 -2px 0 rgba(0, 0, 0, 0.2) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6), inset 0 -2px 0 rgba(0, 0, 0, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(30, 30, 46, 0.6) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero section with Ekko rewind video background
st.markdown("""
<div class="hero-section">
    <div id="yt-player" class="hero-video"></div>
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <div class="main-title">REWIND</div>
        <div class="subtitle">Elite AI Coaching • Powered by AWS Bedrock Claude 3 Sonnet</div>
        <p style="color: rgba(255, 255, 255, 0.8); font-weight: 600; font-size: 1rem; margin-top: 1rem;">
            Ranked Solo/Duo Analysis • Advanced ML Insights • Player Network Discovery
        </p>
    </div>
    <div class="scroll-indicator">↓</div>
</div>
<div class="content-section">
""", unsafe_allow_html=True)

# Inject YouTube API script for video looping
import streamlit.components.v1 as components
components.html("""
<script src="https://www.youtube.com/iframe_api"></script>
<script>
var player;
function onYouTubeIframeAPIReady() {
    var playerDiv = window.parent.document.getElementById('yt-player');
    if (playerDiv) {
        player = new YT.Player(playerDiv, {
            videoId: 'OP8qCVwdly0',
            playerVars: {
                autoplay: 1,
                mute: 1,
                controls: 0,
                showinfo: 0,
                modestbranding: 1,
                rel: 0,
                start: 20,
                end: 30,
                playsinline: 1,
                iv_load_policy: 3,
                disablekb: 1,
                fs: 0
            },
            events: {
                'onReady': onPlayerReady,
                'onStateChange': onPlayerStateChange
            }
        });
    }
}

function onPlayerReady(event) {
    event.target.playVideo();
    event.target.setVolume(0);
}

function onPlayerStateChange(event) {
    if (event.data == YT.PlayerState.ENDED || event.data == YT.PlayerState.PAUSED) {
        player.seekTo(20);
        player.playVideo();
    }
}

// Poll for player position to restart at 30 seconds
setInterval(function() {
    if (player && player.getCurrentTime) {
        var currentTime = player.getCurrentTime();
        if (currentTime >= 30) {
            player.seekTo(20);
        }
    }
}, 100);
</script>
""", height=0)

# Main input section (moved from sidebar)
st.markdown('<div style="max-width: 800px; margin: 0 auto; padding: 1rem 2rem;">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### PLAYER LOOKUP")
    game_name = st.text_input("Summoner Name", value="radpoles", placeholder="Enter summoner name")
    tag_line = st.text_input("Tag Line", value="chill", placeholder="Enter tag line")
    region = st.selectbox("Region", ["americas", "europe", "asia"], index=0)
    match_count = st.slider("Number of Matches", min_value=1, max_value=100, value=5, step=1)
    st.caption(" ")  # Spacer to align with col2's caption
    analyze_button = st.button("ANALYZE MY GAMES", type="primary", use_container_width=True)

with col2:
    st.markdown("### SEVEN DEGREES")
    target_game_name = st.text_input("Target Player Name", placeholder="Enter target name")
    target_tag_line = st.text_input("Target Tag", placeholder="Enter target tag")
    matches_to_check = st.slider("Matches Per Player", min_value=1, max_value=10, value=3, step=1,
                                 help="How many recent matches to check for each player. More matches = better chance to find connection but slower.")
    st.caption("Bidirectional search: checks matches from both players simultaneously")
    separation_button = st.button("FIND CONNECTION", type="secondary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Main content
def get_champion_icon_url(champion_name: str) -> str:
    """Get Data Dragon champion icon URL"""
    # Use latest patch - you can update this periodically
    return f"https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/{champion_name}.png"

async def find_player_connection(source_name: str, source_tag: str, target_name: str, target_tag: str, 
                                 region: str, max_depth: int = 3, matches_per_player: int = 2,
                                 progress_callback=None):
    """
    Find the shortest connection path between two players through shared matches.
    Uses bidirectional DFS (searching from both players) with parallel API calls.
    
    Returns: dict with 'path', 'degree', and 'matches' if found, or 'error' if not found
    """
    import time
    
    start_time = time.time()
    
    riot_api_key = os.getenv('RIOT_API_KEY')
    if not riot_api_key:
        return {'error': 'RIOT_API_KEY not found'}
    
    async with RiotApiClient(riot_api_key, region=region) as client:
        # Get source and target player info
        source_account = await client.get_player_info(source_name, source_tag)
        target_account = await client.get_player_info(target_name, target_tag)
        
        if not source_account:
            return {'error': f'Source player {source_name}#{source_tag} not found'}
        if not target_account:
            return {'error': f'Target player {target_name}#{target_tag} not found'}
        
        # Check if they're the same player
        if source_account.puuid == target_account.puuid:
            return {'error': 'Source and target are the same player!'}
        
        # Bidirectional search setup - search from both ends using QUEUES for BFS
        source_queue = [(source_account.puuid, [source_account.puuid], [], 0)]  # (puuid, path, matches, depth)
        target_queue = [(target_account.puuid, [target_account.puuid], [], 0)]
        
        source_visited = {source_account.puuid: ([], [])}  # puuid -> (path, matches)
        target_visited = {target_account.puuid: ([], [])}
        
        puuid_to_name = {
            source_account.puuid: f"{source_name}#{source_tag}",
            target_account.puuid: f"{target_name}#{target_tag}"
        }
        
        # Track the search tree for visualization
        search_tree = {
            'nodes': [],
            'edges': [],
            'node_ids': set()  # Fast O(1) lookup for node existence
        }
        search_tree['nodes'].append({
            'id': source_account.puuid,
            'name': f"{source_name}#{source_tag}",
            'depth': 0,
            'type': 'source'
        })
        search_tree['node_ids'].add(source_account.puuid)
        search_tree['nodes'].append({
            'id': target_account.puuid,
            'name': f"{target_name}#{target_tag}",
            'depth': 0,
            'type': 'target'
        })
        search_tree['node_ids'].add(target_account.puuid)
        
        # Initial callback to show both starting players
        if progress_callback:
            try:
                progress_callback(0, 0, 2, tree_update=search_tree)
            except:
                pass  # Ignore if browser disconnected
        
        players_checked = 0
        matches_checked = 0
        max_players_to_check = 300
        
        # Process level-by-level (all nodes at depth N before depth N+1)
        while (source_queue or target_queue) and players_checked < max_players_to_check:
            # Process entire source level
            source_level_size = len(source_queue)
            for _ in range(source_level_size):
                if not source_queue or players_checked >= max_players_to_check:
                    break
                    
                current_puuid, path, connection_matches, depth = source_queue.pop(0)
                
                if depth >= max_depth:
                    continue
                
                players_checked += 1
                
                match_ids = await client.get_match_history(current_puuid, count=matches_per_player, queue_type=None)
                
                # Skip players with too few matches (likely inactive)
                if len(match_ids) < matches_per_player // 2:
                    continue
                
                batch_ids = match_ids[:matches_per_player]
                
                match_tasks = [client.get_match_details(match_id) for match_id in batch_ids]
                match_details_list = await asyncio.gather(*match_tasks, return_exceptions=True)
                
                for match_id, match_details in zip(batch_ids, match_details_list):
                    if not match_details or isinstance(match_details, Exception):
                        continue
                    
                    matches_checked += 1
                    
                    # Get all participants in this match
                    participants = match_details.get('info', {}).get('participants', [])
                    
                    # Check each participant
                    for participant in participants:
                        neighbor_puuid = participant.get('puuid')
                        
                        if not neighbor_puuid or neighbor_puuid == current_puuid:
                            continue
                        
                        # Store player name if we don't have it
                        if neighbor_puuid not in puuid_to_name:
                            riot_id = participant.get('riotIdGameName', 'Unknown')
                            riot_tag = participant.get('riotIdTagline', '')
                            puuid_to_name[neighbor_puuid] = f"{riot_id}#{riot_tag}" if riot_tag else riot_id
                        
                        # Add to search tree (use fast set lookup)
                        if neighbor_puuid not in search_tree['node_ids']:
                            node_type = 'target' if neighbor_puuid == target_account.puuid else 'intermediate'
                            search_tree['nodes'].append({
                                'id': neighbor_puuid,
                                'name': puuid_to_name[neighbor_puuid],
                                'depth': depth + 1,
                                'type': node_type
                            })
                            search_tree['node_ids'].add(neighbor_puuid)
                            search_tree['edges'].append({
                                'from': current_puuid,
                                'to': neighbor_puuid,
                                'match_id': match_id
                            })
                            
                            # Update visualization less frequently (every 5 nodes instead of every node)
                            if progress_callback and len(search_tree['nodes']) % 5 == 0:
                                try:
                                    total_visited = len(source_visited) + len(target_visited)
                                    progress_callback(players_checked, matches_checked, total_visited, tree_update=search_tree)
                                except:
                                    pass  # Ignore if browser disconnected
                        
                        # Check if target side has visited this player - CONNECTION FOUND!
                        if neighbor_puuid in target_visited:
                            target_path, target_matches = target_visited[neighbor_puuid]
                            
                            # Combine paths: source -> neighbor <- target
                            # neighbor_puuid is already at the end of 'path', so don't add it again
                            final_path = path + target_path[::-1]
                            
                            match_info = {
                                'match_id': match_id,
                                'from': puuid_to_name[current_puuid],
                                'to': puuid_to_name[neighbor_puuid],
                                'game_mode': match_details.get('info', {}).get('gameMode', 'Unknown'),
                                'game_duration': match_details.get('info', {}).get('gameDuration', 0)
                            }
                            final_matches = connection_matches + [match_info] + target_matches[::-1]
                            
                            elapsed = time.time() - start_time
                            return {
                                'path': [puuid_to_name[puuid] for puuid in final_path],
                                'degree': len(final_path) - 1,
                                'matches': final_matches,
                                'search_tree': search_tree,
                                'stats': {
                                    'players_checked': players_checked,
                                    'matches_checked': matches_checked,
                                    'unique_players': len(source_visited) + len(target_visited),
                                    'time_taken': f"{elapsed:.1f}s"
                                }
                            }
                        
                        # Add to source stack if not visited
                        if neighbor_puuid not in source_visited and len(source_visited) < 200:
                            new_path = path + [neighbor_puuid]
                            new_matches = connection_matches + [{
                                'match_id': match_id,
                                'from': puuid_to_name[current_puuid],
                                'to': puuid_to_name[neighbor_puuid],
                                'game_mode': match_details.get('info', {}).get('gameMode', 'Unknown'),
                                'game_duration': match_details.get('info', {}).get('gameDuration', 0)
                            }]
                            source_visited[neighbor_puuid] = (new_path, new_matches)
                            source_queue.append((neighbor_puuid, new_path, new_matches, depth + 1))
            
            # Process entire target level
            target_level_size = len(target_queue)
            for _ in range(target_level_size):
                if not target_queue or players_checked >= max_players_to_check:
                    break
                    
                current_puuid, path, connection_matches, depth = target_queue.pop(0)
                
                if depth >= max_depth:
                    continue
                
                players_checked += 1
                
                match_ids = await client.get_match_history(current_puuid, count=matches_per_player, queue_type=None)
                
                # Skip players with too few matches (likely inactive)
                if len(match_ids) < matches_per_player // 2:
                    continue
                
                batch_ids = match_ids[:matches_per_player]
                
                match_tasks = [client.get_match_details(match_id) for match_id in batch_ids]
                match_details_list = await asyncio.gather(*match_tasks, return_exceptions=True)
                
                for match_id, match_details in zip(batch_ids, match_details_list):
                    if not match_details or isinstance(match_details, Exception):
                        continue
                    
                    matches_checked += 1
                    
                    # Get all participants in this match
                    participants = match_details.get('info', {}).get('participants', [])
                    
                    # Check each participant
                    for participant in participants:
                        neighbor_puuid = participant.get('puuid')
                        
                        if not neighbor_puuid or neighbor_puuid == current_puuid:
                            continue
                        
                        # Store player name if we don't have it
                        if neighbor_puuid not in puuid_to_name:
                            riot_id = participant.get('riotIdGameName', 'Unknown')
                            riot_tag = participant.get('riotIdTagline', '')
                            puuid_to_name[neighbor_puuid] = f"{riot_id}#{riot_tag}" if riot_tag else riot_id
                        
                        # Add to search tree (use fast set lookup)
                        if neighbor_puuid not in search_tree['node_ids']:
                            node_type = 'source' if neighbor_puuid == source_account.puuid else 'intermediate'
                            search_tree['nodes'].append({
                                'id': neighbor_puuid,
                                'name': puuid_to_name[neighbor_puuid],
                                'depth': depth + 1,
                                'type': node_type
                            })
                            search_tree['node_ids'].add(neighbor_puuid)
                            search_tree['edges'].append({
                                'from': current_puuid,
                                'to': neighbor_puuid,
                                'match_id': match_id
                            })
                            
                            # Update visualization less frequently (every 5 nodes instead of every node)
                            if progress_callback and len(search_tree['nodes']) % 5 == 0:
                                try:
                                    total_visited = len(source_visited) + len(target_visited)
                                    progress_callback(players_checked, matches_checked, total_visited, tree_update=search_tree)
                                except:
                                    pass  # Ignore if browser disconnected
                        
                        # Check if source side has visited this player - CONNECTION FOUND!
                        if neighbor_puuid in source_visited:
                            source_path, source_matches = source_visited[neighbor_puuid]
                            
                            # Combine paths: source -> neighbor <- target
                            # neighbor_puuid is already at the end of source_path, so don't add it again
                            final_path = source_path + path[::-1]
                            
                            match_info = {
                                'match_id': match_id,
                                'from': puuid_to_name[current_puuid],
                                'to': puuid_to_name[neighbor_puuid],
                                'game_mode': match_details.get('info', {}).get('gameMode', 'Unknown'),
                                'game_duration': match_details.get('info', {}).get('gameDuration', 0)
                            }
                            final_matches = source_matches + [match_info] + connection_matches[::-1]
                            
                            elapsed = time.time() - start_time
                            return {
                                'path': [puuid_to_name[puuid] for puuid in final_path],
                                'degree': len(final_path) - 1,
                                'matches': final_matches,
                                'search_tree': search_tree,
                                'stats': {
                                    'players_checked': players_checked,
                                    'matches_checked': matches_checked,
                                    'unique_players': len(source_visited) + len(target_visited),
                                    'time_taken': f"{elapsed:.1f}s"
                                }
                            }
                        
                        # Add to target stack if not visited
                        if neighbor_puuid not in target_visited and len(target_visited) < 200:
                            new_path = path + [neighbor_puuid]
                            new_matches = connection_matches + [{
                                'match_id': match_id,
                                'from': puuid_to_name[current_puuid],
                                'to': puuid_to_name[neighbor_puuid],
                                'game_mode': match_details.get('info', {}).get('gameMode', 'Unknown'),
                                'game_duration': match_details.get('info', {}).get('gameDuration', 0)
                            }]
                            target_visited[neighbor_puuid] = (new_path, new_matches)
                            target_queue.append((neighbor_puuid, new_path, new_matches, depth + 1))
        
        elapsed = time.time() - start_time
        total_visited = len(source_visited) + len(target_visited)
        return {'error': f'No connection found within {max_depth} degrees (checked {players_checked} players, {matches_checked} matches, {total_visited} unique in {elapsed:.1f}s). Try increasing matches per player or they may be in different regions.'}

async def analyze_player(game_name: str, tag_line: str, region: str):
    """Fetch and analyze player data"""
    
    riot_api_key = os.getenv('RIOT_API_KEY')
    if not riot_api_key:
        st.error("RIOT_API_KEY not found in environment variables")
        return
    
    # Initialize services
    with st.spinner("Initializing AI coaching system..."):
        bedrock = BedrockInsightsGenerator()
        connection_status = bedrock.get_connection_status()
        
        if connection_status['status'] == 'connected':
            st.success(connection_status['message'])
        else:
            st.warning(connection_status['message'])
    
    # Fetch player data
    async with RiotApiClient(riot_api_key, region=region) as client:
        with st.spinner(f"Looking up {game_name}#{tag_line}..."):
            account = await client.get_player_info(game_name, tag_line)
            
            if not account:
                st.error(f"Player {game_name}#{tag_line} not found")
                return
            
            st.success(f"Found player: {account.game_name}#{account.tag_line}")
        
        # Fetch matches
        with st.spinner("Fetching ranked solo/duo matches..."):
            puuid = account.puuid
            match_ids = await client.get_match_history(puuid, count=match_count)
            
            if not match_ids:
                st.error("No ranked matches found")
                return
            
            st.info(f"Found {len(match_ids)} recent ranked matches")
        
        # Get match details
        with st.spinner("Analyzing match data..."):
            all_matches = []
            timelines = []
            for match_id in match_ids:
                match = await client.get_match_details(match_id)
                timeline = await client.get_match_timeline(match_id)
                if match:
                    for participant in match.get('info', {}).get('participants', []):
                        if participant.get('puuid') == puuid:
                            all_matches.append(participant)
                            timelines.append((timeline, participant.get('participantId')))
                            break
        
        if not all_matches:
            st.error("Could not fetch match details")
            return
        
        # Display recent matches
        st.markdown('<div class="section-header">RECENT RANKED SOLO/DUO MATCHES</div>', unsafe_allow_html=True)
        num_cols = min(5, len(all_matches))
        cols = st.columns(num_cols)
        for i, match in enumerate(all_matches):
            with cols[i % num_cols]:
                result = "WIN" if match.get('win') else "LOSS"
                kda = (match.get('kills', 0) + match.get('assists', 0)) / max(match.get('deaths', 1), 1)
                
                # Get role and side
                role = match.get('teamPosition', 'UNKNOWN').title() if match.get('teamPosition') else 'Unknown'
                side = 'Blue' if match.get('teamId') == 100 else 'Red' if match.get('teamId') == 200 else ''
                
                champion_name = match.get('championName', 'Unknown')
                icon_url = get_champion_icon_url(champion_name)
                
                # Card styling based on win/loss
                card_class = "win" if match.get('win') else "loss"
                
                st.markdown(f"""
                <div class="champion-card {card_class}">
                    <img src="{icon_url}" class="champion-icon" onerror="this.style.display='none'">
                    <h4 style="margin: 0.25rem 0;">{champion_name}</h4>
                    <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{result}</p>
                    <p style="margin: 0.25rem 0; font-size: 1.1rem;"><strong>KDA:</strong> {kda:.1f}</p>
                    <p style="margin: 0.25rem 0; color: #888; font-size: 0.9rem;">{match.get('kills')}/{match.get('deaths')}/{match.get('assists')}</p>
                    {'<p style="margin: 0.25rem 0; color: #888; font-size: 0.9rem;">' + side + ' • ' + role + '</p>' if role != 'Unknown' else ''}
                </div>
                """, unsafe_allow_html=True)
        
        # Generate AI coaching
        player_data = {
            'player_info': {
                'name': f"{account.game_name}#{account.tag_line}",
                'summoner_level': account.summoner_level
            },
            'matches': all_matches
        }
        
        ml_insights = {
            'playstyle': {'playstyle_name': 'Performance Analysis', 'description': 'AI-powered insights'},
            'coaching_insights': [
                f"Recent performance across {len(all_matches)} games",
                f"Win rate: {sum(1 for m in all_matches if m.get('win')) / len(all_matches):.0%}"
            ]
        }
        
        # Season Summary
        with st.spinner("Generating AI coaching insights with Claude 3..."):
            season_summary = bedrock.generate_season_summary(player_data, ml_insights)
        
        st.markdown('<div class="section-header">PERFORMANCE ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown(season_summary)
        
        # Death Heatmap - Overall
        st.markdown('<div class="section-header">DEATH HEATMAP ANALYSIS</div>', unsafe_allow_html=True)
        with st.spinner("Generating overall death heatmap..."):
            try:
                # Prepare match data with timelines for the visualization function
                matches_with_timelines = []
                for (timeline, participant_id), match_data in zip(timelines, all_matches):
                    match_with_timeline = {
                        'info': {
                            'participants': [{
                                'puuid': puuid,
                                'participantId': participant_id,
                                'championName': match_data.get('championName')
                            }]
                        },
                        'timeline': timeline,
                        'gameId': match_data.get('gameId', 'unknown'),
                        'championName': match_data.get('championName')
                    }
                    matches_with_timelines.append(match_with_timeline)
                
                # Use the map overlay heatmap (gradient over Summoner's Rift image)
                heatmap_base64 = create_map_overlay_heatmap(matches_with_timelines, puuid, use_precise=False)
                
                if heatmap_base64:
                    st.markdown(f'<img src="{heatmap_base64}" style="width:100%">', unsafe_allow_html=True)
                else:
                    st.info("No death position data found for these matches")
                    
                # Individual game heatmaps
                st.markdown("---")
                st.markdown("### INDIVIDUAL MATCH HEATMAPS")
                st.caption("Showing death locations for each match individually")
                
                # Create columns for side-by-side layout (5 columns)
                cols = st.columns(5)
                
                for i, (match_data, (timeline, participant_id)) in enumerate(zip(all_matches, timelines)):
                    champion = match_data.get('championName', 'Unknown')
                    result = "WIN" if match_data.get('win') else "LOSS"
                    kda = f"{match_data.get('kills', 0)}/{match_data.get('deaths', 0)}/{match_data.get('assists', 0)}"
                    role = match_data.get('teamPosition', '').title() if match_data.get('teamPosition') else ''
                    side = 'B' if match_data.get('teamId') == 100 else 'R' if match_data.get('teamId') == 200 else ''
                    icon_url = get_champion_icon_url(champion)
                    
                    with cols[i % 5]:
                        # Champion icon and name
                        st.markdown(f'<img src="{icon_url}" style="width: 40px; height: 40px; border-radius: 50%; border: 2px solid {"#4caf50" if match_data.get("win") else "#f44336"}; margin-bottom: 0.5rem;" onerror="this.style.display=\'none\'">',
                                    unsafe_allow_html=True)
                        
                        role_text = f" • {role}" if role else ""
                        st.markdown(f"**{champion}** {side}")
                        st.caption(f"{result} • {kda}")
                        
                        # Create single match data
                        single_match = [{
                            'info': {
                                'participants': [{
                                    'puuid': puuid,
                                    'participantId': participant_id,
                                    'championName': champion
                                }]
                            },
                            'timeline': timeline,
                            'gameId': match_data.get('gameId', f'match_{i}'),
                            'championName': champion
                        }]
                        
                        try:
                            match_heatmap = create_map_overlay_heatmap(single_match, puuid, use_precise=True)
                            if match_heatmap:
                                st.markdown(f'<img src="{match_heatmap}" style="width:100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">', unsafe_allow_html=True)
                            else:
                                # Empty placeholder to maintain spacing - square aspect ratio to match heatmap
                                st.markdown('<div style="width:100%; aspect-ratio: 1/1; display:flex; align-items:center; justify-content:center; background-color:#1e1e1e; border-radius:8px; border: 2px dashed #444;"><p style="color:#888; font-size: 0.9rem;">No deaths</p></div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Error: {e}")
                        
                        st.markdown("")  # Add spacing
                            
            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")
        
        # Champion Mastery
        with st.spinner("Analyzing champion pool..."):
            champion_insights = bedrock.generate_champion_mastery_insights(player_data)
        
        st.markdown('<div class="section-header">CHAMPION MASTERY</div>', unsafe_allow_html=True)
        st.markdown(champion_insights)
        
        # Improvement Roadmap
        with st.spinner("Creating improvement roadmap..."):
            roadmap = bedrock.generate_improvement_roadmap(player_data, ml_insights)
        
        st.markdown('<div class="section-header">30-DAY IMPROVEMENT PLAN</div>', unsafe_allow_html=True)
        st.markdown(roadmap)

# Run analysis
if analyze_button:
    if not game_name or not tag_line:
        st.error("Please enter both Summoner Name and Tag Line")
    else:
        asyncio.run(analyze_player(game_name, tag_line, region))

# Run seven degrees of separation
if separation_button:
    if not game_name or not tag_line:
        st.error("Please enter your Summoner Name and Tag Line first")
    elif not target_game_name or not target_tag_line:
        st.error("Please enter both Target Player Name and Tag")
    else:
        st.markdown('<div class="section-header">PLAYER CONNECTION SEARCH</div>', unsafe_allow_html=True)
        st.info(f"Searching for connection between **{game_name}#{tag_line}** and **{target_game_name}#{target_tag_line}**...")
        
        # Create container for live updates
        live_container = st.container()
        
        with live_container:
            # Create placeholders for dynamic updates
            tree_placeholder = st.empty()
            status_text = st.empty()
        
        # Tree visualization state
        current_tree = {'nodes': [], 'edges': []}
        
        def render_tree(tree_data):
            """Render the current search tree with hierarchical structure"""
            if not tree_data['nodes']:
                return
            
            # Build tree HTML with proper hierarchy
            tree_html = '<div style="background: rgba(10, 14, 39, 0.6); padding: 2rem; border-radius: 12px; overflow-x: auto; margin-bottom: 1rem;">'
            tree_html += '<div style="text-align: center; color: #667eea; font-weight: 600; margin-bottom: 2rem; font-size: 1.1rem;">LIVE SEARCH NETWORK</div>'
            
            # Group nodes by depth and create parent-child relationships
            nodes_by_depth = {}
            node_lookup = {}
            for node in tree_data['nodes']:
                depth = node['depth']
                if depth not in nodes_by_depth:
                    nodes_by_depth[depth] = []
                nodes_by_depth[depth].append(node)
                node_lookup[node['id']] = node
            
            # Create edge lookup for parent-child relationships
            children_by_parent = {}
            for edge in tree_data['edges']:
                parent = edge['from']
                child = edge['to']
                if parent not in children_by_parent:
                    children_by_parent[parent] = []
                children_by_parent[parent].append(child)
            
            # Render tree with hierarchical structure
            tree_html += '<div style="display: flex; flex-direction: column; align-items: center; gap: 2rem;">'
            
            for depth in sorted(nodes_by_depth.keys()):
                # Depth label
                tree_html += f'<div style="width: 100%;">'
                tree_html += f'<div style="text-align: center; color: #888; margin-bottom: 1rem; font-size: 0.9rem; font-weight: 600; letter-spacing: 1px;">DEPTH {depth}</div>'
                
                # Nodes at this depth
                tree_html += '<div style="display: flex; justify-content: center; align-items: start; gap: 1.5rem; flex-wrap: wrap; position: relative;">'
                
                for node in nodes_by_depth[depth]:
                    node_id = node['id']
                    has_children = node_id in children_by_parent
                    
                    # Color based on type
                    if node['type'] == 'source':
                        bg_color = '#667eea'
                        border = '2px solid #8b9eff'
                        glow = '0 0 20px rgba(102, 126, 234, 0.6)'
                    elif node['type'] == 'target':
                        bg_color = '#764ba2'
                        border = '2px solid #9d6bc4'
                        glow = '0 0 20px rgba(118, 75, 162, 0.6)'
                    else:
                        bg_color = 'rgba(72, 187, 120, 0.7)'
                        border = '2px solid rgba(72, 187, 120, 0.9)'
                        glow = '0 0 15px rgba(72, 187, 120, 0.4)'
                    
                    # Node container with connecting line below if has children
                    tree_html += f'<div style="display: flex; flex-direction: column; align-items: center; position: relative;">'
                    
                    # The node itself
                    tree_html += f'''<div style="
                        background: {bg_color}; 
                        padding: 0.6rem 1rem; 
                        border-radius: 8px; 
                        border: {border};
                        font-size: 0.85rem;
                        font-weight: 600;
                        color: white;
                        min-width: 100px;
                        max-width: 140px;
                        text-align: center;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                        box-shadow: {glow};
                        animation: fadeIn 0.3s ease;
                        position: relative;
                        z-index: 10;
                    ">{node['name'].split('#')[0]}</div>'''
                    
                    # Vertical line down if has children (only show in live view)
                    if has_children and depth < max(nodes_by_depth.keys()):
                        tree_html += f'<div style="width: 2px; height: 30px; background: linear-gradient(to bottom, {bg_color}, rgba(136, 136, 136, 0.5)); margin-top: -1px; z-index: 1;"></div>'
                    
                    tree_html += '</div>'
                
                tree_html += '</div></div>'
            
            tree_html += '</div>'
            tree_html += '<style>@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(-10px); }} to {{ opacity: 1; transform: translateY(0); }} }}</style>'
            tree_html += '</div>'
            
            # Write to placeholder (no .empty() to prevent flashing)
            tree_placeholder.markdown(tree_html, unsafe_allow_html=True)
        
        def update_progress(players_checked, matches_checked, unique_players, tree_update=None):
            """Update progress and tree visualization"""
            try:
                status_text.text(f"Checked {players_checked} players, {matches_checked} matches, {unique_players} unique players found...")
                
                if tree_update:
                    # Update tree data without deep copy (faster)
                    current_tree['nodes'] = tree_update['nodes']
                    current_tree['edges'] = tree_update['edges']
                    render_tree(current_tree)
                    # Removed sleep - Streamlit handles updates asynchronously
            except Exception as e:
                # Ignore WebSocket errors if browser disconnected
                pass
        
        with st.spinner(f"Searching player network (checking {matches_to_check} matches per player)..."):
            result = asyncio.run(find_player_connection(
                game_name, tag_line,
                target_game_name, target_tag_line,
                region, max_depth=3, matches_per_player=matches_to_check,
                progress_callback=update_progress
            ))
        
        # Clear progress display
        tree_placeholder.empty()
        status_text.empty()
        
        if 'error' in result:
            st.error(f"{result['error']}")
        else:
            # Success! Display the connection path
            degree = result['degree']
            path = result['path']
            matches = result['matches']
            stats = result.get('stats', {})
            search_tree = result.get('search_tree', {'nodes': [], 'edges': []})
            
            # Header with degree count
            st.success(f"**Connection Found!** {degree} degree{'s' if degree != 1 else ''} of separation")
            
            # Search Tree Visualization
            if search_tree['nodes']:
                st.markdown("### SEARCH NETWORK")
                st.caption(f"Explored {len(search_tree['nodes'])} players across the network")
                
                # Create a container for the final tree
                final_tree_container = st.container()
                
                with final_tree_container:
                    # Build the tree HTML with hierarchical structure
                    tree_html = '<div style="background: rgba(10, 14, 39, 0.6); padding: 2rem; border-radius: 12px; overflow-x: auto;">'
                    
                    # Group nodes by depth
                    nodes_by_depth = {}
                    node_lookup = {}
                    for node in search_tree['nodes']:
                        depth = node['depth']
                        if depth not in nodes_by_depth:
                            nodes_by_depth[depth] = []
                        nodes_by_depth[depth].append(node)
                        node_lookup[node['id']] = node
                    
                    # Create edge lookup for parent-child relationships
                    children_by_parent = {}
                    for edge in search_tree['edges']:
                        parent = edge['from']
                        child = edge['to']
                        if parent not in children_by_parent:
                            children_by_parent[parent] = []
                        children_by_parent[parent].append(child)
                    
                    # Create path set for highlighting
                    path_names_set = set(path)
                    
                    # Render tree with hierarchical structure
                    tree_html += '<div style="display: flex; flex-direction: column; align-items: center; gap: 2.5rem;">'
                    
                    # Render each depth level
                    for depth in sorted(nodes_by_depth.keys()):
                        tree_html += f'<div style="width: 100%;">'
                        tree_html += f'<div style="text-align: center; color: #888; margin-bottom: 1.2rem; font-size: 0.95rem; font-weight: 600; letter-spacing: 1px;">DEPTH {depth}</div>'
                        tree_html += '<div style="display: flex; justify-content: center; align-items: start; gap: 2rem; flex-wrap: wrap;">'
                        
                        for node in nodes_by_depth[depth]:
                            node_id = node['id']
                            has_children = node_id in children_by_parent
                            
                            # Check if this node is in the solution path
                            is_in_path = node['name'] in path_names_set
                            
                            # Color based on type and path membership
                            if node['type'] == 'source':
                                bg_color = '#667eea'
                                border = '3px solid #8b9eff'
                                glow = '0 0 25px rgba(102, 126, 234, 0.8)'
                            elif node['type'] == 'target':
                                bg_color = '#764ba2'
                                border = '3px solid #9d6bc4'
                                glow = '0 0 25px rgba(118, 75, 162, 0.8)'
                            else:
                                if is_in_path:
                                    bg_color = '#48bb78'
                                    border = '3px solid #5fd98f'
                                    glow = '0 0 20px rgba(72, 187, 120, 0.7)'
                                else:
                                    bg_color = 'rgba(72, 187, 120, 0.25)'
                                    border = '1px solid rgba(72, 187, 120, 0.4)'
                                    glow = 'none'
                            
                            opacity = '1' if is_in_path or node['type'] in ['source', 'target'] else '0.35'
                            
                            # Node container
                            tree_html += f'<div style="display: flex; flex-direction: column; align-items: center;">'
                            
                            # The node
                            tree_html += f'''<div style="background: {bg_color}; padding: 0.7rem 1.2rem; border-radius: 10px; border: {border}; font-size: 0.9rem; font-weight: 600; color: white; opacity: {opacity}; min-width: 110px; max-width: 160px; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; box-shadow: {glow}; transition: all 0.3s ease; position: relative; z-index: 10;">{node['name'].split('#')[0]}</div>'''
                            
                            # Vertical line down if has children
                            if has_children and depth < max(nodes_by_depth.keys()) and (is_in_path or node['type'] in ['source', 'target']):
                                line_color = bg_color if opacity == '1' else 'rgba(136, 136, 136, 0.3)'
                                tree_html += f'<div style="width: 3px; height: 35px; background: linear-gradient(to bottom, {line_color}, rgba(136, 136, 136, 0.3)); margin-top: -2px; z-index: 1; opacity: {opacity};"></div>'
                            
                            tree_html += '</div>'
                        
                        tree_html += '</div></div>'
                    
                    tree_html += '</div></div>'
                    st.markdown(tree_html, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Visual connection path
            st.markdown("### CONNECTION CHAIN")
            
            # Create a visual path with arrows
            path_html = '<div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 1rem; margin: 2rem 0;">'
            for i, player_name in enumerate(path):
                # Highlight source and target differently
                if i == 0:
                    color = "#667eea"  # Source (purple)
                    label = "SOURCE"
                elif i == len(path) - 1:
                    color = "#764ba2"  # Target (darker purple)
                    label = "TARGET"
                else:
                    color = "#48bb78"  # Intermediate (green)
                    label = f"LINK {i}"
                
                path_html += f'''
                <div style="text-align: center;">
                    <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                                padding: 1rem 1.5rem; 
                                border-radius: 10px; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                                min-width: 150px;">
                        <div style="font-size: 0.7rem; margin-bottom: 0.5rem; opacity: 0.8; font-weight: 700; letter-spacing: 1px;">{label}</div>
                        <div style="font-weight: bold; font-size: 1rem; color: white;">{player_name}</div>
                    </div>
                </div>
                '''
                
                # Add arrow between players (except after last player)
                if i < len(path) - 1:
                    path_html += '<div style="font-size: 2rem; color: #888;">➜</div>'
            
            path_html += '</div>'
            st.markdown(path_html, unsafe_allow_html=True)
            
            # Display match details
            st.markdown("---")
            st.markdown("### CONNECTION MATCHES")
            st.caption("These are the matches that connect the players")
            
            for i, match_info in enumerate(matches):
                game_duration_mins = match_info['game_duration'] // 60
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                            padding: 1rem;
                            border-radius: 8px;
                            border-left: 4px solid #667eea;
                            margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0;">Match {i + 1}: {match_info['from']} ➜ {match_info['to']}</h4>
                    <p style="margin: 0.25rem 0; color: #888;">
                        <strong>Game Mode:</strong> {match_info['game_mode']} • 
                        <strong>Duration:</strong> {game_duration_mins} min • 
                        <strong>Match ID:</strong> <code style="font-size: 0.8rem;">{match_info['match_id']}</code>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Fun stats
            st.markdown("---")
            st.markdown("### Connection Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Degrees of Separation", degree)
            with col2:
                st.metric("Players in Chain", len(path))
            with col3:
                st.metric("Matches Checked", stats.get('matches_checked', len(matches)))
            with col4:
                st.metric("Players Explored", stats.get('unique_players', 'N/A'))
            with col5:
                st.metric("Time Taken", stats.get('time_taken', 'N/A'))

# Footer
st.markdown("</div>", unsafe_allow_html=True)  # Close content-section
st.markdown("---")
st.markdown("""
<div class="footer">
    <h2 style="margin-bottom: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;">Powered by Elite Technology</h2>
    <div style="display: flex; justify-content: center; gap: 2rem; margin: 2rem 0; flex-wrap: wrap;">
        <div style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">AI</div>
            <div style="font-weight: 600; color: #667eea;">AWS Bedrock</div>
            <div style="font-size: 0.9rem; color: #888;">Claude 3 Sonnet</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">API</div>
            <div style="font-weight: 600; color: #667eea;">Riot Games API</div>
            <div style="font-size: 0.9rem; color: #888;">Real-time Data</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">ML</div>
            <div style="font-weight: 600; color: #667eea;">Python ML</div>
            <div style="font-size: 0.9rem; color: #888;">Advanced Analytics</div>
        </div>
    </div>
    <div style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid rgba(102, 126, 234, 0.2);">
        <p style="color: #667eea; font-weight: 600; margin: 0.5rem 0; font-size: 1.1rem;">
            Efficient & Affordable
        </p>
        <p style="color: #888; margin-top: 0.5rem; font-size: 0.95rem;">
            ~$0.04/month typical usage
        </p>
    </div>
    <p style="color: #666; font-size: 0.85rem; margin-top: 2rem; font-style: italic; opacity: 0.7;">
        RiftRewind isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games<br>
        or anyone officially involved in producing or managing League of Legends.
    </p>
</div>
""", unsafe_allow_html=True)
