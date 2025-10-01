import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import base64
from io import BytesIO

def debug_match_structure(match_data, index=0):
    """
    Debug function to print the structure of match data for troubleshooting
    """
    print(f"\n=== DEBUG: Match {index} Structure ===")
    print(f"Top-level keys: {list(match_data.keys())}")
    
    if 'info' in match_data:
        print("Has 'info' key (raw match data)")
        if 'participants' in match_data['info']:
            print(f"  Participants count: {len(match_data['info']['participants'])}")
    
    if 'timeline' in match_data:
        timeline = match_data['timeline']
        print(f"Has 'timeline' key: {type(timeline)}")
        if timeline:
            print(f"  Timeline keys: {list(timeline.keys()) if isinstance(timeline, dict) else 'Not a dict'}")
            if isinstance(timeline, dict):
                if 'info' in timeline and 'frames' in timeline['info']:
                    print(f"  Timeline frames (via info): {len(timeline['info']['frames'])}")
                elif 'frames' in timeline:
                    print(f"  Timeline frames (direct): {len(timeline['frames'])}")
    
    if 'full_match_data' in match_data:
        print("Has 'full_match_data' key (extracted player data)")
        full_match = match_data['full_match_data']
        if 'timeline' in full_match:
            print("  Full match has timeline data")
    
    if 'championName' in match_data:
        print(f"Champion: {match_data['championName']}")
    
    print("=" * 40)

def create_death_heatmap(matches_data, player_puuid):
    """
    Creates a death heatmap showing where the player died most frequently
    Different colors for different champions
    """
    deaths_data = []
    total_matches_processed = 0
    matches_with_timeline = 0
    
    # Extract death positions from match data
    for match in matches_data:
        total_matches_processed += 1
        
        # Handle both raw match data and extracted player match data
        match_info = None
        timeline_data = None
        participant_id = None
        champion_name = 'Unknown'
        
        # Check if this is raw match data (from API) or extracted player data
        if 'info' in match and 'participants' in match['info']:
            # This is raw match data
            match_info = match['info']
            timeline_data = match.get('timeline')
            
            # Find the player in this match
            player_participant = None
            for participant in match_info['participants']:
                if participant.get('puuid') == player_puuid:
                    player_participant = participant
                    participant_id = participant.get('participantId')
                    champion_name = participant.get('championName', 'Unknown')
                    break
            
            if not player_participant:
                continue
                
        elif 'full_match_data' in match:
            # This is the format from get_comprehensive_player_data
            full_match = match['full_match_data']
            match_info = full_match.get('info', {})
            timeline_data = full_match.get('timeline')
            champion_name = match.get('championName', 'Unknown')
            
            # Find the player in this match
            player_participant = None
            for participant in match_info.get('participants', []):
                if participant.get('puuid') == player_puuid:
                    player_participant = participant
                    participant_id = participant.get('participantId')
                    champion_name = participant.get('championName', champion_name)
                    break
            
            if not player_participant:
                continue
        else:
            continue
        
        # Process timeline data if available
        if timeline_data and timeline_data is not None:
            matches_with_timeline += 1
            try:
                # Handle different timeline data structures
                if 'info' in timeline_data and 'frames' in timeline_data['info']:
                    frames = timeline_data['info']['frames']
                elif 'frames' in timeline_data:
                    frames = timeline_data['frames']
                else:
                    print(f"Warning: Unexpected timeline structure in match, skipping")
                    continue
                
                for frame in frames:
                    events = frame.get('events', [])
                    for event in events:
                        if (event.get('type') == 'CHAMPION_KILL' and 
                            participant_id and
                            event.get('victimId') == participant_id):
                            
                            position = event.get('position', {})
                            x = position.get('x')
                            y = position.get('y')
                            
                            # Validate position data - ensure coordinates are within expected ranges
                            if (x is not None and y is not None and 
                                0 < x < 20000 and 0 < y < 20000):  # League map bounds check
                                
                                deaths_data.append({
                                    'x': x,  # Use raw coordinates like other functions
                                    'y': y,
                                    'champion': champion_name,
                                    'timestamp': event.get('timestamp', 0),
                                    'match_id': match.get('gameId', 'unknown')
                                })
            except Exception as e:
                print(f"Warning: Error processing timeline for match {match.get('gameId', 'unknown')}: {e}")
    
    # Debug: Processed {total_matches_processed} matches, found {len(deaths_data)} deaths
    
    if not deaths_data:
        return None
    
    df = pd.DataFrame(deaths_data)
    
    # Group by champion and create separate heatmaps
    champions = df['champion'].unique()
    
    # Create subplots for each champion
    n_champions = len(champions)
    cols = min(3, n_champions)
    rows = (n_champions + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{champ} Deaths" for champ in champions],
        specs=[[{"type": "xy"}] * cols for _ in range(rows)]
    )
    
    # Create heatmap for each champion
    for i, champion in enumerate(champions):
        champion_data = df[df['champion'] == champion]
        
        if len(champion_data) > 1:
            # Create 2D histogram for heatmap
            x_coords = champion_data['x'].values
            y_coords = champion_data['y'].values
            
            # Create density heatmap
            heatmap = go.Histogram2d(
                x=x_coords,
                y=y_coords,
                nbinsx=50,
                nbinsy=50,
                colorscale='Reds',
                showscale=(i == 0),  # Only show colorscale for first plot
                name=champion
            )
            
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(heatmap, row=row, col=col)
            
            # Update axes to match Summoner's Rift dimensions
            fig.update_xaxes(
                range=[0, 14870],
                title_text="Map X",
                row=row, col=col
            )
            fig.update_yaxes(
                range=[0, 14980],
                title_text="Map Y",
                row=row, col=col
            )
    
    fig.update_layout(
        height=400 * rows,
        title_text=f"Death Heatmaps by Champion ({len(deaths_data)} deaths analyzed)",
        title_x=0.5
    )
    
    return fig

def create_combined_death_heatmap(matches_data, player_puuid):
    """
    Creates a single density heatmap with all deaths over Summoner's Rift background
    """
    deaths_data = []
    total_matches_processed = 0
    matches_with_timeline = 0
    
    # Extract death positions using the same improved logic
    for match in matches_data:
        total_matches_processed += 1
        
        # Handle both raw match data and extracted player match data
        match_info = None
        timeline_data = None
        participant_id = None
        champion_name = 'Unknown'
        
        # Check if this is raw match data (from API) or extracted player data
        if 'info' in match and 'participants' in match['info']:
            # This is raw match data
            match_info = match['info']
            timeline_data = match.get('timeline')
            
            # Find the player in this match
            player_participant = None
            for participant in match_info['participants']:
                if participant.get('puuid') == player_puuid:
                    player_participant = participant
                    participant_id = participant.get('participantId')
                    champion_name = participant.get('championName', 'Unknown')
                    break
            
            if not player_participant:
                continue
                
        elif 'full_match_data' in match:
            # This is extracted player match data with embedded full match data
            full_match = match['full_match_data']
            match_info = full_match.get('info', {})
            timeline_data = full_match.get('timeline')
            champion_name = match.get('championName', 'Unknown')
            
            # Find the player in the full match data
            player_participant = None
            for participant in match_info.get('participants', []):
                if participant.get('puuid') == player_puuid:
                    player_participant = participant
                    participant_id = participant.get('participantId')
                    break
            
            if not player_participant:
                continue
                
        elif 'timeline' in match and match.get('championName'):
            # This is extracted player match data without full match embedded
            timeline_data = match.get('timeline')
            champion_name = match.get('championName', 'Unknown')
            # For this case, we can't easily get participant ID, so skip
            continue
        else:
            continue
        
        # Process timeline data if available
        if timeline_data and timeline_data is not None:
            matches_with_timeline += 1
            try:
                # Handle different timeline data structures
                if 'info' in timeline_data and 'frames' in timeline_data['info']:
                    frames = timeline_data['info']['frames']
                elif 'frames' in timeline_data:
                    frames = timeline_data['frames']
                else:
                    print(f"Warning: Unexpected timeline structure in combined heatmap, skipping")
                    continue
                
                for frame in frames:
                    events = frame.get('events', [])
                    for event in events:
                        if (event.get('type') == 'CHAMPION_KILL' and 
                            participant_id and
                            event.get('victimId') == participant_id):
                            
                            position = event.get('position', {})
                            x = position.get('x')
                            y = position.get('y')
                            
                            # Validate position data - ensure coordinates are within expected ranges
                            if (x is not None and y is not None and 
                                0 < x < 20000 and 0 < y < 20000):  # League map bounds check
                                # Keep original coordinates for proper heatmap overlay
                                deaths_data.append({
                                    'x': x,
                                    'y': y,
                                    'champion': champion_name
                                })
            except Exception as e:
                print(f"Warning: Error processing timeline in combined heatmap for match {match.get('gameId', 'unknown')}: {e}")
    
    # Debug: Combined heatmap - processed {total_matches_processed} matches, found {len(deaths_data)} deaths
    
    if not deaths_data:
        return None
    
    df = pd.DataFrame(deaths_data)
    
    # Create density heatmap over Summoner's Rift coordinates
    x_coords = df['x'].values
    y_coords = df['y'].values
    
    # Create 2D histogram density heatmap
    fig = go.Figure()
    
    # Add the main heatmap
    heatmap = go.Histogram2d(
        x=x_coords,
        y=y_coords,
        nbinsx=60,  # Higher resolution for better detail
        nbinsy=60,
        colorscale='hot',  # Hot colorscale for death intensity
        reversescale=True,  # Dark = more deaths
        showscale=True,
        colorbar=dict(
            title="Death Density",
            tickmode="linear",
            tick0=0,
            dtick=1
        )
    )
    
    fig.add_trace(heatmap)
    
    # Add major landmarks as reference points
    landmarks = [
        {"name": "Blue Nexus", "x": 1748, "y": 1562, "color": "blue"},
        {"name": "Red Nexus", "x": 13052, "y": 13224, "color": "red"},
        {"name": "Baron", "x": 5007, "y": 10471, "color": "purple"},
        {"name": "Dragon", "x": 9866, "y": 4414, "color": "orange"},
        {"name": "Blue Mid Turret", "x": 5846, "y": 6396, "color": "lightblue"},
        {"name": "Red Mid Turret", "x": 8955, "y": 8236, "color": "lightcoral"}
    ]
    
    for landmark in landmarks:
        fig.add_trace(go.Scatter(
            x=[landmark["x"]],
            y=[landmark["y"]],
            mode="markers+text",
            marker=dict(size=8, color=landmark["color"], symbol="star"),
            text=[landmark["name"]],
            textposition="top center",
            showlegend=False,
            name=landmark["name"]
        ))
    
    # Update layout to match Summoner's Rift dimensions and appearance
    fig.update_layout(
        title=f"Death Density Heatmap ({len(deaths_data)} deaths analyzed)",
        title_x=0.5,
        xaxis=dict(
            title="Map X Coordinate",
            range=[0, 14870],
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.3)",
            zeroline=False
        ),
        yaxis=dict(
            title="Map Y Coordinate",
            range=[0, 14980],
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.3)",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor="rgba(20, 40, 20, 0.8)",  # Dark green background like Summoner's Rift
        paper_bgcolor="rgba(0, 0, 0, 0)",
        height=600,
        width=700
    )
    
    return fig

def create_summoners_rift_heatmap(matches_data, player_puuid):
    """
    Creates a death heatmap overlaid on actual Summoner's Rift map
    """
    deaths_data = []
    total_matches_processed = 0
    matches_with_timeline = 0
    
    # Extract death positions (same logic as other functions)
    for match in matches_data:
        total_matches_processed += 1
        
        match_info = None
        timeline_data = None
        participant_id = None
        
        if 'info' in match and 'participants' in match['info']:
            match_info = match['info']
            timeline_data = match.get('timeline')
            
            player_participant = None
            for participant in match_info['participants']:
                if participant.get('puuid') == player_puuid:
                    player_participant = participant
                    participant_id = participant.get('participantId')
                    break
            
            if not player_participant:
                continue
        elif 'full_match_data' in match:
            full_match = match['full_match_data']
            match_info = full_match.get('info', {})
            timeline_data = full_match.get('timeline')
            
            player_participant = None
            for participant in match_info.get('participants', []):
                if participant.get('puuid') == player_puuid:
                    player_participant = participant
                    participant_id = participant.get('participantId')
                    break
            
            if not player_participant:
                continue
        else:
            continue
        
        if timeline_data and timeline_data is not None:
            matches_with_timeline += 1
            try:
                if 'info' in timeline_data and 'frames' in timeline_data['info']:
                    frames = timeline_data['info']['frames']
                elif 'frames' in timeline_data:
                    frames = timeline_data['frames']
                else:
                    continue
                
                for frame in frames:
                    events = frame.get('events', [])
                    for event in events:
                        if (event.get('type') == 'CHAMPION_KILL' and 
                            participant_id and
                            event.get('victimId') == participant_id):
                            
                            position = event.get('position', {})
                            x = position.get('x')
                            y = position.get('y')
                            
                            if (x is not None and y is not None and 
                                0 < x < 20000 and 0 < y < 20000):
                                deaths_data.append({'x': x, 'y': y})
            except Exception as e:
                continue
    
    if not deaths_data or len(deaths_data) < 2:
        return None
    
    # Create the Plotly figure with Summoner's Rift styling
    fig = go.Figure()
    
    # Extract coordinates
    x_coords = [d['x'] for d in deaths_data]
    y_coords = [d['y'] for d in deaths_data]
    
    # Create density heatmap
    fig.add_trace(go.Histogram2d(
        x=x_coords,
        y=y_coords,
        nbinsx=40,
        nbinsy=40,
        colorscale=[
            [0, 'rgba(0,0,0,0)'],          # Transparent for no deaths
            [0.1, 'rgba(0,0,255,0.3)'],    # Blue for few deaths
            [0.3, 'rgba(0,255,0,0.5)'],    # Green
            [0.5, 'rgba(255,255,0,0.7)'],  # Yellow
            [0.7, 'rgba(255,165,0,0.8)'],  # Orange
            [1.0, 'rgba(255,0,0,0.9)']     # Red for many deaths
        ],
        showscale=True,
        colorbar=dict(title="Deaths", x=1.02)
    ))
    
    # Add Summoner's Rift features as overlays
    
    # River (approximate)
    river_x = [0, 2500, 5000, 7500, 10000, 12500, 14870]
    river_y = [7000, 6800, 6500, 7200, 7800, 8200, 8500]
    fig.add_trace(go.Scatter(
        x=river_x, y=river_y,
        mode='lines',
        line=dict(color='lightblue', width=8, dash='dot'),
        name='River',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Major objectives
    objectives = [
        {"name": "Blue Nexus", "x": 1748, "y": 1562, "color": "blue", "size": 15},
        {"name": "Red Nexus", "x": 13052, "y": 13224, "color": "red", "size": 15},
        {"name": "Baron Nashor", "x": 5007, "y": 10471, "color": "purple", "size": 12},
        {"name": "Dragon", "x": 9866, "y": 4414, "color": "orange", "size": 12},
        {"name": "Blue Buff", "x": 3804, "y": 7875, "color": "lightblue", "size": 8},
        {"name": "Red Buff", "x": 11098, "y": 6950, "color": "lightcoral", "size": 8},
    ]
    
    for obj in objectives:
        fig.add_trace(go.Scatter(
            x=[obj["x"]], y=[obj["y"]],
            mode='markers+text',
            marker=dict(size=obj["size"], color=obj["color"], 
                       symbol='circle', line=dict(color='white', width=2)),
            text=[obj["name"]], textposition="top center",
            name=obj["name"], showlegend=False,
            hovertemplate=f"<b>{obj['name']}</b><br>X: {obj['x']}<br>Y: {obj['y']}<extra></extra>"
        ))
    
    # Add lane indicators
    lanes = [
        # Top lane
        {"x": [1200, 13000], "y": [13500, 2000], "name": "Top Lane"},
        # Mid lane  
        {"x": [1748, 13052], "y": [1562, 13224], "name": "Mid Lane"},
        # Bot lane
        {"x": [2000, 13500], "y": [1200, 13000], "name": "Bot Lane"},
    ]
    
    for lane in lanes:
        fig.add_trace(go.Scatter(
            x=lane["x"], y=lane["y"],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'),
            name=lane["name"], showlegend=False, hoverinfo='skip'
        ))
    
    # Style the plot to look like Summoner's Rift
    fig.update_layout(
        title=dict(
            text=f"Death Heatmap on Summoner's Rift<br><sub>{len(deaths_data)} deaths analyzed</sub>",
            x=0.5, font=dict(size=16, color='white')
        ),
        xaxis=dict(
            title="", range=[0, 14870], showgrid=False, 
            zeroline=False, showticklabels=False,
            fixedrange=True
        ),
        yaxis=dict(
            title="", range=[0, 14980], showgrid=False,
            zeroline=False, showticklabels=False,
            scaleanchor="x", scaleratio=1, fixedrange=True
        ),
        plot_bgcolor='rgba(28, 45, 28, 1)',    # Dark green like Summoner's Rift
        paper_bgcolor='rgba(0, 0, 0, 0.8)',
        font=dict(color='white'),
        showlegend=False,
        height=600, width=700,
        margin=dict(l=20, r=80, t=60, b=20)
    )
    
    return fig

def create_map_overlay_heatmap(matches_data, player_puuid):
    """
    Creates a death heatmap overlaid on the actual Summoner's Rift map image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from io import BytesIO
    import base64
    
    deaths_data = []
    total_matches_processed = 0
    matches_with_timeline = 0
    
    # Extract death positions (same logic as other functions)
    for match in matches_data:
        total_matches_processed += 1
        
        match_info = None
        timeline_data = None
        participant_id = None
        
        if 'info' in match and 'participants' in match['info']:
            match_info = match['info']
            timeline_data = match.get('timeline')
            
            player_participant = None
            for participant in match_info['participants']:
                if participant.get('puuid') == player_puuid:
                    player_participant = participant
                    participant_id = participant.get('participantId')
                    break
            
            if not player_participant:
                continue
        elif 'full_match_data' in match:
            full_match = match['full_match_data']
            match_info = full_match.get('info', {})
            timeline_data = full_match.get('timeline')
            
            player_participant = None
            for participant in match_info.get('participants', []):
                if participant.get('puuid') == player_puuid:
                    player_participant = participant
                    participant_id = participant.get('participantId')
                    break
            
            if not player_participant:
                continue
        else:
            continue
        
        if timeline_data and timeline_data is not None:
            matches_with_timeline += 1
            try:
                if 'info' in timeline_data and 'frames' in timeline_data['info']:
                    frames = timeline_data['info']['frames']
                elif 'frames' in timeline_data:
                    frames = timeline_data['frames']
                else:
                    continue
                
                for frame in frames:
                    events = frame.get('events', [])
                    for event in events:
                        if (event.get('type') == 'CHAMPION_KILL' and 
                            participant_id and
                            event.get('victimId') == participant_id):
                            
                            position = event.get('position', {})
                            x = position.get('x')
                            y = position.get('y')
                            
                            if (x is not None and y is not None and 
                                0 < x < 20000 and 0 < y < 20000):
                                deaths_data.append({'x': x, 'y': y})
            except Exception as e:
                continue
    
    print(f"Debug: Map overlay found {len(deaths_data)} deaths from {matches_with_timeline}/{total_matches_processed} matches with timeline data")
    
    if not deaths_data or len(deaths_data) < 2:
        return None
    
    # Load and display the actual Summoner's Rift map image
    try:
        from PIL import Image
        import os
        
        # Load the PNG image
        map_image_path = os.path.join(os.path.dirname(__file__), 'assets', 'summoners-rift-map.png')
        if not os.path.exists(map_image_path):
            # Fallback to current directory
            map_image_path = os.path.join('assets', 'summoners-rift-map.png')
        
        map_image = Image.open(map_image_path)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(14, 14))
        fig.patch.set_facecolor('#000000')  # Black background
        
        # Display the map image as background with full opacity
        # The image coordinates should match League's coordinate system (0-14870 x 0-14980)
        ax.imshow(map_image, extent=[0, 14870, 0, 14980], aspect='equal', alpha=1.0)
        
    except Exception as e:
        print(f"Could not load map image: {e}")
        # Fallback to stylized background
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor('#2d4a2d')
        fig.patch.set_facecolor('#1a1a1a')
    
    # Extract coordinates and create heatmap
    x_coords = [d['x'] for d in deaths_data]
    y_coords = [d['y'] for d in deaths_data]
    
    # Create uniform base layer covering entire map + death density overlay
    from scipy.stats import gaussian_kde
    
    # Create high-resolution coordinate meshgrid covering entire map
    x_min, x_max = 0, 14870
    y_min, y_max = 0, 14980
    xx, yy = np.mgrid[x_min:x_max:150j, y_min:y_max:150j]
    
    # Create base layer - uniform low value across entire map
    base_density = np.full(xx.shape, 0.2)  # Uniform purple base layer
    
    # Calculate KDE death density
    death_positions = np.vstack([x_coords, y_coords])
    kde = gaussian_kde(death_positions)
    kde.set_bandwidth(bw_method=0.25)
    
    positions = np.vstack([xx.ravel(), yy.ravel()])
    death_density = kde(positions).T.reshape(xx.shape)
    
    # Normalize death density and add to base
    if death_density.max() > 0:
        death_density = death_density / death_density.max() * 0.8  # Scale to 0-0.8
        combined_density = base_density + death_density  # Base 0.2 + death 0-0.8 = 0.2-1.0
    else:
        combined_density = base_density
    
    # Ensure values are in 0-1 range
    combined_density = np.clip(combined_density, 0, 1)
    
    # Create full-map heatmap overlay (no masking - covers entire map)
    extent = [0, 14870, 0, 14980]
    im = ax.imshow(combined_density, origin='lower', extent=extent,
                   cmap='plasma', alpha=0.65, interpolation='bicubic', vmin=0.2, vmax=1.0)
    
    # Add major objectives as markers (optional, can be toggled)
    show_objectives = False  # Set to True if you want objective markers
    
    if show_objectives:
        objectives = [
            {"name": "Blue Nexus", "x": 1748, "y": 1562, "color": "cyan", "size": 150, "marker": "^"},
            {"name": "Red Nexus", "x": 13052, "y": 13224, "color": "red", "size": 150, "marker": "^"},
            {"name": "Baron", "x": 5007, "y": 10471, "color": "purple", "size": 120, "marker": "D"},
            {"name": "Dragon", "x": 9866, "y": 4414, "color": "orange", "size": 120, "marker": "D"},
        ]
        
        for obj in objectives:
            ax.scatter(obj["x"], obj["y"], c=obj["color"], s=obj["size"],
                      marker=obj["marker"], edgecolors='black', linewidth=1,
                      label=obj["name"], zorder=10, alpha=0.8)
    
    # Customize the plot with adjusted limits for better alignment
    ax.set_xlim(0, 14870)
    ax.set_ylim(0, 14980)
    ax.set_aspect('equal')
    ax.axis('off')  # Remove axes for cleaner look
    
    # Add title
    fig.suptitle(f'Death Heatmap on Summoner\'s Rift\n{len(deaths_data)} deaths analyzed', 
                fontsize=18, fontweight='bold', color='white', y=0.95)
    
    # Add colorbar for heatmap
    if 'im' in locals():
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=30)
        cbar.set_label('Death Density', rotation=270, labelpad=20, color='white', fontsize=12)
        cbar.ax.tick_params(colors='white')
        # Make colorbar background semi-transparent
        cbar.ax.patch.set_facecolor('black')
        cbar.ax.patch.set_alpha(0.7)
    
    # Add legend for objectives only if shown
    if show_objectives and 'objectives' in locals():
        legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                          fontsize=9, frameon=True, facecolor='black', 
                          edgecolor='white', framealpha=0.8)
        for text in legend.get_texts():
            text.set_color('white')
    
    plt.tight_layout()
    
    # Convert to base64 for Streamlit
    buffer = BytesIO()
    plt.savefig(buffer, format='png', facecolor='#1a1a1a', dpi=150, 
                bbox_inches='tight', pad_inches=0.1)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def validate_timeline_data_availability(matches_data, player_puuid, debug=False):
    """
    Validates timeline data availability across matches and returns a summary
    """
    summary = {
        'total_matches': len(matches_data),
        'matches_with_timeline': 0,
        'matches_with_valid_timeline': 0,
        'potential_deaths_found': 0,
        'matches_with_player': 0,
        'timeline_data_structures': set()
    }
    
    for i, match in enumerate(matches_data):
        if debug and i < 3:  # Debug first 3 matches
            debug_match_structure(match, i)
        
        # Check if player is in match
        player_found = False
        timeline_data = None
        participant_id = None
        
        if 'info' in match and 'participants' in match['info']:
            # Raw match data
            for participant in match['info']['participants']:
                if participant.get('puuid') == player_puuid:
                    player_found = True
                    participant_id = participant.get('participantId')
                    break
            timeline_data = match.get('timeline')
            
        elif 'full_match_data' in match:
            # Extracted player data with full match
            full_match = match['full_match_data']
            for participant in full_match.get('info', {}).get('participants', []):
                if participant.get('puuid') == player_puuid:
                    player_found = True
                    participant_id = participant.get('participantId')
                    break
            timeline_data = full_match.get('timeline')
            
        elif 'championName' in match:
            # Assume this is player data
            player_found = True
            timeline_data = match.get('timeline')
        
        if player_found:
            summary['matches_with_player'] += 1
        
        if timeline_data:
            summary['matches_with_timeline'] += 1
            
            # Determine timeline structure
            if isinstance(timeline_data, dict):
                if 'info' in timeline_data and 'frames' in timeline_data['info']:
                    summary['timeline_data_structures'].add('nested_info_frames')
                    frames = timeline_data['info']['frames']
                elif 'frames' in timeline_data:
                    summary['timeline_data_structures'].add('direct_frames')
                    frames = timeline_data['frames']
                else:
                    summary['timeline_data_structures'].add('unknown_structure')
                    continue
                
                summary['matches_with_valid_timeline'] += 1
                
                # Count potential death events
                if participant_id:
                    for frame in frames:
                        events = frame.get('events', [])
                        for event in events:
                            if (event.get('type') == 'CHAMPION_KILL' and 
                                event.get('victimId') == participant_id):
                                summary['potential_deaths_found'] += 1
    
    summary['timeline_data_structures'] = list(summary['timeline_data_structures'])
    return summary
