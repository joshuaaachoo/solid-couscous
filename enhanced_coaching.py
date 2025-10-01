"""
Enhanced coaching analysis system for RiftRewind
Provides tactical, strategic, and progressive insights
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json

class EnhancedCoachingAnalyzer:
    """Advanced analysis system for tactical and strategic coaching"""
    
    def __init__(self):
        # Game constants for analysis
        self.EARLY_GAME = 15 * 60 * 1000  # 15 minutes in milliseconds
        self.MID_GAME = 30 * 60 * 1000    # 30 minutes in milliseconds
        
        # Map zones for strategic analysis
        self.MAP_ZONES = {
            'jungle': {'ranges': [(3000, 6000, 8000, 12000), (8000, 12000, 3000, 8000)]},
            'lane': {'ranges': [(0, 15000, 0, 3000), (0, 15000, 12000, 15000), (0, 3000, 0, 15000), (12000, 15000, 0, 15000)]},
            'river': {'ranges': [(0, 15000, 6000, 9000)]},
            'objectives': {
                'baron': (5007, 10471, 1500),  # x, y, radius
                'dragon': (9866, 4414, 1500),
                'enemy_jungle': [(8000, 15000, 8000, 12000)],
                'ally_jungle': [(0, 7000, 3000, 8000)]
            }
        }

    def analyze_death_patterns(self, matches_data: List[Dict], player_puuid: str) -> Dict[str, Any]:
        """Comprehensive death pattern analysis for tactical coaching"""
        
        deaths_data = self._extract_death_data(matches_data, player_puuid)
        
        if not deaths_data:
            return {"error": "No death data available for analysis"}
        
        analysis = {
            'tactical_insights': self._analyze_tactical_patterns(deaths_data),
            'strategic_insights': self._analyze_strategic_patterns(deaths_data),
            'timing_insights': self._analyze_death_timing(deaths_data),
            'positioning_insights': self._analyze_positioning(deaths_data),
            'improvement_areas': self._identify_improvement_areas(deaths_data)
        }
        
        return analysis
    
    def _extract_death_data(self, matches_data: List[Dict], player_puuid: str) -> List[Dict]:
        """Extract detailed death information from matches"""
        deaths_data = []
        
        for match in matches_data:
            match_info = None
            timeline_data = None
            participant_id = None
            game_duration = 0
            
            # Handle different match data formats
            if 'info' in match and 'participants' in match['info']:
                match_info = match['info']
                timeline_data = match.get('timeline')
                game_duration = match_info.get('gameDuration', 0) * 1000  # Convert to milliseconds
                
                # Find player participant
                for participant in match_info['participants']:
                    if participant.get('puuid') == player_puuid:
                        participant_id = participant.get('participantId')
                        break
                        
            elif 'full_match_data' in match:
                full_match = match['full_match_data']
                match_info = full_match.get('info', {})
                timeline_data = full_match.get('timeline')
                game_duration = match_info.get('gameDuration', 0) * 1000
                
                for participant in match_info.get('participants', []):
                    if participant.get('puuid') == player_puuid:
                        participant_id = participant.get('participantId')
                        break
            
            if not (timeline_data and participant_id):
                continue
            
            # Extract death events
            try:
                frames = timeline_data.get('info', {}).get('frames', []) or timeline_data.get('frames', [])
                
                for frame in frames:
                    timestamp = frame.get('timestamp', 0)
                    events = frame.get('events', [])
                    
                    for event in events:
                        if (event.get('type') == 'CHAMPION_KILL' and 
                            event.get('victimId') == participant_id):
                            
                            position = event.get('position', {})
                            x, y = position.get('x'), position.get('y')
                            
                            if x is not None and y is not None and 0 < x < 20000 and 0 < y < 20000:
                                deaths_data.append({
                                    'x': x,
                                    'y': y,
                                    'timestamp': timestamp,
                                    'game_duration': game_duration,
                                    'killer_id': event.get('killerId'),
                                    'assisting_participants': event.get('assistingParticipantIds', []),
                                    'game_phase': self._get_game_phase(timestamp),
                                    'zone': self._get_map_zone(x, y),
                                    'near_objective': self._check_near_objective(x, y)
                                })
            except Exception as e:
                continue
        
        return deaths_data
    
    def _get_game_phase(self, timestamp: int) -> str:
        """Determine game phase based on timestamp"""
        if timestamp < self.EARLY_GAME:
            return 'early'
        elif timestamp < self.MID_GAME:
            return 'mid'
        else:
            return 'late'
    
    def _get_map_zone(self, x: float, y: float) -> str:
        """Determine which zone of the map the death occurred in"""
        # Jungle zones
        if (3000 < x < 8000 and 3000 < y < 12000) or (8000 < x < 12000 and 3000 < y < 8000):
            if x < 7500:  # Rough mid-line
                return 'ally_jungle'
            else:
                return 'enemy_jungle'
        
        # River area
        if 6000 < y < 9000:
            return 'river'
        
        # Lane areas (simplified)
        if y < 3000 or y > 12000 or x < 3000 or x > 12000:
            return 'lane'
        
        return 'neutral'
    
    def _check_near_objective(self, x: float, y: float) -> str:
        """Check if death was near major objectives"""
        objectives = self.MAP_ZONES['objectives']
        
        # Baron
        baron_dist = np.sqrt((x - objectives['baron'][0])**2 + (y - objectives['baron'][1])**2)
        if baron_dist < objectives['baron'][2]:
            return 'baron'
        
        # Dragon
        dragon_dist = np.sqrt((x - objectives['dragon'][0])**2 + (y - objectives['dragon'][1])**2)
        if dragon_dist < objectives['dragon'][2]:
            return 'dragon'
        
        return 'none'
    
    def _analyze_tactical_patterns(self, deaths_data: List[Dict]) -> Dict[str, Any]:
        """Analyze tactical positioning and mechanical issues"""
        if not deaths_data:
            return {}
        
        total_deaths = len(deaths_data)
        
        # Phase analysis
        phase_deaths = {'early': 0, 'mid': 0, 'late': 0}
        for death in deaths_data:
            phase_deaths[death['game_phase']] += 1
        
        # Zone analysis
        zone_deaths = {}
        for death in deaths_data:
            zone = death['zone']
            zone_deaths[zone] = zone_deaths.get(zone, 0) + 1
        
        # Objective deaths
        objective_deaths = {}
        for death in deaths_data:
            obj = death['near_objective']
            if obj != 'none':
                objective_deaths[obj] = objective_deaths.get(obj, 0) + 1
        
        return {
            'phase_distribution': {k: (v/total_deaths)*100 for k, v in phase_deaths.items()},
            'zone_distribution': {k: (v/total_deaths)*100 for k, v in zone_deaths.items()},
            'objective_deaths': objective_deaths,
            'total_deaths': total_deaths,
            'tactical_recommendations': self._generate_tactical_recommendations(
                phase_deaths, zone_deaths, objective_deaths, total_deaths
            )
        }
    
    def _analyze_strategic_patterns(self, deaths_data: List[Dict]) -> Dict[str, Any]:
        """Analyze macro gameplay and strategic decision making"""
        if not deaths_data:
            return {}
        
        # Timing clusters - find when deaths spike
        timestamps = [death['timestamp'] for death in deaths_data]
        
        # Convert to minutes for easier analysis
        death_minutes = [t / (60 * 1000) for t in timestamps]
        
        # Find death clustering in time
        time_clusters = self._find_time_clusters(death_minutes)
        
        # Risk assessment based on zone
        risk_zones = []
        for death in deaths_data:
            if death['zone'] in ['enemy_jungle', 'river'] and death['near_objective'] != 'none':
                risk_zones.append(death)
        
        return {
            'death_timing_clusters': time_clusters,
            'high_risk_deaths': len(risk_zones),
            'risk_percentage': (len(risk_zones) / len(deaths_data)) * 100,
            'strategic_recommendations': self._generate_strategic_recommendations(
                time_clusters, risk_zones, deaths_data
            )
        }
    
    def _analyze_death_timing(self, deaths_data: List[Dict]) -> Dict[str, Any]:
        """Detailed timing analysis for pattern recognition"""
        if not deaths_data:
            return {}
        
        timestamps = [death['timestamp'] / (60 * 1000) for death in deaths_data]  # Convert to minutes
        
        # Calculate statistics
        avg_death_time = np.mean(timestamps)
        std_death_time = np.std(timestamps)
        
        # Find most dangerous time windows
        time_windows = {}
        for timestamp in timestamps:
            window = int(timestamp // 5) * 5  # 5-minute windows
            time_windows[window] = time_windows.get(window, 0) + 1
        
        most_dangerous_window = max(time_windows.items(), key=lambda x: x[1]) if time_windows else (0, 0)
        
        return {
            'average_death_time': avg_death_time,
            'death_time_variance': std_death_time,
            'most_dangerous_window': f"{most_dangerous_window[0]}-{most_dangerous_window[0]+5} minutes",
            'deaths_in_window': most_dangerous_window[1],
            'timing_recommendations': self._generate_timing_recommendations(
                avg_death_time, std_death_time, most_dangerous_window
            )
        }
    
    def _analyze_positioning(self, deaths_data: List[Dict]) -> Dict[str, Any]:
        """Analyze positioning patterns and hotspots"""
        if not deaths_data:
            return {}
        
        positions = [(death['x'], death['y']) for death in deaths_data]
        
        # Calculate position clustering
        position_variance = self._calculate_position_variance(positions)
        
        # Find most dangerous areas
        danger_zones = self._identify_danger_zones(deaths_data)
        
        return {
            'position_consistency': 'high' if position_variance < 2000000 else 'low',
            'position_variance': position_variance,
            'danger_zones': danger_zones,
            'positioning_recommendations': self._generate_positioning_recommendations(
                position_variance, danger_zones
            )
        }
    
    def _identify_improvement_areas(self, deaths_data: List[Dict]) -> List[str]:
        """Identify key areas for improvement based on death patterns"""
        improvements = []
        
        if not deaths_data:
            return improvements
        
        total_deaths = len(deaths_data)
        
        # Check for common issues
        late_game_deaths = sum(1 for death in deaths_data if death['game_phase'] == 'late')
        if late_game_deaths / total_deaths > 0.4:
            improvements.append("Late game positioning - 40%+ deaths occur in late game when they're most costly")
        
        enemy_jungle_deaths = sum(1 for death in deaths_data if death['zone'] == 'enemy_jungle')
        if enemy_jungle_deaths / total_deaths > 0.3:
            improvements.append("Aggressive warding - 30%+ deaths in enemy jungle suggest overextension")
        
        objective_deaths = sum(1 for death in deaths_data if death['near_objective'] != 'none')
        if objective_deaths / total_deaths > 0.25:
            improvements.append("Objective positioning - 25%+ deaths near Baron/Dragon indicate positioning issues")
        
        return improvements
    
    # Helper methods for generating recommendations
    def _generate_tactical_recommendations(self, phase_deaths, zone_deaths, objective_deaths, total_deaths):
        """Generate specific tactical recommendations"""
        recommendations = []
        
        # Phase-based recommendations
        if phase_deaths['late'] / total_deaths > 0.35:
            recommendations.append("Focus on late-game positioning - stay behind frontline in teamfights")
        
        if phase_deaths['early'] / total_deaths > 0.4:
            recommendations.append("Improve laning phase safety - ward bushes and respect enemy jungle ganks")
        
        # Zone-based recommendations
        for zone, count in zone_deaths.items():
            percentage = (count / total_deaths) * 100
            if zone == 'enemy_jungle' and percentage > 25:
                recommendations.append(f"Reduce enemy jungle deaths ({percentage:.1f}%) - improve vision before invading")
            elif zone == 'river' and percentage > 30:
                recommendations.append(f"River deaths are high ({percentage:.1f}%) - ward river bushes before roaming")
        
        # Objective recommendations
        total_obj_deaths = sum(objective_deaths.values())
        if total_obj_deaths > 0:
            obj_percentage = (total_obj_deaths / total_deaths) * 100
            recommendations.append(f"Objective deaths: {obj_percentage:.1f}% - practice positioning during Baron/Dragon fights")
        
        return recommendations
    
    def _generate_strategic_recommendations(self, time_clusters, risk_zones, deaths_data):
        """Generate strategic macro recommendations"""
        recommendations = []
        
        if time_clusters:
            most_dangerous = max(time_clusters, key=time_clusters.get)
            recommendations.append(f"Death spike at {most_dangerous:.0f} minutes - focus on safer macro decisions during this window")
        
        if len(risk_zones) > 0:
            risk_pct = (len(risk_zones) / len(deaths_data)) * 100
            recommendations.append(f"High-risk deaths: {risk_pct:.1f}% - avoid contested objectives without proper vision control")
        
        return recommendations
    
    def _generate_timing_recommendations(self, avg_time, variance, dangerous_window):
        """Generate timing-based recommendations"""
        recommendations = []
        
        if avg_time < 15:
            recommendations.append("Early death pattern - focus on laning fundamentals and jungle awareness")
        elif avg_time > 25:
            recommendations.append("Late death pattern - work on teamfight positioning and objective control")
        
        if dangerous_window[1] > 2:
            recommendations.append(f"Death cluster at {dangerous_window[0]}-{dangerous_window[0]+5} min - be extra cautious during this timing")
        
        return recommendations
    
    def _generate_positioning_recommendations(self, variance, danger_zones):
        """Generate positioning-specific recommendations"""
        recommendations = []
        
        if variance > 5000000:
            recommendations.append("Deaths spread across map - work on consistent positioning habits")
        else:
            recommendations.append("Consistent death locations found - focus on avoiding these specific danger zones")
        
        for zone in danger_zones[:3]:  # Top 3 danger zones
            recommendations.append(f"High death rate in {zone} - practice safer positioning in this area")
        
        return recommendations
    
    # Utility methods
    def _find_time_clusters(self, death_minutes: List[float]) -> Dict[int, int]:
        """Find time periods with death clusters"""
        clusters = {}
        for minute in death_minutes:
            # Group into 5-minute windows
            window = int(minute // 5) * 5
            clusters[window] = clusters.get(window, 0) + 1
        return clusters
    
    def _calculate_position_variance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate how spread out death positions are"""
        if len(positions) < 2:
            return 0
        
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_var = np.var(x_coords)
        y_var = np.var(y_coords)
        
        return x_var + y_var
    
    def _identify_danger_zones(self, deaths_data: List[Dict]) -> List[str]:
        """Identify the most dangerous zones based on death frequency"""
        zone_counts = {}
        for death in deaths_data:
            zone = death['zone']
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        # Sort by frequency and return top zones
        sorted_zones = sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)
        return [zone for zone, count in sorted_zones if count > 1]