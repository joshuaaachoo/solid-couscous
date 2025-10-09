"""
Progressive Performance Tracking System
Tracks improvement over time and sets personalized goals
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ProgressiveTracker:
    """Tracks player performance over time for progressive coaching"""
    
    def __init__(self, data_dir: str = "player_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_session_data(self, player_id: str, session_data: Dict[str, Any]) -> None:
        """Save analysis session for long-term tracking"""
        
        # Create player directory if it doesn't exist
        player_dir = os.path.join(self.data_dir, player_id)
        os.makedirs(player_dir, exist_ok=True)
        
        # Add timestamp to session data
        session_data['timestamp'] = datetime.now().isoformat()
        session_data['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Save session file
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(player_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, cls=NumpyEncoder)
    
    def get_player_history(self, player_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get player's historical performance data"""
        
        player_dir = os.path.join(self.data_dir, player_id)
        if not os.path.exists(player_dir):
            return []
        
        # Get all session files
        session_files = [f for f in os.listdir(player_dir) if f.startswith('session_') and f.endswith('.json')]
        sessions = []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for filename in session_files:
            filepath = os.path.join(player_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    session_data = json.load(f)
                
                # Check if session is within date range
                session_date = datetime.fromisoformat(session_data.get('timestamp', ''))
                if session_date >= cutoff_date:
                    sessions.append(session_data)
            
            except Exception as e:
                continue
        
        # Sort by timestamp
        sessions.sort(key=lambda x: x.get('timestamp', ''))
        return sessions
    
    def analyze_progression(self, player_id: str) -> Dict[str, Any]:
        """Analyze player's improvement over time"""
        
        history = self.get_player_history(player_id, days=90)  # 3 months of data
        
        if len(history) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 sessions for progression analysis",
                "sessions_found": len(history)
            }
        
        # Extract key metrics over time
        progression_data = {
            'death_rates': [],
            'tactical_scores': [],
            'strategic_scores': [],
            'improvement_areas': [],
            'timestamps': []
        }
        
        for session in history:
            coaching_data = session.get('coaching_analysis', {})
            tactical = coaching_data.get('tactical_insights', {})
            strategic = coaching_data.get('strategic_insights', {})
            
            # Calculate scores
            death_rate = tactical.get('total_deaths', 0) / max(session.get('matches_analyzed', 1), 1)
            tactical_score = self._calculate_tactical_score(tactical)
            strategic_score = self._calculate_strategic_score(strategic)
            
            progression_data['death_rates'].append(death_rate)
            progression_data['tactical_scores'].append(tactical_score)
            progression_data['strategic_scores'].append(strategic_score)
            progression_data['improvement_areas'].extend(coaching_data.get('improvement_areas', []))
            progression_data['timestamps'].append(session['timestamp'])
        
        # Calculate trends
        trends = self._calculate_trends(progression_data)
        
        # Generate improvement goals
        goals = self._generate_improvement_goals(progression_data, trends)
        
        # Calculate overall progression score
        overall_score = self._calculate_overall_progression(trends)
        
        return {
            'status': 'success',
            'progression_data': progression_data,
            'trends': trends,
            'improvement_goals': goals,
            'overall_progression': overall_score,
            'sessions_analyzed': len(history),
            'tracking_period_days': (datetime.fromisoformat(history[-1]['timestamp']) - 
                                   datetime.fromisoformat(history[0]['timestamp'])).days,
            'progressive_recommendations': self._generate_progressive_recommendations(trends, goals)
        }
    
    def _calculate_tactical_score(self, tactical_data: Dict) -> float:
        """Calculate tactical performance score (0-100)"""
        if not tactical_data:
            return 50.0
        
        score = 70.0  # Base score
        
        # Adjust based on phase distribution
        phase_dist = tactical_data.get('phase_distribution', {})
        late_game_deaths = phase_dist.get('late', 0)
        
        # Penalize high late game deaths (more costly)
        if late_game_deaths > 40:
            score -= (late_game_deaths - 40) * 0.5
        
        # Penalize high enemy jungle deaths (risky positioning)
        zone_dist = tactical_data.get('zone_distribution', {})
        enemy_jungle_deaths = zone_dist.get('enemy_jungle', 0)
        if enemy_jungle_deaths > 25:
            score -= (enemy_jungle_deaths - 25) * 0.3
        
        # Penalize objective deaths
        obj_deaths = sum(tactical_data.get('objective_deaths', {}).values())
        total_deaths = tactical_data.get('total_deaths', 1)
        obj_percentage = (obj_deaths / total_deaths) * 100
        if obj_percentage > 20:
            score -= (obj_percentage - 20) * 0.2
        
        return max(0, min(100, score))
    
    def _calculate_strategic_score(self, strategic_data: Dict) -> float:
        """Calculate strategic performance score (0-100)"""
        if not strategic_data:
            return 50.0
        
        score = 70.0  # Base score
        
        # Penalize high risk deaths
        risk_percentage = strategic_data.get('risk_percentage', 0)
        if risk_percentage > 30:
            score -= (risk_percentage - 30) * 0.4
        
        # Bonus for consistent timing (less chaotic deaths)
        if len(strategic_data.get('death_timing_clusters', {})) <= 2:
            score += 10  # Bonus for focused death timing
        
        return max(0, min(100, score))
    
    def _calculate_trends(self, progression_data: Dict) -> Dict[str, Any]:
        """Calculate improvement trends"""
        trends = {}
        
        for metric in ['death_rates', 'tactical_scores', 'strategic_scores']:
            values = progression_data[metric]
            if len(values) >= 2:
                # Calculate linear trend
                x = np.arange(len(values))
                trend_slope = np.polyfit(x, values, 1)[0]
                
                # Determine trend direction
                if metric == 'death_rates':
                    # For death rates, negative slope is good (fewer deaths)
                    trend_direction = 'improving' if trend_slope < -0.1 else 'declining' if trend_slope > 0.1 else 'stable'
                else:
                    # For scores, positive slope is good (higher scores)
                    trend_direction = 'improving' if trend_slope > 1 else 'declining' if trend_slope < -1 else 'stable'
                
                trends[metric] = {
                    'slope': trend_slope,
                    'direction': trend_direction,
                    'current_value': values[-1],
                    'starting_value': values[0],
                    'change': values[-1] - values[0]
                }
        
        return trends
    
    def _generate_improvement_goals(self, progression_data: Dict, trends: Dict) -> Dict[str, Any]:
        """Generate personalized improvement goals"""
        goals = {}
        
        # Death rate goals
        if 'death_rates' in trends:
            current_deaths = trends['death_rates']['current_value']
            if current_deaths > 7:
                goals['death_reduction'] = {
                    'target': max(5, current_deaths - 2),
                    'timeframe': '2 weeks',
                    'priority': 'high',
                    'description': f'Reduce deaths from {current_deaths:.1f} to {max(5, current_deaths - 2):.1f} per game'
                }
        
        # Tactical improvement goals
        if 'tactical_scores' in trends:
            current_tactical = trends['tactical_scores']['current_value']
            if current_tactical < 70:
                goals['tactical_improvement'] = {
                    'target': min(85, current_tactical + 10),
                    'timeframe': '3 weeks',
                    'priority': 'medium',
                    'description': f'Improve tactical score from {current_tactical:.1f} to {min(85, current_tactical + 10):.1f}'
                }
        
        # Strategic improvement goals  
        if 'strategic_scores' in trends:
            current_strategic = trends['strategic_scores']['current_value']
            if current_strategic < 70:
                goals['strategic_improvement'] = {
                    'target': min(85, current_strategic + 10),
                    'timeframe': '4 weeks',
                    'priority': 'medium',
                    'description': f'Improve strategic score from {current_strategic:.1f} to {min(85, current_strategic + 10):.1f}'
                }
        
        return goals
    
    def _calculate_overall_progression(self, trends: Dict) -> Dict[str, Any]:
        """Calculate overall progression assessment"""
        
        improving_trends = sum(1 for trend in trends.values() if trend.get('direction') == 'improving')
        declining_trends = sum(1 for trend in trends.values() if trend.get('direction') == 'declining')
        total_trends = len(trends)
        
        if total_trends == 0:
            return {'status': 'no_data', 'score': 0}
        
        improvement_ratio = improving_trends / total_trends
        
        if improvement_ratio >= 0.7:
            status = 'excellent_progress'
            score = 85 + (improvement_ratio - 0.7) * 50
        elif improvement_ratio >= 0.5:
            status = 'good_progress'
            score = 70 + (improvement_ratio - 0.5) * 75
        elif improvement_ratio >= 0.3:
            status = 'moderate_progress'
            score = 50 + (improvement_ratio - 0.3) * 100
        else:
            status = 'needs_focus'
            score = improvement_ratio * 167  # Scale to 0-50 range
        
        return {
            'status': status,
            'score': min(100, score),
            'improving_metrics': improving_trends,
            'declining_metrics': declining_trends,
            'stable_metrics': total_trends - improving_trends - declining_trends
        }
    
    def _generate_progressive_recommendations(self, trends: Dict, goals: Dict) -> List[str]:
        """Generate recommendations based on progression analysis"""
        recommendations = []
        
        # Based on trends
        declining_metrics = [metric for metric, data in trends.items() 
                           if data.get('direction') == 'declining']
        
        if 'death_rates' in declining_metrics:
            recommendations.append("🚨 Death rate increasing - focus on positioning fundamentals this week")
        
        if 'tactical_scores' in declining_metrics:
            recommendations.append("⚔️ Tactical performance declining - practice laning phase and teamfight positioning")
        
        if 'strategic_scores' in declining_metrics:
            recommendations.append("🧠 Strategic play needs work - study macro gameplay and objective control")
        
        # Based on goals
        if goals:
            priority_goals = [goal for goal in goals.values() if goal.get('priority') == 'high']
            if priority_goals:
                recommendations.append(f"🎯 Priority focus: {priority_goals[0]['description']}")
        
        # Positive reinforcement for improvements
        improving_metrics = [metric for metric, data in trends.items() 
                           if data.get('direction') == 'improving']
        
        if improving_metrics:
            recommendations.append(f"✅ Great improvement in {', '.join(improving_metrics)} - keep it up!")
        
        return recommendations
    
    def get_weekly_summary(self, player_id: str) -> Dict[str, Any]:
        """Generate weekly performance summary"""
        
        # Get last 7 days of data
        recent_history = self.get_player_history(player_id, days=7)
        
        if not recent_history:
            return {'status': 'no_recent_data'}
        
        # Calculate weekly stats
        total_matches = sum(session.get('matches_analyzed', 0) for session in recent_history)
        total_deaths = sum(session.get('coaching_analysis', {}).get('tactical_insights', {}).get('total_deaths', 0) 
                          for session in recent_history)
        
        avg_deaths_per_game = total_deaths / max(total_matches, 1)
        
        # Get most common improvement areas this week
        all_improvements = []
        for session in recent_history:
            all_improvements.extend(session.get('coaching_analysis', {}).get('improvement_areas', []))
        
        # Count improvement area frequency
        improvement_counts = {}
        for area in all_improvements:
            improvement_counts[area] = improvement_counts.get(area, 0) + 1
        
        top_improvement_areas = sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'status': 'success',
            'week_summary': {
                'sessions': len(recent_history),
                'total_matches': total_matches,
                'avg_deaths_per_game': avg_deaths_per_game,
                'top_improvement_areas': [area for area, count in top_improvement_areas],
                'consistency_rating': self._calculate_consistency_rating(recent_history)
            }
        }