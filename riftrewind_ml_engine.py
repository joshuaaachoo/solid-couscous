# RiftRewind Pro - Core ML Engine
# Advanced machine learning for League of Legends player analysis

from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import boto3
import json
from typing import Dict, List, Optional, Tuple

class RiftRewindMLEngine:
    """
    Advanced ML engine for League of Legends player analysis
    Combines multiple models for comprehensive player insights
    """
    
    def __init__(self):
        self.playstyle_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.win_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.performance_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Playstyle cluster definitions
        self.cluster_names = {
            0: "Aggressive Carry",
            1: "Defensive Support", 
            2: "Consistent Farmer",
            3: "High-Risk Playmaker",
            4: "Team-Oriented Utility"
        }
        
        self.cluster_descriptions = {
            0: "High damage, aggressive positioning, focuses on eliminations",
            1: "Prioritizes team safety, vision control, and utility",
            2: "Strong farming, consistent performance, reliable scaling",
            3: "High variance plays, game-changing potential, feast or famine",
            4: "Team-first mentality, enables teammates, objective-focused"
        }
    
    def extract_advanced_features(self, match_data: Dict) -> Dict:
        """Extract comprehensive ML features from match data"""
        
        features = {}
        
        # Core performance metrics
        features['kills'] = match_data.get('kills', 0)
        features['deaths'] = max(match_data.get('deaths', 1), 1)  # Avoid division by zero
        features['assists'] = match_data.get('assists', 0)
        features['kda_ratio'] = (features['kills'] + features['assists']) / features['deaths']
        
        # Farming and economy
        features['total_cs'] = match_data.get('totalMinionsKilled', 0) + match_data.get('neutralMinionsKilled', 0)
        features['cs_per_minute'] = features['total_cs'] / max(match_data.get('gameDuration', 1800) / 60, 1)
        features['gold_earned'] = match_data.get('goldEarned', 0)
        features['gold_per_minute'] = features['gold_earned'] / max(match_data.get('gameDuration', 1800) / 60, 1)
        
        # Combat effectiveness
        features['damage_dealt'] = match_data.get('totalDamageDealtToChampions', 0)
        features['damage_per_minute'] = features['damage_dealt'] / max(match_data.get('gameDuration', 1800) / 60, 1)
        features['damage_taken'] = match_data.get('totalDamageTaken', 0)
        features['damage_ratio'] = features['damage_dealt'] / max(features['damage_taken'], 1)
        
        # Vision and map control
        features['vision_score'] = match_data.get('visionScore', 0)
        features['vision_per_minute'] = features['vision_score'] / max(match_data.get('gameDuration', 1800) / 60, 1)
        features['wards_placed'] = match_data.get('wardsPlaced', 0)
        features['wards_killed'] = match_data.get('wardsKilled', 0)
        
        # Objective participation
        features['turret_kills'] = match_data.get('turretKills', 0)
        features['objective_score'] = (
            match_data.get('dragonKills', 0) * 2 +
            match_data.get('baronKills', 0) * 3 +
            features['turret_kills']
        )
        
        # Meta information
        features['champion_id'] = match_data.get('championId', 0)
        features['role'] = self._encode_role(match_data.get('teamPosition', 'UTILITY'))
        features['win'] = int(match_data.get('win', False))
        features['game_duration'] = match_data.get('gameDuration', 1800) / 60  # Convert to minutes
        
        # Advanced calculated metrics
        features['kill_participation'] = (features['kills'] + features['assists']) / max(
            match_data.get('teamKills', 1), 1
        )
        features['death_rate'] = features['deaths'] / features['game_duration']
        features['efficiency_score'] = (
            features['kda_ratio'] * 0.3 +
            features['cs_per_minute'] * 0.2 +
            (features['damage_per_minute'] / 1000) * 0.3 +
            features['vision_per_minute'] * 0.2
        )
        
        return features
    
    def _encode_role(self, role: str) -> int:
        """Encode role string to integer for ML models"""
        role_mapping = {
            'TOP': 0,
            'JUNGLE': 1, 
            'MIDDLE': 2,
            'BOTTOM': 3,
            'UTILITY': 4,
            'SUPPORT': 4  # Alias for UTILITY
        }
        return role_mapping.get(role, 4)
    
    def _decode_role(self, role_int: int) -> str:
        """Decode role integer back to string"""
        role_mapping = {0: 'TOP', 1: 'JUNGLE', 2: 'MIDDLE', 3: 'BOTTOM', 4: 'UTILITY'}
        return role_mapping.get(role_int, 'UTILITY')
    
    def create_player_profile(self, matches: List[Dict]) -> Dict:
        """Create comprehensive player profile from match history"""
        
        if not matches:
            return {"error": "No matches provided"}
        
        # Extract features for all matches
        match_features = []
        for match in matches:
            features = self.extract_advanced_features(match)
            match_features.append(features)
        
        df = pd.DataFrame(match_features)
        
        # Basic statistics
        profile = {
            'total_games': len(matches),
            'wins': df['win'].sum(),
            'losses': len(matches) - df['win'].sum(),
            'win_rate': df['win'].mean(),
            
            # Performance averages
            'avg_kda': df['kda_ratio'].mean(),
            'avg_cs_per_min': df['cs_per_minute'].mean(),
            'avg_damage_per_min': df['damage_per_minute'].mean(),
            'avg_gold_per_min': df['gold_per_minute'].mean(),
            'avg_vision_per_min': df['vision_per_minute'].mean(),
            
            # Meta information
            'primary_role': self._decode_role(df['role'].mode().iloc[0]) if len(df) > 0 else 'UTILITY',
            'champion_diversity': df['champion_id'].nunique(),
            'avg_game_length': df['game_duration'].mean(),
            
            # Recent form (last 10 games)
            'recent_form': df.tail(10)['win'].mean() if len(df) >= 10 else df['win'].mean(),
            'recent_kda': df.tail(10)['kda_ratio'].mean() if len(df) >= 10 else df['kda_ratio'].mean(),
            
            # Consistency metrics
            'kda_consistency': 1 / (1 + df['kda_ratio'].std()),  # Inverse of std, higher = more consistent
            'performance_consistency': 1 / (1 + df['efficiency_score'].std()),
            
            # Advanced metrics
            'avg_kill_participation': df['kill_participation'].mean(),
            'avg_death_rate': df['death_rate'].mean(),
            'avg_efficiency_score': df['efficiency_score'].mean(),
        }
        
        # Role distribution
        role_counts = df['role'].value_counts()
        profile['role_distribution'] = {
            self._decode_role(int(role)): count for role, count in role_counts.items()
        }
        
        # Champion performance breakdown
        champion_stats = df.groupby('champion_id').agg({
            'win': ['count', 'mean'],
            'kda_ratio': 'mean',
            'cs_per_minute': 'mean',
            'damage_per_minute': 'mean'
        }).round(3)
        
        profile['champion_performance'] = {}
        for champ_id in champion_stats.index:
            games_played = champion_stats.loc[champ_id, ('win', 'count')]
            if games_played >= 3:  # Only include champions with 3+ games
                profile['champion_performance'][int(champ_id)] = {
                    'games': int(games_played),
                    'win_rate': champion_stats.loc[champ_id, ('win', 'mean')],
                    'avg_kda': champion_stats.loc[champ_id, ('kda_ratio', 'mean')],
                    'avg_cs_per_min': champion_stats.loc[champ_id, ('cs_per_minute', 'mean')],
                    'avg_damage_per_min': champion_stats.loc[champ_id, ('damage_per_minute', 'mean')]
                }
        
        # Performance trends (if enough games)
        if len(matches) >= 20:
            # Split into early and recent halves
            mid_point = len(df) // 2
            early_games = df.iloc[:mid_point]
            recent_games = df.iloc[mid_point:]
            
            profile['improvement_trends'] = {
                'win_rate_change': recent_games['win'].mean() - early_games['win'].mean(),
                'kda_change': recent_games['kda_ratio'].mean() - early_games['kda_ratio'].mean(),
                'cs_change': recent_games['cs_per_minute'].mean() - early_games['cs_per_minute'].mean(),
                'damage_change': recent_games['damage_per_minute'].mean() - early_games['damage_per_minute'].mean(),
            }
        
        return profile
    
    def train_playstyle_classifier(self, player_profiles: List[Dict]) -> Dict:
        """Train K-means clustering model to identify playstyles"""
        
        if len(player_profiles) < 5:
            return {"error": "Need at least 5 player profiles for clustering"}
        
        # Prepare feature matrix
        features = []
        valid_profiles = []
        
        for profile in player_profiles:
            if profile.get('total_games', 0) >= 10:  # Minimum games threshold
                feature_vector = [
                    profile.get('avg_kda', 1.0),
                    profile.get('avg_cs_per_min', 5.0),
                    profile.get('avg_damage_per_min', 500),
                    profile.get('avg_vision_per_min', 1.0),
                    profile.get('champion_diversity', 5),
                    profile.get('kda_consistency', 0.5),
                    profile.get('avg_kill_participation', 0.5),
                    profile.get('avg_death_rate', 0.2),
                    profile.get('recent_form', 0.5)
                ]
                features.append(feature_vector)
                valid_profiles.append(profile)
        
        if len(features) < 5:
            return {"error": "Not enough valid profiles for clustering"}
        
        # Normalize features
        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train clustering model
        cluster_labels = self.playstyle_model.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(5):
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_profiles = [p for i, p in enumerate(valid_profiles) if cluster_labels[i] == cluster_id]
                cluster_features = X[cluster_mask]
                
                cluster_analysis[cluster_id] = {
                    'name': self.cluster_names[cluster_id],
                    'description': self.cluster_descriptions[cluster_id],
                    'size': int(np.sum(cluster_mask)),
                    'avg_stats': {
                        'kda': float(np.mean(cluster_features[:, 0])),
                        'cs_per_min': float(np.mean(cluster_features[:, 1])),
                        'damage_per_min': float(np.mean(cluster_features[:, 2])),
                        'vision_per_min': float(np.mean(cluster_features[:, 3])),
                        'champion_diversity': float(np.mean(cluster_features[:, 4])),
                        'win_rate': float(np.mean([p['win_rate'] for p in cluster_profiles]))
                    }
                }
        
        return {
            'success': True,
            'clusters': cluster_analysis,
            'total_profiles': len(valid_profiles)
        }
    
    def predict_playstyle(self, player_profile: Dict) -> Dict:
        """Predict player's playstyle cluster"""
        
        feature_vector = np.array([[
            player_profile.get('avg_kda', 1.0),
            player_profile.get('avg_cs_per_min', 5.0),
            player_profile.get('avg_damage_per_min', 500),
            player_profile.get('avg_vision_per_min', 1.0),
            player_profile.get('champion_diversity', 5),
            player_profile.get('kda_consistency', 0.5),
            player_profile.get('avg_kill_participation', 0.5),
            player_profile.get('avg_death_rate', 0.2),
            player_profile.get('recent_form', 0.5)
        ]])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict cluster
        cluster = self.playstyle_model.predict(feature_vector_scaled)[0]
        
        return {
            'cluster_id': int(cluster),
            'playstyle_name': self.cluster_names.get(cluster, 'Unknown'),
            'description': self.cluster_descriptions.get(cluster, 'Unknown playstyle'),
        }
    
    def generate_coaching_insights(self, player_profile: Dict, playstyle: Dict) -> List[str]:
        """Generate personalized coaching recommendations"""
        
        insights = []
        
        # Performance-based insights
        if player_profile.get('avg_kda', 1.0) < 2.0:
            insights.append("🎯 **Positioning Focus**: Your KDA suggests frequent deaths. Practice safer positioning in team fights and avoid overextending.")
            
        if player_profile.get('avg_cs_per_min', 5.0) < 6.0:
            insights.append("⚔️ **Farming Improvement**: Aim for 7+ CS/min by practicing last-hitting in practice tool and improving wave management.")
            
        if player_profile.get('avg_vision_per_min', 1.0) < 1.2:
            insights.append("👁️ **Vision Control**: Increase ward placement and map awareness. Vision wins games!")
            
        if player_profile.get('avg_kill_participation', 0.5) < 0.6:
            insights.append("🤝 **Team Fighting**: Work on being present for more team fights and skirmishes to help your team.")
            
        # Win rate based insights
        win_rate = player_profile.get('win_rate', 0.5)
        if win_rate < 0.45:
            insights.append("📈 **Fundamentals**: Focus on the basics - farming, positioning, and objective control will improve your win rate.")
        elif win_rate > 0.6:
            insights.append("🔥 **Hot Streak**: You're performing well! Consider playing more games to climb while you're in good form.")
            
        # Consistency insights
        if player_profile.get('kda_consistency', 0.5) < 0.6:
            insights.append("📊 **Consistency**: Your performance varies significantly between games. Focus on maintaining your best practices every game.")
            
        # Champion pool insights
        diversity = player_profile.get('champion_diversity', 5)
        if diversity > 15:
            insights.append("🎭 **Champion Focus**: You play many different champions. Consider focusing on 3-5 champions to master them.")
        elif diversity < 3:
            insights.append("🎪 **Expand Pool**: Your champion pool is narrow. Learning 2-3 more champions can help in different situations.")
            
        # Playstyle-specific insights
        cluster_insights = {
            0: "Your aggressive style can carry games, but work on knowing when to back off to avoid feeding streaks.",
            1: "Your supportive play is valuable - practice shot-calling and team coordination to maximize impact.",
            2: "Your consistent farming is strong - focus on translating gold leads into map pressure and objectives.",
            3: "Your high-risk plays can be game-changing - improve risk assessment to maximize success rate.",
            4: "Your team-first approach is excellent - practice carrying when your team falls behind."
        }
        
        cluster_id = playstyle.get('cluster_id', 0)
        if cluster_id in cluster_insights:
            insights.append(f"🎨 **{playstyle.get('playstyle_name', 'Your')} Style**: {cluster_insights[cluster_id]}")
            
        # Recent form insights
        recent_form = player_profile.get('recent_form', 0.5)
        if recent_form > player_profile.get('win_rate', 0.5) + 0.1:
            insights.append("⬆️ **Improving**: Your recent games show significant improvement. Keep up the good work!")
        elif recent_form < player_profile.get('win_rate', 0.5) - 0.1:
            insights.append("⬇️ **Recent Struggles**: Your recent performance has dipped. Take a break or review your recent games to identify issues.")
            
        return insights[:7]  # Limit to 7 most relevant insights
    
    def save_models(self, filepath_prefix: str):
        """Save trained models to disk"""
        joblib.dump(self.playstyle_model, f"{filepath_prefix}_playstyle_model.pkl")
        joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")
        
    def load_models(self, filepath_prefix: str):
        """Load trained models from disk"""
        self.playstyle_model = joblib.load(f"{filepath_prefix}_playstyle_model.pkl")
        self.scaler = joblib.load(f"{filepath_prefix}_scaler.pkl")

# Example usage and testing
if __name__ == "__main__":
    # Initialize ML engine
    ml_engine = RiftRewindMLEngine()
    
    # Sample match data for testing
    sample_matches = [
        {
            'kills': 8, 'deaths': 3, 'assists': 12, 'win': True,
            'totalMinionsKilled': 180, 'neutralMinionsKilled': 20,
            'goldEarned': 15000, 'totalDamageDealtToChampions': 25000,
            'visionScore': 25, 'wardsPlaced': 12, 'championId': 67,
            'teamPosition': 'BOTTOM', 'gameDuration': 1800,
            'turretKills': 2, 'dragonKills': 1
        },
        {
            'kills': 4, 'deaths': 7, 'assists': 8, 'win': False,
            'totalMinionsKilled': 160, 'neutralMinionsKilled': 15,
            'goldEarned': 12000, 'totalDamageDealtToChampions': 18000,
            'visionScore': 20, 'wardsPlaced': 8, 'championId': 67,
            'teamPosition': 'BOTTOM', 'gameDuration': 1650,
            'turretKills': 1, 'dragonKills': 0
        }
    ]
    
    # Test profile creation
    profile = ml_engine.create_player_profile(sample_matches)
    print("Player Profile:", json.dumps(profile, indent=2))
    
    # Test coaching insights
    playstyle = {'cluster_id': 0, 'playstyle_name': 'Aggressive Carry', 'description': 'High damage focus'}
    insights = ml_engine.generate_coaching_insights(profile, playstyle)
    print("\nCoaching Insights:")
    for insight in insights:
        print(f"  {insight}")