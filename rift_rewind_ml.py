# Rift Rewind ML Analysis Module
# Integrates with your existing JavaScript analysis

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import json
import boto3
from dotenv import load_dotenv
load_dotenv()
import os
from typing import Dict, List, Optional, Tuple

class RiftRewindMLAnalyzer:
    """
    Advanced ML analysis for League of Legends player data
    Designed to work with your existing JavaScript analysis output
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        self.scaler = StandardScaler()
        self.models = {}
        self.aws_region = aws_region
        
        # Initialize AWS clients (same as your JS setup)
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
    def process_js_analysis_output(self, js_output: Dict) -> pd.DataFrame:
        """
        Convert your JavaScript analysis output to ML-ready DataFrame
        Expects the format from your analyzePlayerPerformance function
        """
        
        games_data = []
        
        for match in js_output.get('detailedMatches', []):
            # Basic match data
            game_data = {
                'champion': match.get('champion'),
                'role': match.get('role'),
                'win': match.get('win'),
                'kills': match.get('kills', 0),
                'deaths': match.get('deaths', 0),
                'assists': match.get('assists', 0),
                'cs': match.get('cs', 0),
                'damage_dealt': match.get('damage', 0),
                'vision_score': match.get('visionScore', 0),
                'game_length': match.get('gameLength', 1800),
            }
            
            # Advanced metrics from your deep analysis
            deep_analysis = js_output.get('deepAnalysis', {})
            
            # Warding data
            warding = deep_analysis.get('wardingPatterns', {})
            game_data.update({
                'wards_per_game': float(warding.get('averageWardsPerGame', 0)),
                'early_warding_ratio': warding.get('earlyWarding', 0) / max(warding.get('totalWards', 1), 1),
                'late_warding_ratio': warding.get('lateWarding', 0) / max(warding.get('totalWards', 1), 1),
            })
            
            # Positioning insights
            positioning = deep_analysis.get('positioningInsights', {})
            total_deaths = positioning.get('totalDeaths', 1)
            game_data.update({
                'dangerous_death_rate': positioning.get('dangerousDeaths', 0) / max(total_deaths, 1),
                'overextension_rate': positioning.get('laneOverextensions', 0) / max(total_deaths, 1),
                'teamfight_success_rate': positioning.get('teamfightSuccess', 0) / max(positioning.get('teamfightParticipation', 1), 1),
                'successful_aggression': positioning.get('successfulAggression', 0),
            })
            
            # Efficiency metrics
            recall_eff = deep_analysis.get('recallEfficiency', {})
            item_timing = deep_analysis.get('itemTimingInsights', {})
            game_data.update({
                'recall_efficiency': float(recall_eff.get('averageEfficiency', 50)),
                'item_timing_efficiency': float(item_timing.get('efficiency', 50)),
            })
            
            # Jungle-specific (if applicable)
            jungle_perf = deep_analysis.get('junglePerformance', {})
            if jungle_perf and jungle_perf.get('totalGanks', 0) > 0:
                game_data.update({
                    'gank_efficiency': float(jungle_perf.get('gankEfficiency', 0)),
                    'ganks_per_game': float(jungle_perf.get('averageGanks', 0)),
                    'early_gank_ratio': jungle_perf.get('earlyGanks', 0) / max(jungle_perf.get('totalGanks', 1), 1),
                })
            
            games_data.append(game_data)
        
        df = pd.DataFrame(games_data)
        return self._engineer_features(df)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for ML analysis"""
        
        # Basic calculated features
        df['kda_ratio'] = (df['kills'] + df['assists']) / np.maximum(df['deaths'], 1)
        df['cs_per_minute'] = df['cs'] / (df['game_length'] / 60)
        df['damage_per_minute'] = df['damage_dealt'] / (df['game_length'] / 60)
        df['kill_participation'] = (df['kills'] + df['assists']) / np.maximum(df['kills'] + df['assists'] + df['deaths'], 1)
        
        # Advanced efficiency metrics
        df['macro_efficiency'] = (df['recall_efficiency'] + df['item_timing_efficiency']) / 2
        df['positioning_skill'] = 100 - (df['dangerous_death_rate'] * 100)
        df['teamfight_impact'] = df['teamfight_success_rate'] * df['successful_aggression']
        
        # Role-specific features
        df['role_performance'] = df.apply(self._calculate_role_performance, axis=1)
        
        return df
    
    def _calculate_role_performance(self, row) -> float:
        """Calculate role-specific performance score"""
        role = row.get('role', 'UNKNOWN')
        
        if role == 'ADC':
            return (row['damage_per_minute'] / 10) + (row['cs_per_minute'] * 5) + (row['kda_ratio'] * 10)
        elif role == 'JUNGLE':
            gank_score = row.get('gank_efficiency', 50) / 10
            return gank_score + (row['kda_ratio'] * 8) + (row['successful_aggression'] * 3)
        elif role == 'SUPPORT':
            return (row['vision_score'] / 2) + (row['kda_ratio'] * 5) + (row['teamfight_success_rate'] * 20)
        elif role in ['MID', 'TOP']:
            return (row['damage_per_minute'] / 12) + (row['cs_per_minute'] * 4) + (row['kda_ratio'] * 8)
        else:
            return row['kda_ratio'] * 10
    
    def analyze_playstyle_clusters(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Identify distinct playstyles using clustering
        Returns dataframe with cluster labels and cluster descriptions
        """
        
        # Features for clustering
        cluster_features = [
            'kda_ratio', 'cs_per_minute', 'damage_per_minute',
            'positioning_skill', 'teamfight_impact', 'successful_aggression',
            'macro_efficiency', 'early_warding_ratio'
        ]
        
        # Prepare data
        feature_data = df[cluster_features].fillna(df[cluster_features].mean())
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Perform clustering
        n_clusters = min(6, len(df) // 3)  # Adjust based on data size
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['playstyle_cluster'] = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        cluster_profiles = {}
        for cluster in range(n_clusters):
            cluster_data = df[df['playstyle_cluster'] == cluster]
            
            cluster_profiles[cluster] = {
                'size': len(cluster_data),
                'win_rate': cluster_data['win'].mean(),
                'avg_kda': cluster_data['kda_ratio'].mean(),
                'avg_positioning': cluster_data['positioning_skill'].mean(),
                'avg_macro': cluster_data['macro_efficiency'].mean(),
                'primary_roles': cluster_data['role'].value_counts().to_dict(),
                'characteristics': self._describe_cluster(cluster_data)
            }
        
        self.models['playstyle_kmeans'] = kmeans
        return df, cluster_profiles
    
    def _describe_cluster(self, cluster_data: pd.DataFrame) -> str:
        """Generate human-readable cluster description"""
        avg_kda = cluster_data['kda_ratio'].mean()
        avg_positioning = cluster_data['positioning_skill'].mean()
        avg_aggression = cluster_data['successful_aggression'].mean()
        win_rate = cluster_data['win'].mean()
        
        if avg_kda > 4 and avg_positioning > 70:
            return "Conservative Carry - High KDA with safe positioning"
        elif avg_aggression > 3 and win_rate > 0.6:
            return "Aggressive Playmaker - High risk, high reward style"
        elif avg_positioning < 60 and win_rate < 0.4:
            return "Risky Player - Needs positioning improvement"
        elif cluster_data['macro_efficiency'].mean() > 70:
            return "Macro Specialist - Strong game knowledge and efficiency"
        else:
            return "Balanced Player - Well-rounded gameplay style"
    
    def predict_performance_improvement(self, df: pd.DataFrame) -> Dict:
        """
        Train models to predict performance improvements
        """
        
        # Prepare features for prediction
        feature_columns = [
            'cs_per_minute', 'positioning_skill', 'macro_efficiency',
            'early_warding_ratio', 'teamfight_impact', 'role_performance'
        ]
        
        X = df[feature_columns].fillna(df[feature_columns].mean())
        
        # Multiple target predictions
        predictions = {}
        
        # 1. Win probability prediction
        y_win = df['win']
        win_model = RandomForestClassifier(n_estimators=100, random_state=42)
        win_model.fit(X, y_win)
        predictions['win_probability'] = {
            'model': win_model,
            'feature_importance': dict(zip(feature_columns, win_model.feature_importances_))
        }
        
        # 2. KDA prediction
        y_kda = df['kda_ratio']
        kda_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        kda_model.fit(X, y_kda)
        predictions['kda_prediction'] = {
            'model': kda_model,
            'feature_importance': dict(zip(feature_columns, kda_model.feature_importances_))
        }
        
        self.models.update(predictions)
        return predictions
    
    def generate_personalized_insights(self, df: pd.DataFrame, cluster_profiles: Dict, player_cluster: int) -> List[str]:
        """
        Generate personalized improvement suggestions based on ML analysis
        """
        
        insights = []
        player_data = df[df['playstyle_cluster'] == player_cluster]
        
        if len(player_data) == 0:
            return ["Insufficient data for personalized analysis"]
        
        # Cluster-specific insights
        cluster_info = cluster_profiles[player_cluster]
        avg_stats = player_data.mean()
        
        insights.append(f"**Playstyle Analysis**: You're classified as a '{cluster_info['characteristics']}'")
        insights.append(f"**Cluster Performance**: Your playstyle has a {cluster_info['win_rate']:.1%} average win rate")
        
        # Specific improvement areas
        if avg_stats['positioning_skill'] < 60:
            insights.append("🚨 **Critical**: Your positioning needs immediate attention - 40%+ of deaths in dangerous areas")
            insights.append("💡 **Tip**: Focus on ward placement and map awareness before engaging")
        
        if avg_stats['macro_efficiency'] < 60:
            insights.append("⏰ **Macro Game**: Your recall timing and item efficiency can be improved")
            insights.append("💡 **Tip**: Back when you have 800+ gold and can push wave to enemy tower")
        
        if avg_stats['cs_per_minute'] < 5:
            insights.append("⚔️ **Farming**: Work on last-hitting and wave management")
            insights.append("💡 **Tip**: Aim for 7+ CS/min in laning phase")
        
        # Role-specific insights
        primary_role = player_data['role'].mode().iloc[0] if len(player_data['role'].mode()) > 0 else 'UNKNOWN'
        
        if primary_role == 'JUNGLE':
            if avg_stats.get('gank_efficiency', 0) < 40:
                insights.append("🌟 **Jungle Focus**: Your gank success rate is below average")
                insights.append("💡 **Tip**: Wait for enemy cooldowns or coordinate with laners before ganking")
        
        elif primary_role == 'ADC':
            if avg_stats['damage_per_minute'] < 500:
                insights.append("🎯 **Damage Output**: Your damage per minute is low for ADC")
                insights.append("💡 **Tip**: Focus on positioning to maximize DPS uptime in fights")
        
        return insights
    
    async def generate_ai_coaching_report(self, insights: List[str], player_data: Dict) -> str:
        """
        Use AWS Bedrock to generate comprehensive coaching report
        """
        
        try:
            prompt = f"""You are a professional League of Legends coach. Based on advanced machine learning analysis of {player_data.get('totalGames', 0)} games, provide a comprehensive coaching report.

ANALYSIS RESULTS:
{chr(10).join(insights)}

PLAYER STATS:
- Total Games: {player_data.get('totalGames', 0)}
- Win Rate: {player_data.get('winRate', 0)}%
- Primary Role: {player_data.get('primaryRole', 'Unknown')}

Provide a structured coaching report with:
1. Overall performance assessment
2. Top 3 priority improvement areas
3. Specific practice recommendations
4. Short-term goals (next 10 games)
5. Long-term development path

Make it actionable and specific to their playstyle."""

            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                contentType="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "temperature": 0.7,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
            
        except Exception as e:
            return f"AI coaching analysis temporarily unavailable. Raw insights: {' | '.join(insights)}"

# Usage example function
def main():
    """Example usage of the ML analyzer"""
    analyzer = RiftRewindMLAnalyzer()
    
    # This would be called from your JavaScript with the analysis output
    print("Rift Rewind ML Analyzer initialized successfully!")
    print("Ready to process JavaScript analysis output...")
    
    # Example of how to use:
    # 1. Your JS code runs analyzePlayerPerformance()
    # 2. Pass that output to analyzer.process_js_analysis_output()
    # 3. Run ML analysis with the methods above
    # 4. Return insights back to your frontend

if __name__ == "__main__":
    main()