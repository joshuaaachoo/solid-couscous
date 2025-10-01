# RiftRewind Pro - AWS Bedrock AI Insights Generator
# Natural language insights using Claude 3.5 Sonnet

from dotenv import load_dotenv
load_dotenv()
import os
import boto3
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

class BedrockInsightsGenerator:
    """
    Advanced AI insights generator using Amazon Bedrock
    Creates personalized coaching and season summaries
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
    def generate_season_summary(self, player_data: Dict, ml_insights: Dict) -> str:
        """Generate personalized season recap using Claude"""
        
        player_info = player_data.get('player_info', {})
        matches = player_data.get('matches', [])
        
        # Calculate additional stats
        total_games = len(matches)
        wins = sum(1 for m in matches if m.get('win', False))
        win_rate = wins / max(total_games, 1)
        
        # Get champion stats
        champion_games = {}
        for match in matches:
            champ_name = match.get('championName', 'Unknown')
            if champ_name not in champion_games:
                champion_games[champ_name] = {'games': 0, 'wins': 0}
            champion_games[champ_name]['games'] += 1
            if match.get('win', False):
                champion_games[champ_name]['wins'] += 1
        
        # Find best and most played champions
        most_played = max(champion_games.items(), key=lambda x: x[1]['games']) if champion_games else ('Unknown', {'games': 0, 'wins': 0})
        best_champ = max(champion_games.items(), key=lambda x: x[1]['wins'] / max(x[1]['games'], 1)) if champion_games else ('Unknown', {'games': 0, 'wins': 0})
        
        playstyle = ml_insights.get('playstyle', {})
        coaching_tips = ml_insights.get('coaching_insights', [])
        
        prompt = f"""You are a professional League of Legends coach writing a personalized season recap for {player_info.get('name', 'a player')}. 

SEASON STATS:
• Total Games: {total_games}
• Wins: {wins} | Losses: {total_games - wins}
• Win Rate: {win_rate:.1%}
• Level: {player_info.get('summoner_level', 'Unknown')}

CHAMPION PERFORMANCE:
• Most Played: {most_played[0]} ({most_played[1]['games']} games, {most_played[1]['wins'] / max(most_played[1]['games'], 1):.1%} WR)
• Best Champion: {best_champ[0]} ({best_champ[1]['wins']}/{best_champ[1]['games']} games, {best_champ[1]['wins'] / max(best_champ[1]['games'], 1):.1%} WR)

PLAYSTYLE ANALYSIS:
• Type: {playstyle.get('playstyle_name', 'Balanced Player')}
• Description: {playstyle.get('description', 'Well-rounded gameplay')}

KEY INSIGHTS:
{chr(10).join('• ' + tip.replace('🎯 **', '').replace('**', '').replace('⚔️ **', '').replace('👁️ **', '').replace('🤝 **', '').replace('📈 **', '').replace('🔥 **', '').replace('📊 **', '').replace('🎭 **', '').replace('🎪 **', '').replace('🎨 **', '').replace('⬆️ **', '').replace('⬇️ **', '') for tip in coaching_tips[:4])}

Write an engaging, personalized season story that:
1. Opens with a compelling hook about their journey
2. Celebrates their key achievements and growth moments
3. Acknowledges challenges they've overcome
4. Provides specific, actionable next steps for improvement
5. Ends with motivation for the upcoming season
6. Uses a warm, encouraging coaching tone
7. Makes it feel like their unique story, not generic advice

Write 250-350 words. Be specific to their data, inspirational, and forward-looking."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 600,
                    "temperature": 0.7,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
            
        except Exception as e:
            logging.error(f"Error generating season summary: {e}")
            return f"Season Analysis for {player_info.get('name', 'Player')}: Analyzed {total_games} games with a {win_rate:.1%} win rate. Focus on the improvement areas identified in your coaching insights to climb higher next season!"
    
    def generate_champion_mastery_insights(self, player_data: Dict) -> str:
        """Generate insights about champion performance and recommendations"""
        
        matches = player_data.get('matches', [])
        
        # Analyze champion performance
        champion_stats = {}
        for match in matches:
            champ_id = match.get('championId')
            champ_name = match.get('championName', f'Champion_{champ_id}')
            
            if champ_name not in champion_stats:
                champion_stats[champ_name] = {
                    'games': 0, 'wins': 0, 'kills': 0, 'deaths': 0, 'assists': 0,
                    'damage': 0, 'cs': 0
                }
            
            stats = champion_stats[champ_name]
            stats['games'] += 1
            if match.get('win', False):
                stats['wins'] += 1
            stats['kills'] += match.get('kills', 0)
            stats['deaths'] += match.get('deaths', 0)
            stats['assists'] += match.get('assists', 0)
            stats['damage'] += match.get('totalDamageDealtToChampions', 0)
            stats['cs'] += match.get('totalMinionsKilled', 0) + match.get('neutralMinionsKilled', 0)
        
        # Find champions with 3+ games
        significant_champs = {name: stats for name, stats in champion_stats.items() if stats['games'] >= 3}
        
        if not significant_champs:
            return "Play more games with individual champions to get detailed champion mastery insights!"
        
        # Calculate averages and find best/worst
        for stats in significant_champs.values():
            stats['win_rate'] = stats['wins'] / stats['games']
            stats['avg_kda'] = (stats['kills'] + stats['assists']) / max(stats['deaths'], 1)
            stats['avg_damage'] = stats['damage'] / stats['games']
            stats['avg_cs'] = stats['cs'] / stats['games']
        
        best_champ = max(significant_champs.items(), key=lambda x: x[1]['win_rate'])
        worst_champ = min(significant_champs.items(), key=lambda x: x[1]['win_rate'])
        highest_kda = max(significant_champs.items(), key=lambda x: x[1]['avg_kda'])
        
        prompt = f"""Analyze this player's champion mastery and provide strategic recommendations:

CHAMPION PERFORMANCE DATA:
• Best Win Rate: {best_champ[0]} - {best_champ[1]['win_rate']:.1%} WR ({best_champ[1]['wins']}/{best_champ[1]['games']} games)
• Lowest Win Rate: {worst_champ[0]} - {worst_champ[1]['win_rate']:.1%} WR ({worst_champ[1]['wins']}/{worst_champ[1]['games']} games)  
• Highest KDA: {highest_kda[0]} - {highest_kda[1]['avg_kda']:.2f} KDA

CHAMPION POOL SIZE: {len(significant_champs)} champions with 3+ games

Provide specific, actionable advice about:
1. Which champions to focus on for climbing
2. Which champions to practice more or avoid
3. Champion pool optimization strategy
4. Role/position recommendations based on their champion success

Keep it concise but insightful (150-200 words). Focus on practical advice they can immediately apply."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 400,
                    "temperature": 0.6,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
            
        except Exception as e:
            logging.error(f"Error generating champion insights: {e}")
            return "Focus on mastering your highest win rate champions while practicing mechanics on your weaker picks in normals before bringing them to ranked."
    
    def generate_improvement_roadmap(self, player_data: Dict, ml_insights: Dict) -> str:
        """Generate a personalized improvement plan with specific steps"""
        
        coaching_tips = ml_insights.get('coaching_insights', [])
        playstyle = ml_insights.get('playstyle', {})
        
        # Calculate current performance level
        matches = player_data.get('matches', [])
        if matches:
            avg_kda = sum((m.get('kills', 0) + m.get('assists', 0)) / max(m.get('deaths', 1), 1) for m in matches) / len(matches)
            avg_cs_per_min = sum((m.get('totalMinionsKilled', 0) + m.get('neutralMinionsKilled', 0)) / max(m.get('gameDuration', 1800) / 60, 1) for m in matches) / len(matches)
            win_rate = sum(1 for m in matches if m.get('win', False)) / len(matches)
        else:
            avg_kda, avg_cs_per_min, win_rate = 1.0, 5.0, 0.5
        
        prompt = f"""Create a personalized 30-day improvement roadmap for this League of Legends player:

CURRENT PERFORMANCE:
• Win Rate: {win_rate:.1%}
• Average KDA: {avg_kda:.2f}
• CS per minute: {avg_cs_per_min:.1f}
• Playstyle: {playstyle.get('playstyle_name', 'Balanced')}

KEY AREAS FOR IMPROVEMENT:
{chr(10).join('• ' + tip.replace('🎯 **', '').replace('**', '').replace('⚔️ **', '').replace('👁️ **', '').replace('🤝 **', '').replace('📈 **', '').replace('🔥 **', '').replace('📊 **', '').replace('🎭 **', '').replace('🎪 **', '').replace('🎨 **', '').replace('⬆️ **', '').replace('⬇️ **', '') for tip in coaching_tips[:3])}

Create a structured 30-day plan with:

**Week 1-2 (Foundation):**
- Daily practice routine (15-20 min)
- 2-3 specific drills/exercises
- Focus areas for ranked games

**Week 3-4 (Advanced):**
- Advanced concepts to work on
- Review and analysis practices
- Goal-setting for next month

Make it actionable with specific times, rep counts, or measurable targets. Include both mechanical practice and game knowledge improvements. Keep it realistic and achievable (200-250 words)."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.6,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
            
        except Exception as e:
            logging.error(f"Error generating improvement roadmap: {e}")
            return "Focus on your fundamentals: practice CS in training tool for 10 minutes daily, review your deaths in recent games, and maintain consistent vision control. Small improvements compound over time!"
    
    def generate_social_comparison(self, player_data: Dict, comparison_data: Optional[Dict] = None) -> str:
        """Generate social comparison insights vs friends or average players"""
        
        if not comparison_data:
            return "Add friends to see how you compare with your gaming squad!"
        
        player_stats = self._calculate_player_stats(player_data)
        comparison_stats = comparison_data.get('average_stats', {})
        friend_name = comparison_data.get('friend_name', 'your friends')
        
        prompt = f"""Compare this player's performance with {friend_name}:

YOUR STATS:
• Win Rate: {player_stats.get('win_rate', 0):.1%}
• KDA: {player_stats.get('avg_kda', 1.0):.2f}
• CS/min: {player_stats.get('avg_cs_per_min', 5.0):.1f}
• Damage/min: {player_stats.get('avg_damage_per_min', 500):.0f}

{friend_name.upper()} AVERAGE:
• Win Rate: {comparison_stats.get('win_rate', 0.5):.1%}
• KDA: {comparison_stats.get('avg_kda', 2.0):.2f}  
• CS/min: {comparison_stats.get('avg_cs_per_min', 6.0):.1f}
• Damage/min: {comparison_stats.get('avg_damage_per_min', 600):.0f}

Write a fun, friendly comparison that:
1. Celebrates areas where they're ahead
2. Identifies areas to catch up
3. Suggests friendly competition goals
4. Maintains a positive, motivational tone

Keep it light and engaging (100-150 words)."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json", 
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
            
        except Exception as e:
            logging.error(f"Error generating social comparison: {e}")
            return f"You're performing well compared to {friend_name}! Keep playing together and pushing each other to improve."
    
    def _calculate_player_stats(self, player_data: Dict) -> Dict:
        """Calculate summary statistics for a player"""
        
        matches = player_data.get('matches', [])
        if not matches:
            return {}
        
        total_games = len(matches)
        wins = sum(1 for m in matches if m.get('win', False))
        
        total_kills = sum(m.get('kills', 0) for m in matches)
        total_deaths = sum(max(m.get('deaths', 1), 1) for m in matches)
        total_assists = sum(m.get('assists', 0) for m in matches)
        
        total_cs = sum((m.get('totalMinionsKilled', 0) + m.get('neutralMinionsKilled', 0)) for m in matches)
        total_damage = sum(m.get('totalDamageDealtToChampions', 0) for m in matches)
        total_time = sum(m.get('gameDuration', 1800) for m in matches) / 60  # Convert to minutes
        
        return {
            'win_rate': wins / total_games,
            'avg_kda': (total_kills + total_assists) / total_deaths,
            'avg_cs_per_min': total_cs / total_time,
            'avg_damage_per_min': total_damage / total_time,
            'total_games': total_games
        }
    
    def generate_complete_insights_package(self, player_data: Dict, ml_insights: Dict, comparison_data: Optional[Dict] = None) -> Dict:
        """Generate all AI insights in one comprehensive package"""
        
        try:
            insights_package = {
                'season_summary': self.generate_season_summary(player_data, ml_insights),
                'champion_mastery': self.generate_champion_mastery_insights(player_data),
                'improvement_roadmap': self.generate_improvement_roadmap(player_data, ml_insights),
                'social_comparison': self.generate_social_comparison(player_data, comparison_data) if comparison_data else None,
                'generated_at': datetime.now().isoformat(),
                'player_name': player_data.get('player_info', {}).get('name', 'Unknown Player')
            }
            
            return insights_package
            
        except Exception as e:
            logging.error(f"Error generating complete insights package: {e}")
            return {
                'season_summary': f"Season analysis complete for {player_data.get('player_info', {}).get('name', 'player')}. Focus on consistency and improvement!",
                'champion_mastery': "Continue practicing with your best champions while expanding your champion pool.",
                'improvement_roadmap': "Focus on fundamentals: farming, positioning, and map awareness.",
                'social_comparison': None,
                'generated_at': datetime.now().isoformat(),
                'error': str(e)
            }

# AWS Lambda integration
def lambda_handler(event, context):
    """Lambda function for generating AI insights"""
    
    try:
        body = json.loads(event.get('body', '{}'))
        player_data = body.get('playerData')
        ml_insights = body.get('mlInsights')
        comparison_data = body.get('comparisonData')
        
        if not player_data or not ml_insights:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Missing required data'})
            }
        
        # Generate insights
        generator = BedrockInsightsGenerator()
        insights = generator.generate_complete_insights_package(
            player_data, ml_insights, comparison_data
        )
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(insights)
        }
        
    except Exception as e:
        logging.error(f"Lambda error: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }

# Example usage
if __name__ == "__main__":
    # Test data
    sample_player_data = {
        'player_info': {'name': 'TestPlayer#NA1', 'summoner_level': 150},
        'matches': [
            {'championName': 'Jinx', 'win': True, 'kills': 8, 'deaths': 3, 'assists': 12, 
             'totalMinionsKilled': 180, 'gameDuration': 1800, 'totalDamageDealtToChampions': 25000},
            {'championName': 'Jinx', 'win': False, 'kills': 4, 'deaths': 7, 'assists': 6,
             'totalMinionsKilled': 160, 'gameDuration': 1650, 'totalDamageDealtToChampions': 18000}
        ]
    }
    
    sample_ml_insights = {
        'playstyle': {'playstyle_name': 'Aggressive Carry', 'description': 'High damage focus'},
        'coaching_insights': [
            "🎯 **Focus on positioning**: Your deaths suggest positioning needs work",
            "⚔️ **Improve CS**: Aim for higher creep score per minute"
        ]
    }
    
    generator = BedrockInsightsGenerator()
    
    # Test season summary
    summary = generator.generate_season_summary(sample_player_data, sample_ml_insights)
    print("Season Summary:")
    print(summary)
    print("\n" + "="*50 + "\n")
    
    # Test champion insights
    champion_insights = generator.generate_champion_mastery_insights(sample_player_data)
    print("Champion Insights:")
    print(champion_insights)