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
from botocore.exceptions import NoCredentialsError, ClientError, EndpointConnectionError

class BedrockInsightsGenerator:
    """
    Advanced AI insights generator using Amazon Bedrock
    Creates personalized coaching and season summaries
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        self.bedrock = None
        self.credentials_valid = False
        
        # Initialize Bedrock client with proper error handling
        self._initialize_bedrock()
    
    def _initialize_bedrock(self):
        """Initialize Bedrock client with credential validation"""
        try:
            # Set AWS credentials from environment if available
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_REGION', self.region)
            
            if aws_access_key and aws_secret_key:
                # Create session with explicit credentials
                session = boto3.Session(
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                self.bedrock = session.client('bedrock-runtime')
                logging.info(f"✅ AWS credentials loaded from environment")
            else:
                # Try default credential chain
                self.bedrock = boto3.client('bedrock-runtime', region_name=self.region)
                logging.info(f"✅ Using default AWS credential chain")
            
            # Test connection
            self.credentials_valid = self._test_bedrock_access()
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize Bedrock: {e}")
            self.bedrock = None
            self.credentials_valid = False
    
    def _test_bedrock_access(self) -> bool:
        """Test if Bedrock access is working"""
        if not self.bedrock:
            return False
            
        try:
            # Test with a minimal request
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": "Test"}]
                })
            )
            
            result = json.loads(response['body'].read())
            if result.get('content'):
                logging.info("✅ Bedrock access test successful")
                return True
                
        except NoCredentialsError:
            logging.error("❌ AWS credentials not found or invalid")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'UnauthorizedOperation':
                logging.error("❌ AWS credentials lack Bedrock permissions")
            elif error_code == 'AccessDeniedException':
                logging.error("❌ Access denied to Bedrock service")
            else:
                logging.error(f"❌ AWS API error: {e}")
        except EndpointConnectionError:
            logging.error("❌ Cannot connect to Bedrock endpoint")
        except Exception as e:
            logging.error(f"❌ Bedrock access test failed: {e}")
            
        return False
    
    def _invoke_bedrock_model(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Safely invoke Bedrock model with fallback"""
        if not self.credentials_valid or not self.bedrock:
            return self._generate_fallback_response(prompt, max_tokens)
        
        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
            
        except Exception as e:
            logging.error(f"❌ Bedrock invocation failed: {e}")
            return self._generate_fallback_response(prompt, max_tokens)
    
    def _generate_fallback_response(self, prompt: str, max_tokens: int) -> str:
        """Generate fallback response when Bedrock is unavailable"""
        if "season" in prompt.lower():
            return "Season analysis complete! Focus on improving your weakest areas and maintaining consistency with your best champions. Review your recent matches to identify patterns and work on your fundamentals."
        elif "champion" in prompt.lower():
            return "Focus on mastering 2-3 champions that you have the highest win rate with. Practice their mechanics in normals before taking them to ranked games."
        elif "improvement" in prompt.lower():
            return "Work on these fundamentals daily: CS practice (10 min), vision control, and positioning. Small improvements compound over time!"
        elif "comparison" in prompt.lower():
            return "Keep playing with your friends and pushing each other to improve! Friendly competition helps everyone get better."
        else:
            return "Analysis complete! Focus on consistency, practice fundamentals, and review your replays to identify improvement opportunities."
    
    def generate_precision_coaching_insights(self, vod_analysis: Dict, ml_insights: Dict) -> str:
        """Generate enhanced precision coaching based on detailed VOD analysis"""
        
        # Extract precision metrics from enhanced analysis
        positioning_score = vod_analysis.get('positioning_accuracy', 85)
        ward_efficiency = vod_analysis.get('ward_efficiency_score', 88) 
        decision_timing = vod_analysis.get('decision_timing_score', 79)
        team_coordination = vod_analysis.get('team_coordination_rating', 77)
        frames_analyzed = vod_analysis.get('frames_to_analyze', 3600)
        
        micro_analysis = vod_analysis.get('micro_analysis', {})
        early_game = micro_analysis.get('early_game_execution', {}).get('score', 85)
        mid_game = micro_analysis.get('mid_game_transitions', {}).get('score', 79) 
        late_game = micro_analysis.get('late_game_precision', {}).get('score', 82)
        
        prompt = f"""You are an elite League of Legends coach analyzing MAXIMUM PRECISION VOD data with {frames_analyzed:,} frames at 3fps. Provide advanced micro-coaching based on these precision metrics:

🎯 PRECISION PERFORMANCE ANALYSIS:
• Positioning Accuracy: {positioning_score}% (Target: 85%+)
• Ward Efficiency Score: {ward_efficiency}% (Optimal timing analysis)
• Decision Timing Score: {decision_timing}% (Macro speed analysis) 
• Team Coordination Rating: {team_coordination}% (Communication timing)

📊 GAME PHASE BREAKDOWN:
• Early Game Execution: {early_game}/100 (0-15 minutes)
• Mid Game Transitions: {mid_game}/100 (15-25 minutes)
• Late Game Precision: {late_game}/100 (25+ minutes)

🔬 MICRO-ANALYSIS REQUEST:
Provide specific, measurable coaching improvements:

1. **Positioning Optimization**: Exact unit distances, timing windows
2. **Decision Speed Enhancement**: Specific reaction time improvements  
3. **Vision Mastery**: Precise ward timing and placement coordinates
4. **Team Coordination**: Exact communication timing for plays
5. **Micro-Improvement Targets**: Weekly measurable goals

Write professional coaching advice with specific numbers, timing windows, and actionable micro-improvements. Focus on precision metrics that can be tracked and measured. 300-400 words with tactical depth."""

        return self._invoke_bedrock_model(prompt, max_tokens=700, temperature=0.6)
    
    def get_connection_status(self) -> Dict[str, str]:
        """Get current AWS Bedrock connection status"""
        if self.credentials_valid:
            return {
                "status": "connected",
                "message": "✅ AWS Bedrock connection active",
                "model": self.model_id
            }
        else:
            return {
                "status": "disconnected", 
                "message": "❌ AWS Bedrock unavailable - using fallback responses",
                "help": "Check AWS credentials and Bedrock permissions"
            }
        
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
        
        # Calculate detailed stats for better coaching
        total_deaths = sum(m.get('deaths', 0) for m in matches)
        total_kills = sum(m.get('kills', 0) for m in matches)
        total_assists = sum(m.get('assists', 0) for m in matches)
        avg_kda = (total_kills + total_assists) / max(total_deaths, 1)
        
        total_cs = sum(m.get('totalMinionsKilled', 0) + m.get('neutralMinionsKilled', 0) for m in matches)
        total_time = sum(m.get('gameDuration', 1800) for m in matches) / 60
        cs_per_min = total_cs / max(total_time, 1)
        
        avg_vision = sum(m.get('visionScore', 0) for m in matches) / max(total_games, 1)
        avg_damage = sum(m.get('totalDamageDealtToChampions', 0) for m in matches) / max(total_games, 1)
        
        prompt = f"""You are a Diamond+ League of Legends coach analyzing {player_info.get('name', 'a player')}'s ranked performance. Give SPECIFIC, TACTICAL advice.

PERFORMANCE DATA:
• Games: {total_games} | Win Rate: {win_rate:.1%}
• KDA: {avg_kda:.2f} ({total_kills}K / {total_deaths}D / {total_assists}A total)
• CS/min: {cs_per_min:.1f} | Vision Score: {avg_vision:.1f}/game
• Avg Damage: {avg_damage:,.0f} per game
• Main Champion: {most_played[0]} ({most_played[1]['games']} games, {most_played[1]['wins'] / max(most_played[1]['games'], 1):.1%} WR)

Give brutally honest, specific coaching:

1. **KDA Analysis**: If KDA < 2.0, they're dying too much - explain WHY (facechecking? bad trades? positioning?). If CS/min < 6, they're missing too much farm - give specific farming drills.

2. **Win Rate Reality Check**: If WR < 45%, identify the core issue (champion pool? macro? mechanics?). If WR > 55%, tell them what's working and how to maintain it.

3. **Champion Pool**: If playing {most_played[0]}, give champion-specific advice (power spikes, combos, matchup tips). Should they one-trick or expand pool?

4. **Immediate Fixes** (Top 3):
   - Most impactful thing to improve THIS WEEK
   - Specific drill or practice method (with numbers)
   - How to measure success

5. **Rank Trajectory**: Based on stats, what rank should they aim for? What's the ONE stat holding them back?

No fluff. No motivation speeches. Just tactical League advice like you're reviewing VODs together. 200-250 words."""

        return self._invoke_bedrock_model(prompt, max_tokens=600, temperature=0.7)
    
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

        return self._invoke_bedrock_model(prompt, max_tokens=400, temperature=0.6)
    
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
        
        # Identify biggest weakness
        biggest_weakness = "farming" if avg_cs_per_min < 6 else "deaths" if avg_kda < 2.5 else "damage" if win_rate < 0.45 else "consistency"
        
        prompt = f"""Create a 30-day improvement plan. Be SPECIFIC with drills, numbers, and measurables.

CURRENT STATS:
• Win Rate: {win_rate:.1%} | KDA: {avg_kda:.2f} | CS/min: {avg_cs_per_min:.1f}
• Biggest Weakness: {biggest_weakness}

**WEEK 1-2: Fix The Obvious Flaw**

Daily Pre-Game Warm-up (10 min):
- If CS < 6/min: Practice Tool - 80 CS by 10min, 5 days straight
- If KDA < 2.5: Review last 3 deaths each game - WHY did you die?
- If low damage: Practice Tool - combo on dummy, 10 reps

Ranked Focus:
- Play ONLY 3-5 games per day to avoid tilt
- Specific goal: Improve weakest stat by 15%
- Track: CS, deaths, damage after each game

**WEEK 3-4: Advanced Concepts**

Macro Drills:
- Learn wave management (freeze/slow push/crash)
- Watch 1 VOD per day: Note YOUR mistakes (timestamp them)
- Jungle tracking: Ward pixel brush, track 3-camp start

Measurable Goals:
- CS/min: {avg_cs_per_min:.1f} → {avg_cs_per_min + 1:.1f}
- Deaths/game: Reduce by 2
- Vision score: Place 1.5 wards per minute
- Win rate target: {win_rate:.0%} → {min(win_rate + 0.10, 0.65):.0%}

**Daily Checklist:**
☐ 10min practice tool
☐ 3-5 ranked games (stop if 2 loss streak)
☐ Review 1 death replay
☐ Track stats in notepad

No bullshit. These are the exact drills that got people to climb. 200-250 words max."""

        return self._invoke_bedrock_model(prompt, max_tokens=500, temperature=0.6)
    
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

        return self._invoke_bedrock_model(prompt, max_tokens=300, temperature=0.7)
    
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
    
    def generate_enhanced_coaching_insights(self, player_data: Dict, ml_insights: Dict, coaching_analysis: Dict) -> Dict:
        """Generate comprehensive coaching insights based on enhanced analysis"""
        
        insights = {}
        
        try:
            # Tactical Coaching based on death patterns
            if coaching_analysis.get('tactical_insights'):
                tactical_data = coaching_analysis['tactical_insights']
                insights['tactical_coaching'] = self._generate_tactical_coaching(tactical_data)
            
            # Strategic Coaching based on macro patterns  
            if coaching_analysis.get('strategic_insights'):
                strategic_data = coaching_analysis['strategic_insights']
                insights['strategic_coaching'] = self._generate_strategic_coaching(strategic_data)
            
            # Progressive Development Plan
            if coaching_analysis.get('improvement_areas'):
                insights['development_plan'] = self._generate_development_plan(
                    coaching_analysis['improvement_areas'], tactical_data, strategic_data
                )
            
            # Comprehensive Performance Summary
            insights['performance_summary'] = self._generate_enhanced_performance_summary(
                player_data, ml_insights, coaching_analysis
            )
            
        except Exception as e:
            print(f"Error generating enhanced coaching insights: {e}")
            insights['error'] = "Unable to generate enhanced coaching insights"
        
        return insights
    
    def _generate_tactical_coaching(self, tactical_data: Dict) -> str:
        """Generate tactical coaching based on death pattern analysis"""
        
        total_deaths = tactical_data.get('total_deaths', 0)
        phase_dist = tactical_data.get('phase_distribution', {})
        zone_dist = tactical_data.get('zone_distribution', {})
        recommendations = tactical_data.get('tactical_recommendations', [])
        
        coaching = f"""🎯 **TACTICAL COACHING ANALYSIS**

**Death Pattern Summary:**
• Total Deaths Analyzed: {total_deaths}
• Early Game: {phase_dist.get('early', 0):.1f}% | Mid Game: {phase_dist.get('mid', 0):.1f}% | Late Game: {phase_dist.get('late', 0):.1f}%

**Problem Areas Identified:**
"""
        
        # Add zone-specific advice
        for zone, percentage in zone_dist.items():
            if percentage > 20:  # Significant death concentration
                if zone == 'enemy_jungle':
                    coaching += f"• **Enemy Jungle Deaths ({percentage:.1f}%)**: You're dying too often while invading. Ward deeper before entering enemy territory and always have an escape route planned.\n"
                elif zone == 'river':
                    coaching += f"• **River Deaths ({percentage:.1f}%)**: Vision control issues in river. Ward river brushes before rotating and avoid face-checking.\n"
                elif zone == 'lane':
                    coaching += f"• **Lane Deaths ({percentage:.1f}%)**: Laning phase positioning needs work. Respect enemy jungle gank timings and maintain proper minion wave positioning.\n"
        
        # Add phase-specific advice
        if phase_dist.get('late', 0) > 35:
            coaching += f"""
**🚨 Late Game Critical Issue:**
You're dying too much in late game ({phase_dist.get('late', 0):.1f}% of deaths). Late game deaths cost your team the most:
• Stay behind your frontline in teamfights
• Don't chase kills - protect carries instead
• Ward before checking objectives
• Group with team instead of splitting
"""
        
        coaching += "\n**Immediate Action Items:**\n"
        for i, rec in enumerate(recommendations[:3], 1):
            coaching += f"{i}. {rec}\n"
        
        return coaching
    
    def _generate_strategic_coaching(self, strategic_data: Dict) -> str:
        """Generate strategic coaching based on macro gameplay patterns"""
        
        time_clusters = strategic_data.get('death_timing_clusters', {})
        risk_percentage = strategic_data.get('risk_percentage', 0)
        recommendations = strategic_data.get('strategic_recommendations', [])
        
        coaching = f"""🧠 **STRATEGIC COACHING ANALYSIS**

**Macro Decision Making:**
• High-Risk Deaths: {risk_percentage:.1f}% of your deaths
• Death Timing Patterns: {len(time_clusters)} distinct danger windows identified

"""
        
        # Analyze timing clusters
        if time_clusters:
            most_dangerous = max(time_clusters.items(), key=lambda x: x[1])
            coaching += f"""**⏰ Critical Timing Window:**
• {most_dangerous[0]}-{most_dangerous[0]+5} minute mark: {most_dangerous[1]} deaths
• This is your most dangerous period - extra caution needed
• Likely causes: Power spike misunderstanding, objective setup errors

"""
        
        # Risk assessment advice
        if risk_percentage > 30:
            coaching += f"""**🎯 Risk Management Priority:**
Your high-risk death rate ({risk_percentage:.1f}%) suggests aggressive decision making without proper setup:

**Immediate Fixes:**
• Don't contest objectives without vision advantage
• Count enemy cooldowns before engaging
• Always have escape route planned
• Coordinate with team before risky plays
"""
        
        coaching += "\n**Strategic Recommendations:**\n"
        for i, rec in enumerate(recommendations, 1):
            coaching += f"{i}. {rec}\n"
        
        return coaching
    
    def _generate_development_plan(self, improvement_areas: List[str], tactical_data: Dict, strategic_data: Dict) -> str:
        """Generate 4-week progressive development plan"""
        
        plan = """📈 **4-WEEK PROGRESSIVE DEVELOPMENT PLAN**

"""
        
        # Prioritize improvement areas
        priority_areas = improvement_areas[:3]  # Top 3 issues
        
        for week in range(1, 5):
            plan += f"""**Week {week} Focus:**
"""
            
            if week == 1:  # Foundation
                plan += "**Foundation Building** - Awareness and Recognition\n"
                if priority_areas:
                    plan += f"• Primary Focus: {priority_areas[0]}\n"
                plan += "• Practice: Review replays daily, identify problem patterns\n"
                plan += "• Goal: Recognize dangerous situations 2 seconds earlier\n"
                
            elif week == 2:  # Skill Development
                plan += "**Skill Development** - Mechanical Improvements\n"
                if len(priority_areas) > 1:
                    plan += f"• Primary Focus: {priority_areas[1]}\n"
                plan += "• Practice: 15 minutes pre-game positioning drills\n"
                plan += "• Goal: Reduce deaths in primary problem area by 30%\n"
                
            elif week == 3:  # Application
                plan += "**Application & Practice** - In-Game Implementation\n"
                if len(priority_areas) > 2:
                    plan += f"• Primary Focus: {priority_areas[2]}\n"
                plan += "• Practice: Conscious decision-making in ranked games\n"
                plan += "• Goal: Apply new positioning consistently for 70% of games\n"
                
            else:  # Mastery
                plan += "**Mastery & Consistency** - Habit Formation\n"
                plan += "• Primary Focus: Integrate all improvements\n"
                plan += "• Practice: Full game awareness and positioning\n"
                plan += "• Goal: Demonstrate consistent improvement across all metrics\n"
            
            plan += "\n"
        
        plan += """**Success Metrics to Track:**
• Deaths per game (target: reduce by 2-3)
• Late game deaths (target: <30% of total)
• High-risk deaths (target: <20% of total)
• Consistent improvement across 5+ games

**Weekly Check-ins:**
Run RiftRewind analysis each week to track your progress and adjust focus areas."""
        
        return plan
    
    def _generate_enhanced_performance_summary(self, player_data: Dict, ml_insights: Dict, coaching_analysis: Dict) -> str:
        """Generate comprehensive performance summary with all analysis integrated"""
        
        player_info = player_data.get('player_info', {})
        matches = player_data.get('recent_matches', [])
        
        if not matches:
            return "No recent match data available for analysis."
        
        total_games = len(matches)
        wins = sum(1 for match in matches if match.get('win', False))
        win_rate = wins / total_games if total_games > 0 else 0
        
        tactical_insights = coaching_analysis.get('tactical_insights', {})
        strategic_insights = coaching_analysis.get('strategic_insights', {})
        improvement_areas = coaching_analysis.get('improvement_areas', [])
        
        summary = f"""🎮 **COMPREHENSIVE PERFORMANCE ANALYSIS**

**Player Overview:**
• Summoner: {player_info.get('name', 'Unknown')}#{player_info.get('tag', '')}
• Rank: {player_info.get('rank', 'Unranked')}
• Recent Performance: {wins}W/{total_games-wins}L ({win_rate:.1%} win rate)

**🎯 Tactical Assessment:**
• Death Analysis: {tactical_insights.get('total_deaths', 0)} deaths across {total_games} games
• Avg Deaths/Game: {tactical_insights.get('total_deaths', 0) / max(total_games, 1):.1f}
"""

        # Add phase analysis
        phase_dist = tactical_insights.get('phase_distribution', {})
        if phase_dist:
            summary += f"• Game Phase Deaths: Early {phase_dist.get('early', 0):.0f}% | Mid {phase_dist.get('mid', 0):.0f}% | Late {phase_dist.get('late', 0):.0f}%\n"

        summary += f"""
**🧠 Strategic Assessment:**
• Risk Management: {strategic_insights.get('risk_percentage', 0):.1f}% high-risk deaths
• Decision Making: {"Needs Focus" if strategic_insights.get('risk_percentage', 0) > 30 else "Good Control"}

**📊 Key Strengths:**"""

        # Identify strengths
        if phase_dist.get('late', 0) < 30:
            summary += "\n• Good late-game positioning and patience"
        if strategic_insights.get('risk_percentage', 0) < 25:
            summary += "\n• Strong risk assessment and decision making"
        if tactical_insights.get('total_deaths', 0) / max(total_games, 1) < 6:
            summary += "\n• Low death rate shows good survival instincts"

        summary += f"""

**⚠️ Priority Improvement Areas:**"""
        
        for i, area in enumerate(improvement_areas[:3], 1):
            summary += f"\n{i}. {area}"
        
        summary += f"""

**🎯 This Week's Focus:**
Based on your death pattern analysis, concentrate on:
1. **Immediate**: {improvement_areas[0] if improvement_areas else "General positioning improvement"}
2. **Secondary**: {"Objective positioning" if tactical_insights.get('objective_deaths') else "Game timing awareness"}
3. **Long-term**: {"Strategic decision making" if strategic_insights.get('risk_percentage', 0) > 25 else "Consistency in execution"}

Run another analysis in 5-10 games to track your improvement!"""
        
        return summary

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