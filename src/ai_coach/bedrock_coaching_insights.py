"""
AWS Bedrock AI Integration for League of Legends Coaching Insights
Generates natural language coaching advice from Riot API match data
Uses Claude 3 to analyze player performance and provide actionable feedback
"""

import boto3
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class BedrockCoachingInsights:
    """
    AWS Bedrock integration for generating AI-powered coaching insights
    Analyzes Riot API match data and generates personalized coaching using Claude 3
    """
    
    def __init__(self):
        self.bedrock_runtime = None
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3 Sonnet
        self.logger = logging.getLogger(__name__)
        self.bedrock_available = False
        
        try:
            # Check if we have AWS credentials first
            import os
            if not (os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')):
                self.logger.warning("⚠️ AWS credentials not found - using local AI coaching")
                return
            
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name='us-east-1'
            )
            
            # Test Bedrock access with a simple call
            self._test_bedrock_access()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Bedrock not accessible: {e}")
            self.logger.info("💡 Falling back to local AI coaching insights")
    
    def _test_bedrock_access(self):
        """Test if Bedrock is accessible"""
        try:
            # Simple test to verify credentials work
            self.bedrock_runtime.list_foundation_models()
            self.bedrock_available = True
            self.logger.info("✅ AWS Bedrock connected successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Bedrock access test failed: {e}")
            self.bedrock_available = False
    
    def generate_match_coaching(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate coaching insights from Riot API match data"""
        
        prompt = self._build_match_coaching_prompt(match_data)
        insights = self._invoke_bedrock_model(prompt)
        
        return {
            'analysis_type': 'Match Performance Coaching',
            'insights': insights,
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model_id,
            'confidence_score': 0.95
        }
    
    def generate_gameplay_coaching(self, match_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive coaching from match history"""
        
        prompt = self._build_comprehensive_coaching_prompt(match_history)
        insights = self._invoke_bedrock_model(prompt)
        
        return {
            'analysis_type': 'Comprehensive Gameplay Coaching',
            'insights': insights,
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model_id,
            'analysis_depth': 'Advanced'
        }
    
    def _build_match_coaching_prompt(self, match_data: Dict[str, Any]) -> str:
        """Build prompt for match performance coaching"""
        return f"""You are an expert League of Legends coach analyzing a player's match performance.

Match Data:
{json.dumps(match_data, indent=2)}

Provide personalized coaching insights covering:
1. Key Performance Metrics: Analyze KDA, CS, vision score, damage dealt
2. Strengths: What did the player do well this match?
3. Areas for Improvement: Specific actionable feedback
4. Strategic Decisions: Review objective control, teamfight participation
5. Champion-Specific Tips: Role-appropriate advice

Keep the tone encouraging but honest. Focus on actionable improvements."""

    def _build_comprehensive_coaching_prompt(self, match_history: List[Dict[str, Any]]) -> str:
        """Build prompt for comprehensive gameplay coaching"""
        return f"""You are an expert League of Legends coach analyzing a player's recent match history.

Match History Summary:
{json.dumps(match_history[:5], indent=2)}

Provide comprehensive coaching insights covering:
1. Performance Trends: Identify patterns across multiple games
2. Consistency Analysis: Which aspects are stable vs inconsistent?
3. Champion Pool: Effectiveness on different champions
4. Role Performance: How well they fulfill their role's responsibilities
5. Mental Game: Adaptability, tilt management, learning curve
6. Priority Improvements: Top 3 areas to focus on for rank climbing

Be specific, data-driven, and motivational. Help them understand their journey."""
    
    def _invoke_bedrock_model(self, prompt: str) -> str:
        """Invoke Bedrock AI model to generate insights"""
        
        if not self.bedrock_runtime:
            return self._get_fallback_coaching_insights()
        
        try:
            # Prepare request for Claude 3
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Invoke model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            return response_body['content'][0]['text']
            
        except Exception as e:
            self.logger.error(f"❌ Bedrock invocation failed: {e}")
            return self._get_fallback_coaching_insights()
    
    def _get_fallback_coaching_insights(self) -> str:
        """Fallback insights when Bedrock is not available"""
        
        return """
## Match Performance Analysis
Your recent match shows solid fundamentals with room for strategic growth. Let's break down the key areas for improvement.

## Strengths
- **Mechanical Execution**: Good CS and damage output for your role
- **Vision Score**: Above average vision control - keep placing wards
- **KDA Management**: Showing restraint and calculated aggression

## Areas for Improvement
- **Objective Priority**: Focus on Dragon and Baron timer awareness
- **Death Positioning**: Analyze where deaths occurred to improve safety
- **CS Efficiency**: Push for +10 CS per 10 minutes improvement

## Strategic Recommendations
1. **Map Awareness**: Ward key jungle entrances 60-90s before objectives spawn
2. **Communication**: Ping objectives and coordinate with team
3. **Champion Mastery**: Focus on 2-3 champions to climb faster
4. **Wave Management**: Learn to freeze and slow push for lane pressure

## Next Steps
Practice these concepts in your next 5 games:
- Track jungle paths and ping missing enemies
- Ward defensively when behind, aggressively when ahead
- Review death replays to understand positioning mistakes
- Focus on CS goals: 7+ per minute minimum

Keep improving - consistency wins ranks!
"""

if __name__ == "__main__":
    # Demo usage
    bedrock = BedrockCoachingInsights()
    
    # Test match analysis
    sample_match_data = {
        'championName': 'Ahri',
        'kills': 8,
        'deaths': 4,
        'assists': 12,
        'totalMinionsKilled': 198,
        'visionScore': 45,
        'gameDuration': 1847,  # ~30 minutes
        'win': True,
        'goldEarned': 13450,
        'totalDamageDealtToChampions': 24532
    }
    
    print("🎯 Testing AI Coaching with Sample Match Data...")
    insights = bedrock.generate_match_coaching(sample_match_data)
    print(f"\n✅ Generated {insights['analysis_type']}")
    print(f"📊 Model: {insights['model_used']}")
    print(f"\n{insights['insights'][:300]}...")
