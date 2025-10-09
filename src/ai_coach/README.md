# AI Coach - AWS Bedrock Integration

Personalized League of Legends coaching powered by AWS Bedrock (Claude 3).

## Overview

This module uses AWS Bedrock's Claude 3 AI model to generate personalized coaching insights from Riot API match data. Instead of complex ML training for ward detection, we leverage Claude's natural language understanding to analyze player performance and provide actionable feedback.

## Features

- **Match Performance Analysis**: AI-powered insights for individual matches
- **Comprehensive Coaching**: Performance trend analysis across multiple games
- **Cost-Effective**: ~$0.04/month for typical usage (vs. $17/day for SageMaker)
- **No Training Required**: Works immediately with Riot API data
- **Fallback Mode**: Local coaching insights when AWS is unavailable

## Architecture

```
Riot API → Match Data → Bedrock (Claude 3) → Coaching Insights
```

**Key Components:**
- `bedrock_coaching_insights.py`: Main Bedrock integration
- `enhanced_coaching.py`: Advanced coaching analytics

## Setup

### 1. AWS Credentials

Create `.env` file with:
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
RIOT_API_KEY=your_riot_api_key
```

### 2. Install Dependencies

```bash
pip install boto3 python-dotenv
```

### 3. AWS Bedrock Access

Enable Claude 3 Sonnet in AWS Bedrock console (us-east-1 region).

## Usage

### Basic Example

```python
from ai_coach.bedrock_coaching_insights import BedrockCoachingInsights

# Initialize
coach = BedrockCoachingInsights()

# Analyze a match
match_data = {
    'championName': 'Ahri',
    'kills': 8,
    'deaths': 4,
    'assists': 12,
    'totalMinionsKilled': 198,
    'visionScore': 45,
    'win': True
}

insights = coach.generate_match_coaching(match_data)
print(insights['insights'])
```

### Full Demo

Run the complete demo with Riot API integration:

```bash
python demo_ai_coaching.py
```

This will:
1. Fetch recent matches from Riot API
2. Generate AI coaching for the latest match
3. Analyze performance trends across multiple games
4. Provide comprehensive improvement recommendations

## API Reference

### `BedrockCoachingInsights`

**Methods:**

- `generate_match_coaching(match_data: Dict) -> Dict`
  - Generate coaching for a single match
  - Input: Riot API match participant data
  - Output: AI-generated insights with metadata

- `generate_gameplay_coaching(match_history: List[Dict]) -> Dict`
  - Generate comprehensive coaching from match history
  - Input: List of recent match data
  - Output: Performance trends and improvement plan

**Private Methods:**

- `_invoke_bedrock_model(prompt: str) -> str`
  - Calls Claude 3 Sonnet via Bedrock
  - Handles fallback to local insights if unavailable

## Cost Management

**AWS Bedrock Pricing:**
- Input: $0.003 per 1K tokens
- Output: $0.015 per 1K tokens

**Typical Usage:**
- Single match analysis: ~2K tokens = $0.03
- Monthly cost (10 analyses): ~$0.30

**Comparison:**
- SageMaker ML Endpoints: $0.71/hour = $510/month
- Bedrock AI Coaching: $0.04/month (basic usage)
- **Savings: 99.99%**

## Why We Pivoted from ML

**Original Plan:**
- TensorFlow/YOLOv5 for ward detection
- SageMaker endpoint deployment
- Temporal tracking system

**Challenges:**
1. Training data collection difficulty
2. High AWS costs ($0.71/hour)
3. Complex deployment pipeline
4. Ward detection accuracy issues

**Current Solution:**
- Simple Riot API integration
- Claude 3 for natural language coaching
- Minimal infrastructure costs
- Immediate insights without training

## Project Evolution

```
❌ ML Ward Detection (Too Complex)
   ↓
✅ AI Coaching (Simple + Effective)
```

**Deprecated Components:**
- All ML/computer vision code → `_deprecated/`
- SageMaker endpoints → deleted
- Temporal tracking → deprecated

**Active Components:**
- Bedrock AI coaching ✅
- Riot API client ✅
- Simple demo system ✅

## Future Enhancements

- [ ] Champion-specific coaching prompts
- [ ] Rank prediction based on performance
- [ ] Team composition analysis
- [ ] Patch-aware meta insights
- [ ] Discord bot integration

## Files

- `bedrock_coaching_insights.py` - Main Bedrock integration
- `enhanced_coaching.py` - Advanced analytics (WIP)
- `../demo_ai_coaching.py` - Complete demo script
- `README.md` - This file

## License

MIT License - See project root for details

---

**Built with:** AWS Bedrock, Claude 3 Sonnet, Riot Games API
**Status:** ✅ Production Ready (Demo Phase)
