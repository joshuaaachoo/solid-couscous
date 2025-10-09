# 🎮 RiftRewind: AI-Powered League of Legends Coach

> Personalized gameplay coaching powered by AWS Bedrock AI and Riot Games API

[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange)](https://aws.amazon.com/bedrock/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![Claude 3](https://img.shields.io/badge/Claude-3%20Sonnet-purple)](https://www.anthropic.com/claude)

## 🎯 Overview

RiftRewind analyzes your League of Legends match history and generates personalized coaching insights using AWS Bedrock's Claude 3 AI model. Get expert-level feedback on your gameplay, identify improvement areas, and climb the ranked ladder faster.

**Key Features:**
- 🤖 AI-powered coaching with Claude 3 Sonnet
- 📊 Match performance analysis from Riot API
- 📈 Performance trend tracking across games
- 💰 Cost-effective solution (~$0.04/month)
- 🎯 Actionable, personalized improvement plans

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Riot API   │───▶│ Match Data   │───▶│ AWS Bedrock     │
│  (Live)     │    │ Processing   │    │ (Claude 3)      │
└─────────────┘    └──────────────┘    └─────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │ Coaching Insights    │
                                    │ - Strengths          │
                                    │ - Weaknesses         │
                                    │ - Action Items       │
                                    └──────────────────────┘
```

## ✨ What You Get

### Single Match Analysis
- KDA and CS performance breakdown
- Vision control assessment
- Champion-specific tips
- Win condition analysis

### Comprehensive Coaching
- Performance trends across multiple games
- Champion pool effectiveness
- Role mastery evaluation
- Rank improvement roadmap

### AI-Generated Insights
- Natural language coaching advice
- Specific, actionable recommendations
- Encouraging but honest feedback
- Strategic and tactical guidance

## 🚀 Quick Start

### Prerequisites

```bash
# Required
- Python 3.8+
- Riot Games API Key (free from developer.riotgames.com)

# Optional (for AI coaching)
- AWS Account with Bedrock access
- AWS credentials configured
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/RiftRewind.git
cd RiftRewind

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration

Edit `.env`:
```bash
# Required
RIOT_API_KEY=your_riot_api_key_here

# Optional (for AI coaching)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1
```

### Run Demo

```bash
python demo_ai_coaching.py
```

This will:
1. Fetch your recent matches from Riot API
2. Analyze your latest match performance
3. Generate AI coaching insights
4. Show performance trends across games

## 📊 Example Output

```
🎯 MATCH SUMMARY
═══════════════════════════════════════════════════════════
Champion: Ahri
KDA: 8/4/12 (5.0 KDA)
CS: 198 (6.6 CS/min)
Vision Score: 45
Result: Victory
Duration: 30 minutes
═══════════════════════════════════════════════════════════

🤖 AI COACHING INSIGHTS
═══════════════════════════════════════════════════════════
## Strengths
- Excellent KDA management with minimal deaths
- Strong vision control for your role
- Good objective participation

## Areas for Improvement
- CS could be higher - aim for 7+ CS/min
- Death positioning needs work (check replays)
- Ward timing around objectives

## Action Items
1. Practice last-hitting in practice tool
2. Review death locations - were you overextended?
3. Ward 60-90s before Drake/Baron spawns
═══════════════════════════════════════════════════════════
```

## 📁 Project Structure

```
RiftRewind/
├── src/
│   ├── ai_coach/              # AI coaching module
│   │   ├── bedrock_coaching_insights.py
│   │   ├── enhanced_coaching.py
│   │   └── README.md
│   ├── api/                   # Riot API client
│   │   └── riot_api_client.py
│   └── core/                  # Core utilities
├── demo_ai_coaching.py        # Demo script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 💰 Cost Analysis

### AWS Bedrock Pricing
- **Input tokens**: $0.003 per 1K tokens
- **Output tokens**: $0.015 per 1K tokens
- **Typical analysis**: ~2K tokens = $0.03

**Monthly Cost:**
- 10 match analyses: ~$0.30
- 100 match analyses: ~$3.00
- Average user: **~$0.04/month**

### Comparison with ML Approach

| Solution | Monthly Cost | Setup Time | Accuracy |
|----------|-------------|------------|----------|
| SageMaker ML | ~$510 | Weeks | Uncertain |
| Bedrock AI | ~$0.04 | Minutes | Excellent |
| **Savings** | **99.99%** | **Instant** | **Better** |

## 🔧 API Reference

### BedrockCoachingInsights

```python
from ai_coach.bedrock_coaching_insights import BedrockCoachingInsights

coach = BedrockCoachingInsights()

# Analyze single match
insights = coach.generate_match_coaching(match_data)

# Analyze match history
comprehensive = coach.generate_gameplay_coaching(match_history)
```

**Methods:**

- `generate_match_coaching(match_data: Dict) -> Dict`
  - Analyzes single match performance
  - Returns AI-generated insights with metadata

- `generate_gameplay_coaching(match_history: List[Dict]) -> Dict`
  - Analyzes performance trends across multiple games
  - Returns comprehensive improvement plan

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint
flake8 src/
```

## 📖 Documentation

- [AI Coach Module](src/ai_coach/README.md) - Detailed Bedrock integration docs
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Architecture](docs/TECHNICAL_ARCHITECTURE.md) - System design details

## 🗺️ Roadmap

- [x] Basic match analysis
- [x] AI coaching with Claude 3
- [x] Performance trend tracking
- [ ] Champion-specific coaching
- [ ] Rank prediction
- [ ] Discord bot integration
- [ ] Web dashboard
- [ ] Team composition analysis

## 🤔 Why Not ML/Computer Vision?

**Original Plan:** TensorFlow ward detection with SageMaker endpoints

**Why We Pivoted:**
1. ❌ Training data collection too difficult
2. ❌ AWS SageMaker costs ($0.71/hour = $510/month)
3. ❌ Complex deployment pipeline
4. ❌ Ward detection accuracy challenges

**Current Solution:**
1. ✅ Simple Riot API integration
2. ✅ Claude 3 for natural language coaching
3. ✅ Minimal costs (~$0.04/month)
4. ✅ Better insights, no training needed

**Result:** 99.99% cost reduction + better user experience

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Riot Games for the excellent API
- AWS Bedrock team for Claude 3 access
- Anthropic for Claude's AI capabilities

## 📧 Contact

Questions? Open an issue or reach out!

---

**Built with:** AWS Bedrock • Claude 3 Sonnet • Riot Games API • Python

**Status:** ✅ Production Ready (Demo Phase)
