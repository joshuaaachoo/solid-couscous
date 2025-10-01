# 🎮 RiftRewind: League of Legends AI Coach

An AI-powered League of Legends performance analyzer that provides comprehensive gameplay insights and beautiful death heatmap visualizations.

## ✨ Features

### 📊 Advanced Analytics
- **Player Performance Analysis**: Comprehensive statistics and performance metrics
- **ML-Powered Insights**: Playstyle classification and behavioral analysis
- **AI-Generated Coaching**: Natural language insights powered by AWS Bedrock

### 🗺️ Interactive Visualizations
- **Death Heatmaps**: Beautiful gradient overlays on Summoner's Rift map
- **Multiple Visualization Modes**: 
  - Map overlay with smooth KDE gradients
  - Density heatmaps with landmark references
  - Champion-specific death patterns
- **Real Map Integration**: Uses actual Summoner's Rift PNG background

### 🔧 Technical Features
- **Timeline Data Processing**: Extracts death locations from match timeline data
- **Multi-format Support**: Handles both raw API and processed match data
- **Gaussian KDE**: Smooth, continuous density estimation for natural-looking heatmaps
- **Streamlit Web Interface**: Clean, interactive web application

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Riot Games API Key
- AWS credentials (for AI insights)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd riftrewind
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application:
```bash
streamlit run app.py
```

## 🎯 Usage

1. **Enter Player Details**: Input game name and tag line
2. **Select Analysis Scope**: Choose number of matches to analyze (5-20)
3. **View Results**: Explore multiple visualization tabs:
   - **Map Overlay**: Death heatmap with smooth gradients on Summoner's Rift
   - **Density Heatmap**: Interactive Plotly visualization with landmarks
   - **By Champion**: Champion-specific death pattern analysis

## 🛠️ Architecture

### Core Components
- **`app.py`**: Streamlit web interface
- **`riftrewind_app.py`**: Main application orchestration
- **`riot_api_client.py`**: Riot Games API integration
- **`visualization.py`**: Death heatmap generation and processing
- **`riftrewind_ml_engine.py`**: ML analysis and playstyle classification
- **`bedrock_insights.py`**: AI-powered natural language insights

### Key Technologies
- **Streamlit**: Web interface framework
- **Matplotlib + PIL**: High-quality image generation and processing
- **Scipy**: Gaussian KDE for smooth density estimation
- **Plotly**: Interactive visualizations
- **AWS Bedrock**: AI-powered insights generation
- **Riot Games API**: Match and timeline data

## 📊 Heatmap Technology

The death heatmap visualization uses advanced techniques for professional-quality results:

- **Kernel Density Estimation (KDE)**: Creates smooth, continuous density fields
- **Multi-resolution Processing**: 150x150 meshgrid for detailed gradients
- **Gaussian Smoothing**: Multiple passes for organic, cloud-like heat zones
- **Full-map Coverage**: Uniform base layer ensures gradient covers entire map
- **Plasma Colormap**: Beautiful purple → yellow gradient progression

## 🔑 Configuration

### Environment Variables
```env
RIOT_API_KEY=your_riot_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

### Assets
- Place Summoner's Rift map image in `assets/summoners-rift-map.png`
- Map should match League's coordinate system (14870x14980)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Riot Games**: For providing comprehensive API access
- **League of Legends Community**: For inspiration and feedback
- **Open Source Libraries**: All the amazing Python libraries that make this possible

## 🐛 Issues & Support

Found a bug or have a feature request? Please open an issue on GitHub.

---

**Disclaimer**: RiftRewind isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends.