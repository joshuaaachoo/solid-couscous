import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
# SageMaker and Bedrock AI imports
from sagemaker_deployment import SageMakerModelDeployment
from bedrock_coaching_insights import BedrockCoachingInsights, MLCoachingPipeline

load_dotenv()

# Initialize session state for all modes
if 'vod_analysis_result' not in st.session_state:
    st.session_state.vod_analysis_result = None
if 'vod_analysis_running' not in st.session_state:
    st.session_state.vod_analysis_running = False
if 'show_vod_analysis' not in st.session_state:
    st.session_state.show_vod_analysis = False
if 'show_sagemaker_mgmt' not in st.session_state:
    st.session_state.show_sagemaker_mgmt = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None

st.title("🎮 RiftRewind: League of Legends AI Coach")

# Clear session state button
if st.button("🗑️ Clear All Data", help="Reset the app state"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("✅ All data cleared!")

# Mode selector at the top
st.write("**Choose your mode:**")
mode_col1, mode_col2, mode_col3 = st.columns(3)

with mode_col1:
    if st.button("📊 Player Analysis", use_container_width=True):
        st.session_state.show_vod_analysis = False
        st.session_state.show_sagemaker_mgmt = False

with mode_col2:
    if st.button("🎬 VOD Analysis", use_container_width=True):
        st.session_state.show_vod_analysis = True
        st.session_state.show_sagemaker_mgmt = False

with mode_col3:
    if st.button("🤖 SageMaker Models", use_container_width=True):
        st.session_state.show_vod_analysis = False
        st.session_state.show_sagemaker_mgmt = True

# Show SageMaker Management mode
if st.session_state.show_sagemaker_mgmt:
    from sagemaker_management_ui import render_sagemaker_management
    render_sagemaker_management()

# Show VOD Analysis if in VOD mode
elif st.session_state.show_vod_analysis:
    st.header("🎬 Professional VOD Analysis")
    st.write("Analyze professional players' VODs to learn optimal strategies and patterns")
    
    # Check AWS credentials and configuration
    aws_configured = bool(os.getenv('AWS_ACCESS_KEY_ID')) and bool(os.getenv('AWS_SECRET_ACCESS_KEY'))
    s3_bucket = os.getenv('AWS_S3_BUCKET', 'riftrewind-vods')
    
    col_aws1, col_aws2 = st.columns(2)
    with col_aws1:
        if aws_configured:
            st.success("🚀 **AWS SageMaker Ready** - Professional ML analysis available")
        else:
            st.warning("⚠️ **AWS Not Configured** - Set up AWS credentials for SageMaker analysis")
            st.code("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file")
    
    with col_aws2:
        st.info(f"📦 **S3 Bucket**: `{s3_bucket}`")
        st.caption("Set AWS_S3_BUCKET in .env to customize bucket name")
    
    # VOD Analysis Interface
    with st.form("vod_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            youtube_url = st.text_input(
                "🔗 YouTube/Twitch URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste the URL of a professional player's gameplay video"
            )
        
        with col2:
            player_name = st.text_input(
                "👤 Pro Player Name",
                placeholder="Faker, Canyon, Caps, etc.",
                help="Name of the professional player in the video"
            )
        
        # Upload option with enhanced features
        st.write("**Or upload a video file:**")
        uploaded_file = st.file_uploader(
            "Upload VOD", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV (max 200MB recommended)"
        )
        
        if uploaded_file:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            col_file1, col_file2, col_file3 = st.columns(3)
            with col_file1:
                st.metric("File Size", f"{file_size_mb:.1f} MB")
            with col_file2:
                st.metric("Duration", "~Unknown")
            with col_file3:
                if file_size_mb > 200:
                    st.error("⚠️ Large file")
                else:
                    st.success("✅ Good size")
        
        submitted = st.form_submit_button("🚀 Analyze with AWS SageMaker")
        
        if submitted:
            if (youtube_url and player_name) or (uploaded_file and player_name):
                if aws_configured:
                    st.session_state.vod_analysis_running = True
                    
                    # Initialize analyzer
                    from sagemaker_vod_analyzer import SageMakerVODAnalyzer
                    analyzer = SageMakerVODAnalyzer()
                    
                    if uploaded_file:
                        # File upload workflow
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Step 1: Upload to S3
                            status_text.text("📤 Uploading video to AWS S3...")
                            progress_bar.progress(25)
                            
                            upload_result = analyzer.upload_video_to_s3(uploaded_file, s3_bucket)
                            
                            if upload_result.get('success'):
                                progress_bar.progress(50)
                                status_text.text("🤖 Analyzing video with SageMaker...")
                                
                                # Step 2: Analyze uploaded video
                                result = analyzer.analyze_uploaded_vod(
                                    upload_result['s3_url'], 
                                    player_name,
                                    upload_result['key']
                                )
                                
                                progress_bar.progress(100)
                                status_text.text("✅ Analysis complete!")
                                
                                if result.get('success'):
                                    st.session_state.vod_analysis_result = result
                                    st.success("🎯 VOD analysis complete!")
                                else:
                                    st.error(f"Analysis failed: {result.get('error')}")
                            else:
                                st.error(f"Upload failed: {upload_result.get('error')}")
                                
                        except Exception as e:
                            st.error(f"File analysis error: {str(e)}")
                        finally:
                            progress_bar.empty()
                            status_text.empty()
                    
                    else:
                        # YouTube URL workflow
                        with st.spinner(f"🤖 AWS SageMaker analyzing {player_name}'s YouTube VOD..."):
                            try:
                                result = analyzer.analyze_youtube_vod(youtube_url, player_name)
                                
                                if result.get('success'):
                                    st.session_state.vod_analysis_result = result
                                    st.success("✅ YouTube VOD analysis complete!")
                                else:
                                    st.error(f"Analysis failed: {result.get('error')}")
                                    
                            except Exception as e:
                                st.error(f"YouTube analysis error: {str(e)}")
                    
                    st.session_state.vod_analysis_running = False
                else:
                    st.error("AWS SageMaker not configured. Please set up AWS credentials.")
            else:
                st.warning("Please provide YouTube URL or upload a file, plus player name")
    
    # Display analysis results with AI coaching
    if st.session_state.vod_analysis_result:
        result = st.session_state.vod_analysis_result
        
        st.markdown("---")
        st.header("📊 VOD Analysis Results + AI Coaching")
        
        # Analysis info header
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            analysis_type = "📁 Uploaded File" if result.get('file_processed') else "🔗 YouTube URL"
            st.info(f"**Source**: {analysis_type}")
        with col_info2:
            confidence = result.get('analysis_confidence', 0.8)
            st.info(f"**Confidence**: {confidence:.1%}")
        with col_info3:
            processing_time = result.get('processing_time', 'Unknown')
            st.info(f"**Processing**: {processing_time}")
        
        # Enhanced Precision Metrics
        st.markdown("### 🎯 Enhanced Precision Analysis")
        
        # Primary metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔍 Vision Score", f"{result.get('vision_control_score', 0)}/100")
        with col2:
            st.metric("📍 Total Wards", result.get('total_wards_placed', 0))
        with col3:
            duration_min = result.get('game_duration_seconds', 1800) // 60
            st.metric("⏱️ Duration", f"{duration_min}m")
        with col4:
            player_name = result.get('player_name', 'Unknown')
            st.metric("🎮 Player", player_name)
        
        # Enhanced precision metrics row
        if result.get('analysis_quality') == 'MAXIMUM':
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                positioning = result.get('positioning_accuracy', 85)
                st.metric("🎯 Positioning", f"{positioning}%", delta=f"{positioning-80}% vs avg")
            with col6:
                ward_eff = result.get('ward_efficiency_score', 88)
                st.metric("⚡ Ward Efficiency", f"{ward_eff}%", delta=f"{ward_eff-75}% vs avg")
            with col7:
                decision = result.get('decision_timing_score', 79)
                st.metric("🧠 Decision Speed", f"{decision}%", delta=f"{decision-70}% vs avg")
            with col8:
                coord = result.get('team_coordination_rating', 77)
                st.metric("🤝 Coordination", f"{coord}%", delta=f"{coord-65}% vs avg")
        
        # Display Micro-Analysis if available
        if 'micro_analysis' in result:
            st.markdown("---")
            st.subheader("🔬 Micro-Analysis Breakdown")
            
            micro = result['micro_analysis']
            
            col_micro1, col_micro2 = st.columns(2)
            with col_micro1:
                st.markdown("**🌅 Early Game Execution**")
                early = micro.get('early_game_execution', {})
                st.metric("Score", f"{early.get('score', 85)}/100")
                st.caption(early.get('details', 'Strong performance'))
                st.info(f"✨ **Tip**: {early.get('improvement', 'Keep up the good work!')}")
                
                st.markdown("**🌆 Late Game Precision**")
                late = micro.get('late_game_precision', {})
                st.metric("Score", f"{late.get('score', 82)}/100")
                st.caption(late.get('details', 'Solid execution'))
                st.info(f"✨ **Tip**: {late.get('improvement', 'Maintain consistency!')}")
            
            with col_micro2:
                st.markdown("**🌇 Mid Game Transitions**")
                mid = micro.get('mid_game_transitions', {})
                st.metric("Score", f"{mid.get('score', 79)}/100")
                st.caption(mid.get('details', 'Good transitions'))
                st.info(f"✨ **Tip**: {mid.get('improvement', 'Focus on timing!')}")
                
                st.markdown("**👁️ Vision Mastery**")
                vision = micro.get('vision_mastery', {})
                st.metric("Score", f"{vision.get('score', 88)}/100")
                st.caption(vision.get('details', 'Excellent vision control'))
                st.info(f"✨ **Tip**: {vision.get('improvement', 'Maintain excellence!')}")
        
        # Generate Enhanced AI Coaching Insights
        st.markdown("---")
        st.subheader("🤖 Enhanced Precision AI Coaching")
        
        precision_note = result.get('precision_note', '')
        if precision_note:
            st.info(precision_note)
        
        if st.button("🎯 Generate Precision AI Coaching", help="Use AWS Bedrock Claude 3 to analyze maximum precision results and generate micro-coaching insights"):
            try:
                # Initialize Enhanced Bedrock coaching
                from bedrock_insights import BedrockInsightsGenerator
                bedrock_coach = BedrockInsightsGenerator()
                
                with st.spinner("🎯 AWS Bedrock generating MAXIMUM PRECISION coaching insights..."):
                    # Generate precision coaching based on enhanced analysis
                    ml_insights = {'playstyle': {'playstyle_name': 'Precision Analysis'}, 'coaching_insights': []}
                    precision_coaching = bedrock_coach.generate_precision_coaching_insights(result, ml_insights)
                    
                    if precision_coaching:
                        st.success("✅ Enhanced Precision AI Coaching generated!")
                        
                        # Display precision coaching with enhanced formatting
                        st.markdown("### 🎯 Precision Coaching Report")
                        
                        # Format the coaching text with better presentation
                        coaching_lines = precision_coaching.split('\n')
                        for line in coaching_lines:
                            if line.strip():
                                if '**' in line or line.startswith('#'):
                                    st.markdown(line)
                                elif line.startswith('•') or line.startswith('-'):
                                    st.markdown(f"- {line.lstrip('•- ')}")
                                else:
                                    st.markdown(line)
                        
                    else:
                        st.warning("⚠️ Precision coaching generation failed, using enhanced fallback insights")
                        
                        # Show enhanced fallback coaching
                        st.markdown("### 🎯 Enhanced Analysis Insights")
                        st.markdown("📊 **Performance Summary**: Analysis complete with enhanced metrics")
                        st.markdown("🎯 **Positioning**: Focus on maintaining optimal distances and angles")  
                        st.markdown("⚡ **Decision Speed**: Practice faster macro decision making")
                        st.markdown("👁️ **Vision Control**: Optimize ward timing and placement efficiency")
                        
            except Exception as e:
                st.error(f"❌ Precision coaching error: {str(e)}")
                
                # Show Bedrock connection status
                try:
                    from bedrock_insights import BedrockInsightsGenerator
                    bedrock_coach = BedrockInsightsGenerator()
                    status = bedrock_coach.get_connection_status()
                    if status['status'] == 'connected':
                        st.info("ℹ️ AWS Bedrock is connected. Please try again.")
                    else:
                        st.warning(f"⚠️ {status['message']}")
                        st.info("Using fallback analysis. Check AWS credentials for full AI coaching.")
                except:
                    st.info("💡 Tip: Make sure AWS Bedrock is configured in your AWS account")
        
        # Key learnings
        if result.get('key_learnings'):
            st.write("**🎯 Key Learnings:**")
            for learning in result['key_learnings'][:5]:
                st.write(f"• {learning}")
        
        # Practice suggestions
        if result.get('practice_suggestions'):
            st.write("**💡 Practice Suggestions:**")
            for i, suggestion in enumerate(result['practice_suggestions'][:5], 1):
                st.write(f"{i}. {suggestion}")
        
        # Clear results
        if st.button("🗑️ Clear Results"):
            st.session_state.vod_analysis_result = None
            st.success("✅ Results cleared!")

# Regular Player Analysis Mode
else:
    # Player input
    st.header("📊 Player Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        game_name = st.text_input("Game Name", placeholder="radpoles")
    with col2:
        tag_line = st.text_input("Tag Line", placeholder="chill")
    
    # Check if we have ML models available
    try:
        from riftrewind_ml_engine import RiftRewindMLEngine
        ml_available = True
        st.success("🤖 **ML Models Active** - Real machine learning analysis available!")
    except ImportError as e:
        ml_available = False
        st.error(f"❌ **No ML Models** - {str(e)}")
        st.code("pip install scikit-learn pandas numpy joblib boto3")
    
    # Match count slider - always show regardless of API key
    col_slider1, col_slider2 = st.columns([3, 1])
    with col_slider1:
        match_count = st.slider(
            "Number of matches to analyze:",
            min_value=3,
            max_value=20,
            value=7,
            help="More matches = better accuracy but longer analysis time"
        )
    with col_slider2:
        estimated_time = "5-10s" if match_count <= 5 else "15-30s" if match_count <= 10 else "30-60s"
        st.metric("Est. Time", estimated_time)
    
    # Analysis type selection - always show
    analysis_type = st.radio(
        "Analysis Depth:",
        ["⚡ Basic ML", "🔍 Full + AI"],
        horizontal=True,
        help="Basic ML: Fast local analysis. Full + AI: Includes AWS insights and pro comparisons."
    )

    if st.button("🎯 Analyze Player"):
        if game_name and tag_line:
            if not ml_available:
                st.error("Cannot analyze - ML models not available. Install dependencies first.")
            else:
                try:
                    from riftrewind_app import RiftRewindApp
                    
                    # Initialize the RiftRewind app
                    riot_api_key = os.getenv('RIOT_API_KEY')
                    if not riot_api_key:
                        st.error("⚠️ RIOT_API_KEY not found in environment variables")
                        st.info("Please set your Riot API key in a .env file")
                    else:
                        app = RiftRewindApp(riot_api_key)
                        
                        if analysis_type.startswith("⚡"):
                            with st.spinner(f"⚡ Analyzing {match_count} matches for {game_name}#{tag_line}..."):
                                result = asyncio.run(app.analyze_player_quick(game_name, tag_line, match_count))
                        else:
                            with st.spinner(f"🔍 Deep analyzing {match_count} matches for {game_name}#{tag_line}..."):
                                result = asyncio.run(app.analyze_player_complete(game_name, tag_line, match_count))
                        
                        if 'error' not in result:
                            st.session_state.analysis_results = result
                            st.session_state.selected_player = f"{game_name}#{tag_line}"
                            
                            # Show analysis results
                            st.success("✅ Analysis complete!")
                            
                            # Basic stats - handle both quick and complete analysis formats
                            stats = result.get('stats') or result.get('summary_stats', {})
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Win Rate", f"{stats.get('win_rate', 0):.1%}")
                            with col2:
                                st.metric("Avg KDA", f"{stats.get('avg_kda', 0):.2f}")
                            with col3:
                                st.metric("CS/min", f"{stats.get('avg_cs_per_min', 0):.1f}")
                            with col4:
                                vision_val = stats.get('avg_vision_per_min', 0)
                                st.metric("Vision/min", f"{vision_val:.2f}" if vision_val > 0.01 else f"{vision_val:.3f}")
                            
                            # AI Insights
                            if result.get('ai_insights'):
                                st.write("**🤖 AI Insights:**")
                                insights = result['ai_insights']
                                if insights.get('season_summary'):
                                    st.write(insights['season_summary'])
                            
                            # Additional insights
                            if result.get('coaching_feedback'):
                                st.write("**📍 Performance Analysis:**")
                                st.write("Advanced analytics and heatmaps coming soon with SageMaker integration!")
                        
                        else:
                            st.error(f"Analysis failed: {result.get('error')}")
                        
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
        else:
            st.warning("Please enter both Game Name and Tag Line")
    
    # Show previous results if they exist
    if st.session_state.analysis_results and st.session_state.selected_player:
        st.markdown("---")
        st.write(f"**Previous Analysis:** {st.session_state.selected_player}")
        if st.button("🗑️ Clear Player Results"):
            st.session_state.analysis_results = None
            st.session_state.selected_player = None
            st.success("✅ Player results cleared!")

# Footer
st.markdown("---")
st.markdown("**RiftRewind** - AI-powered League of Legends coaching system")