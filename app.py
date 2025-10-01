import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
from visualization import create_death_heatmap, create_combined_death_heatmap, create_summoners_rift_heatmap, create_map_overlay_heatmap, validate_timeline_data_availability

load_dotenv()

st.title("🎮 RiftRewind: League of Legends AI Coach")

col1, col2 = st.columns(2)
with col1:
    game_name = st.text_input("Enter Game Name", placeholder="radpoles")
with col2:
    tag_line = st.text_input("Enter Tag Line", placeholder="chill")

match_count = st.slider("Number of matches to analyze", min_value=5, max_value=50, value=20)

if st.button("🔍 Analyze Player", type="primary"):
    if game_name and tag_line:
        with st.spinner('Analyzing player data...'):
            from riftrewind_app import RiftRewindApp
            app = RiftRewindApp(os.getenv("RIOT_API_KEY"))
            result = asyncio.run(app.analyze_player_complete(game_name, tag_line, match_count))
            
            if result.get('success'):
                # Display basic results
                st.success(f"Analysis complete for {game_name}#{tag_line}")
                
                # Show summary stats
                stats = result['summary_stats']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Games Analyzed", stats['total_games'])
                with col2:
                    st.metric("Win Rate", f"{stats['win_rate']:.1%}")
                with col3:
                    st.metric("Average KDA", f"{stats['avg_kda']:.2f}")
                with col4:
                    st.metric("Primary Role", stats['primary_role'])
                
                # Death Heatmap Section
                st.header("💀 Death Analysis")
                
                if 'raw_matches' in result:
                    player_puuid = result['player_info'].get('puuid')
                    
                    # Validate timeline data availability
                    timeline_summary = validate_timeline_data_availability(
                        result['raw_matches'], player_puuid, debug=False
                    )
                    
                    # Display timeline data summary
                    with st.expander("Timeline Data Summary", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Matches", timeline_summary['total_matches'])
                            st.metric("Matches with Player", timeline_summary['matches_with_player'])
                        with col2:
                            st.metric("Matches with Timeline", timeline_summary['matches_with_timeline'])
                            st.metric("Valid Timeline Structure", timeline_summary['matches_with_valid_timeline'])
                        with col3:
                            st.metric("Potential Deaths Found", timeline_summary['potential_deaths_found'])
                        
                        if timeline_summary['timeline_data_structures']:
                            st.write("**Timeline Data Structures Found:**")
                            for structure in timeline_summary['timeline_data_structures']:
                                st.write(f"- {structure}")
                    
                    if timeline_summary['potential_deaths_found'] > 0:
                        tab1, tab2, tab3 = st.tabs(["🗺️ Map Overlay", "📊 Density Heatmap", "🎯 By Champion"])
                        
                        with tab1:
                            st.subheader("🗺️ Death Heatmap on Summoner's Rift")
                            map_overlay = create_map_overlay_heatmap(
                                result['raw_matches'], player_puuid
                            )
                            if map_overlay:
                                st.markdown(f'<img src="{map_overlay}" style="width: 100%; border-radius: 10px;">', unsafe_allow_html=True)
                                st.caption("🗺️ Death density overlaid on Summoner's Rift map with objectives")
                            else:
                                st.warning("Could not generate map overlay heatmap")
                        
                        with tab2:
                            st.subheader("📊 Death Density Analysis")
                            combined_heatmap = create_combined_death_heatmap(
                                result['raw_matches'], player_puuid
                            )
                            if combined_heatmap:
                                st.plotly_chart(combined_heatmap, use_container_width=True)
                                st.caption("📊 High-resolution density map with landmark references")
                            else:
                                st.warning("Could not generate combined heatmap despite finding death data")
                        
                        with tab3:
                            st.subheader("🎯 Champion-Specific Death Patterns")
                            champion_heatmap = create_death_heatmap(
                                result['raw_matches'], player_puuid
                            )
                            if champion_heatmap:
                                st.plotly_chart(champion_heatmap, use_container_width=True)
                                st.caption("🎯 Separate analysis for each champion played")
                            else:
                                st.warning("Could not generate champion-specific heatmap despite finding death data")
                    else:
                        st.warning(f"No death data found in {timeline_summary['matches_with_valid_timeline']} matches with valid timeline data")
                        if timeline_summary['matches_with_timeline'] == 0:
                            st.error("No timeline data found in any matches. Check API configuration or try with different matches.")
                else:
                    st.info("Timeline data analysis requires raw match data to be included in the response")
                
                # Show other insights
                st.header("🎯 AI Insights")
                playstyle = result['ml_analysis']['playstyle']
                st.write(f"**Playstyle:** {playstyle['playstyle_name']}")
                st.write(playstyle['description'])
                
                if result['ai_insights'].get('season_summary'):
                    st.write("**Season Summary:**")
                    st.write(result['ai_insights']['season_summary'])
                
            else:
                st.error(f"Analysis failed: {result.get('error')}")
    else:
        st.warning("Please enter both Game Name and Tag Line")
