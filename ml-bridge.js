//Node.js Bridge to Python ML Analysis
//Add this to your existing JavaScript code

const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

class MLAnalysisBridge {
    constructor() {
        this.pythonScript = path.join(__dirname, 'rift_rewind_ml.py');
        this.tempDir = path.join(__dirname, 'temp');
    }

    async ensureTempDir() {
        try {
            await fs.mkdir(this.tempDir, { recursive: true });
        } catch (error) {
            // Directory already exists
        }
    }

    async runMLAnalysis(playerStats, playerData) {
        console.log('🧠 Running ML analysis on player data...');
        
        await this.ensureTempDir();
        
        // Prepare data for Python
        const mlInputData = {
            playerInfo: {
                gameName: playerData.account.gameName,
                tagLine: playerData.account.tagLine,
                level: playerData.summoner.summonerLevel,
                totalGames: playerStats.totalGames,
                wins: playerStats.wins,
                losses: playerStats.losses,
                winRate: playerStats.totalGames > 0 ? ((playerStats.wins / playerStats.totalGames) * 100).toFixed(1) : 0,
                primaryRole: this.getPrimaryRole(playerStats.roles)
            },
            detailedMatches: playerStats.detailedMatches,
            deepAnalysis: playerStats.deepAnalysis
        };

        const inputFile = path.join(this.tempDir, `ml_input_${Date.now()}.json`);
        const outputFile = path.join(this.tempDir, `ml_output_${Date.now()}.json`);

        try {
            // Write input data
            await fs.writeFile(inputFile, JSON.stringify(mlInputData, null, 2));

            // Run Python ML analysis
            const pythonCode = `
import json
import sys
sys.path.append('${__dirname}')

from rift_rewind_ml import RiftRewindMLAnalyzer
import asyncio

async def main():
    # Load input data
    with open('${inputFile}', 'r') as f:
        input_data = json.load(f)
    
    # Initialize ML analyzer
    analyzer = RiftRewindMLAnalyzer()
    
    try:
        # Convert JS data to ML format
        df = analyzer.process_js_analysis_output(input_data)
        
        if len(df) < 3:
            result = {
                'success': False,
                'error': 'Insufficient games for ML analysis (need at least 3 games)',
                'basic_stats': input_data['playerInfo']
            }
        else:
            # Run clustering analysis
            df_clustered, cluster_profiles = analyzer.analyze_playstyle_clusters(df)
            
            # Get player's cluster
            player_cluster = df_clustered['playstyle_cluster'].mode().iloc[0] if len(df_clustered) > 0 else 0
            
            # Generate insights
            insights = analyzer.generate_personalized_insights(df_clustered, cluster_profiles, player_cluster)
            
            # Train performance models
            performance_models = analyzer.predict_performance_improvement(df_clustered)
            
            # Generate AI coaching report
            coaching_report = await analyzer.generate_ai_coaching_report(insights, input_data['playerInfo'])
            
            result = {
                'success': True,
                'playerCluster': int(player_cluster),
                'clusterProfiles': cluster_profiles,
                'personalizedInsights': insights,
                'coachingReport': coaching_report,
                'modelPredictions': {
                    'winProbabilityFeatures': performance_models['win_probability']['feature_importance'],
                    'kdaPredictionFeatures': performance_models['kda_prediction']['feature_importance']
                },
                'mlStats': {
                    'totalGamesAnalyzed': len(df_clustered),
                    'clusterSize': len(df_clustered[df_clustered['playstyle_cluster'] == player_cluster]),
                    'avgWinRateInCluster': float(cluster_profiles[player_cluster]['win_rate'])
                }
            }
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'basic_stats': input_data['playerInfo']
        }
    
    # Write output
    with open('${outputFile}', 'w') as f:
        json.dump(result, f, indent=2)

# Run async main
asyncio.run(main())
`;

            await fs.writeFile(path.join(this.tempDir, 'run_ml.py'), pythonCode);

            return new Promise((resolve, reject) => {
                const python = spawn('python', [path.join(this.tempDir, 'run_ml.py')], {
                    stdio: ['pipe', 'pipe', 'pipe']
                });

                let stdout = '';
                let stderr = '';

                python.stdout.on('data', (data) => {
                    stdout += data.toString();
                });

                python.stderr.on('data', (data) => {
                    stderr += data.toString();
                });

                python.on('close', async (code) => {
                    try {
                        if (code === 0) {
                            const result = JSON.parse(await fs.readFile(outputFile, 'utf8'));
                            
                            // Cleanup
                            await fs.unlink(inputFile).catch(() => {});
                            await fs.unlink(outputFile).catch(() => {});
                            await fs.unlink(path.join(this.tempDir, 'run_ml.py')).catch(() => {});
                            
                            resolve(result);
                        } else {
                            console.error('Python ML analysis failed:', stderr);
                            reject(new Error(`Python process failed with code ${code}: ${stderr}`));
                        }
                    } catch (error) {
                        reject(new Error(`Failed to parse ML results: ${error.message}`));
                    }
                });

                // Set timeout
                setTimeout(() => {
                    python.kill();
                    reject(new Error('ML analysis timed out'));
                }, 60000); // 60 second timeout
            });

        } catch (error) {
            console.error('ML analysis setup failed:', error);
            throw error;
        }
    }

    getPrimaryRole(roles) {
        if (!roles || Object.keys(roles).length === 0) return 'Unknown';
        
        return Object.entries(roles)
            .sort((a, b) => b[1].games - a[1].games)[0][0];
    }

    // Enhanced version of your existing generateAIInsights function
    async generateEnhancedAIInsights(playerData, stats) {
        try {
            console.log('🚀 Running enhanced ML analysis...');
            
            // Run ML analysis
            const mlResults = await this.runMLAnalysis(stats, playerData);
            
            if (!mlResults.success) {
                console.warn('ML analysis failed, falling back to basic insights:', mlResults.error);
                return this.generateBasicInsights(playerData, stats);
            }

            console.log('✅ ML analysis completed successfully!');
            
            // Combine your original AI insights with ML insights
            const enhancedInsights = {
                ...mlResults,
                originalAnalysis: {
                    totalGames: stats.totalGames,
                    winRate: stats.totalGames > 0 ? ((stats.wins / stats.totalGames) * 100).toFixed(1) : 0,
                    averageKDA: `${stats.averageKDA.kills}/${stats.averageKDA.deaths}/${stats.averageKDA.assists}`,
                    recentForm: stats.recentForm.slice(0, 10).join(''),
                    wardingEfficiency: stats.deepAnalysis.wardingPatterns.averageWardsPerGame,
                    positioningInsights: stats.deepAnalysis.positioningInsights,
                    junglePerformance: stats.deepAnalysis.junglePerformance
                }
            };

            return this.formatEnhancedResults(enhancedInsights);

        } catch (error) {
            console.error('Enhanced ML analysis failed:', error.message);
            return this.generateBasicInsights(playerData, stats);
        }
    }

    formatEnhancedResults(mlResults) {
        let report = `🧠 **ENHANCED AI ANALYSIS** 🧠\n\n`;
        
        if (mlResults.success) {
            report += `**🎯 PLAYSTYLE CLASSIFICATION**\n`;
            report += `You are classified as: ${mlResults.clusterProfiles[mlResults.playerCluster].characteristics}\n`;
            report += `Players with your style have a ${(mlResults.clusterProfiles[mlResults.playerCluster].win_rate * 100).toFixed(1)}% average win rate\n\n`;
            
            report += `**📊 MACHINE LEARNING INSIGHTS**\n`;
            mlResults.personalizedInsights.forEach(insight => {
                report += `${insight}\n`;
            });
            
            report += `\n**🤖 AI COACHING REPORT**\n`;
            report += mlResults.coachingReport;
            
            report += `\n\n**📈 PERFORMANCE PREDICTORS**\n`;
            report += `Top factors for winning:\n`;
            Object.entries(mlResults.modelPredictions.winProbabilityFeatures)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .forEach(([feature, importance]) => {
                    report += `• ${feature.replace('_', ' ')}: ${(importance * 100).toFixed(1)}% importance\n`;
                });
        } else {
            report += `ML analysis unavailable: ${mlResults.error}\n\n`;
            report += `**Basic Analysis:**\n`;
            report += `Win Rate: ${mlResults.basic_stats.winRate}%\n`;
            report += `Total Games: ${mlResults.basic_stats.totalGames}\n`;
        }

        return report;
    }

    generateBasicInsights(playerData, stats) {
        const insights = [];
        const winRate = stats.totalGames > 0 ? ((stats.wins / stats.totalGames) * 100).toFixed(1) : 0;
        insights.push(`Win Rate: ${winRate}% across ${stats.validSRGames} Summoner's Rift games`);
        
        if (stats.deepAnalysis.wardingPatterns.averageWardsPerGame > 0) {
            insights.push(`Warding: ${stats.deepAnalysis.wardingPatterns.averageWardsPerGame} wards per game average`);
        }
        
        const objTotal = stats.deepAnalysis.objectiveControl.dragonParticipation +
                         stats.deepAnalysis.objectiveControl.baronParticipation;
        if (objTotal > 0) {
            insights.push(`Objective Control: Participated in ${objTotal} major objectives`);
        }
        
        return insights.join('\n\n');
    }
}

// Export for use in your existing LoLRewindAI class
module.exports = MLAnalysisBridge;

// USAGE: Update your LoLRewindAI class
/*
// In your LoLRewindAI constructor:
constructor() {
    // ... existing code ...
    this.mlBridge = new MLAnalysisBridge();
}

// Replace your generateAIInsights function with:
async generateAIInsights(playerData, stats) {
    return await this.mlBridge.generateEnhancedAIInsights(playerData, stats);
}
*/