#!/usr/bin/env python3
"""
Ward Ownership Detection System
Determines who placed each detected ward using multiple methods
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class WardOwnershipDetector:
    """Detect ward ownership using visual and contextual analysis"""
    
    def __init__(self):
        self.player_side = None  # 'blue' or 'red'
        self.ward_placement_history = []  # Track ward placements over time
        
        # Color ranges for ward team detection (HSV)
        self.blue_ward_range = {
            'lower': np.array([100, 50, 50]),   # Blue hue range
            'upper': np.array([130, 255, 255])
        }
        
        self.red_ward_range = {
            'lower': np.array([0, 50, 50]),     # Red hue range  
            'upper': np.array([10, 255, 255])
        }
        
        # Visual patterns for different ward states
        self.ward_patterns = {
            'my_ward': {
                'brightness_threshold': 0.7,    # My wards are brighter
                'particle_intensity': 'high',
                'outline_visibility': True
            },
            'teammate_ward': {
                'brightness_threshold': 0.5,
                'particle_intensity': 'medium', 
                'outline_visibility': True
            },
            'enemy_ward': {
                'brightness_threshold': 0.3,    # Enemy wards are dimmer
                'particle_intensity': 'low',
                'outline_visibility': False     # No outline on enemy wards
            }
        }
    
    def detect_player_side(self, game_frame: np.ndarray) -> str:
        """
        Determine if player is on blue or red team
        Uses UI elements and minimap analysis
        """
        
        # Method 1: Minimap analysis (bottom-right corner)
        h, w = game_frame.shape[:2]
        minimap_region = game_frame[int(h*0.75):h, int(w*0.75):w]
        
        # Convert to HSV for color analysis
        hsv_minimap = cv2.cvtColor(minimap_region, cv2.COLOR_BGR2HSV)
        
        # Count blue vs red pixels in minimap
        blue_mask = cv2.inRange(hsv_minimap, self.blue_ward_range['lower'], self.blue_ward_range['upper'])
        red_mask = cv2.inRange(hsv_minimap, self.red_ward_range['lower'], self.red_ward_range['upper'])
        
        blue_pixels = cv2.countNonZero(blue_mask)
        red_pixels = cv2.countNonZero(red_mask)
        
        # Player's side typically has more prominent color in minimap
        if blue_pixels > red_pixels * 1.2:
            self.player_side = 'blue'
        elif red_pixels > blue_pixels * 1.2:
            self.player_side = 'red'
        else:
            # Fallback: analyze HUD elements
            self.player_side = self._analyze_hud_elements(game_frame)
        
        return self.player_side
    
    def _analyze_hud_elements(self, game_frame: np.ndarray) -> str:
        """Analyze HUD elements to determine player side"""
        
        # Method 2: Health bar analysis (bottom of screen)
        h, w = game_frame.shape[:2]
        hud_region = game_frame[int(h*0.85):h, int(w*0.3):int(w*0.7)]
        
        hsv_hud = cv2.cvtColor(hud_region, cv2.COLOR_BGR2HSV)
        
        # Look for dominant team color in HUD
        blue_mask = cv2.inRange(hsv_hud, self.blue_ward_range['lower'], self.blue_ward_range['upper'])
        red_mask = cv2.inRange(hsv_hud, self.red_ward_range['lower'], self.red_ward_range['upper'])
        
        blue_hud = cv2.countNonZero(blue_mask)
        red_hud = cv2.countNonZero(red_mask)
        
        return 'blue' if blue_hud > red_hud else 'red'
    
    def analyze_ward_ownership(self, ward_region: np.ndarray, ward_bbox: Dict) -> Dict:
        """
        Determine ward ownership using visual analysis
        
        Args:
            ward_region: Cropped image containing just the ward
            ward_bbox: Bounding box coordinates of the ward
            
        Returns:
            Dictionary with ownership analysis
        """
        
        # Convert to different color spaces for analysis
        hsv_ward = cv2.cvtColor(ward_region, cv2.COLOR_BGR2HSV)
        lab_ward = cv2.cvtColor(ward_region, cv2.COLOR_BGR2LAB)
        
        # Method 1: Team color analysis
        team_ownership = self._detect_team_color(hsv_ward)
        
        # Method 2: Brightness and visibility analysis
        ownership_level = self._analyze_ward_visibility(ward_region)
        
        # Method 3: Particle effect analysis
        particle_intensity = self._analyze_particle_effects(ward_region)
        
        # Method 4: Outline detection (player wards have visible outlines)
        has_outline = self._detect_ward_outline(ward_region)
        
        # Combine all methods for final decision
        ownership_confidence = self._calculate_ownership_confidence(
            team_ownership, ownership_level, particle_intensity, has_outline
        )
        
        return {
            'owner': ownership_confidence['most_likely_owner'],
            'confidence': ownership_confidence['confidence'],
            'team': team_ownership,
            'visibility_level': ownership_level,
            'has_player_outline': has_outline,
            'analysis_methods': {
                'team_color': team_ownership,
                'brightness': ownership_level,
                'particles': particle_intensity,
                'outline': has_outline
            }
        }
    
    def _detect_team_color(self, hsv_ward: np.ndarray) -> str:
        """Detect if ward has blue or red team coloring"""
        
        blue_mask = cv2.inRange(hsv_ward, self.blue_ward_range['lower'], self.blue_ward_range['upper'])
        red_mask = cv2.inRange(hsv_ward, self.red_ward_range['lower'], self.red_ward_range['upper'])
        
        blue_pixels = cv2.countNonZero(blue_mask)
        red_pixels = cv2.countNonZero(red_mask)
        
        if blue_pixels > red_pixels * 1.5:
            return 'blue_team'
        elif red_pixels > blue_pixels * 1.5:
            return 'red_team'
        else:
            return 'neutral'
    
    def _analyze_ward_visibility(self, ward_region: np.ndarray) -> str:
        """Analyze ward brightness/visibility to determine ownership level"""
        
        # Convert to grayscale and calculate average brightness
        gray_ward = cv2.cvtColor(ward_region, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_ward) / 255.0
        
        # Calculate contrast and sharpness
        contrast = np.std(gray_ward) / 255.0
        
        # Determine ownership level based on visibility
        if avg_brightness > 0.7 and contrast > 0.3:
            return 'my_ward'      # Brightest and most visible
        elif avg_brightness > 0.5 and contrast > 0.2:
            return 'teammate_ward'  # Moderately visible
        else:
            return 'enemy_ward'     # Dimmer, less visible
    
    def _analyze_particle_effects(self, ward_region: np.ndarray) -> str:
        """Analyze particle effects around ward"""
        
        # Look for glowing/sparkling effects using edge detection
        gray = cv2.cvtColor(ward_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels (more edges = more particle effects)
        edge_density = cv2.countNonZero(edges) / (ward_region.shape[0] * ward_region.shape[1])
        
        if edge_density > 0.15:
            return 'high_particles'    # My ward
        elif edge_density > 0.08:
            return 'medium_particles'  # Teammate ward
        else:
            return 'low_particles'     # Enemy ward
    
    def _detect_ward_outline(self, ward_region: np.ndarray) -> bool:
        """Detect if ward has player-visible outline"""
        
        # Player wards have subtle outlines for better visibility
        # Use morphological operations to detect outlines
        
        gray = cv2.cvtColor(ward_region, cv2.COLOR_BGR2GRAY)
        
        # Create kernel for outline detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Morphological gradient highlights edges/outlines
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold to get outline pixels
        _, outline_mask = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
        
        # Count outline pixels
        outline_pixels = cv2.countNonZero(outline_mask)
        total_pixels = ward_region.shape[0] * ward_region.shape[1]
        
        # Player wards typically have 5-15% outline pixels
        outline_ratio = outline_pixels / total_pixels
        
        return 0.05 <= outline_ratio <= 0.15
    
    def _calculate_ownership_confidence(self, team_color: str, visibility: str, 
                                     particles: str, has_outline: bool) -> Dict:
        """Calculate final ownership confidence using all methods"""
        
        ownership_scores = {
            'my_ward': 0,
            'teammate_ward': 0, 
            'enemy_ward': 0
        }
        
        # Team color scoring
        if team_color == self.player_side + '_team':
            ownership_scores['my_ward'] += 0.3
            ownership_scores['teammate_ward'] += 0.2
        elif team_color != 'neutral':
            ownership_scores['enemy_ward'] += 0.4
        
        # Visibility scoring
        if visibility == 'my_ward':
            ownership_scores['my_ward'] += 0.4
        elif visibility == 'teammate_ward':
            ownership_scores['teammate_ward'] += 0.3
        elif visibility == 'enemy_ward':
            ownership_scores['enemy_ward'] += 0.3
        
        # Particle effects scoring
        if particles == 'high_particles':
            ownership_scores['my_ward'] += 0.2
        elif particles == 'medium_particles':
            ownership_scores['teammate_ward'] += 0.2
        elif particles == 'low_particles':
            ownership_scores['enemy_ward'] += 0.2
        
        # Outline scoring
        if has_outline:
            ownership_scores['my_ward'] += 0.1
            ownership_scores['teammate_ward'] += 0.05
        else:
            ownership_scores['enemy_ward'] += 0.1
        
        # Find most likely owner
        most_likely = max(ownership_scores.items(), key=lambda x: x[1])
        
        return {
            'most_likely_owner': most_likely[0],
            'confidence': most_likely[1],
            'all_scores': ownership_scores
        }
    
    def track_ward_placement_timing(self, current_wards: List[Dict], 
                                  timestamp: float) -> List[Dict]:
        """
        Track ward placements over time to improve ownership detection
        
        Args:
            current_wards: List of currently detected wards
            timestamp: Current video timestamp
            
        Returns:
            Updated ward list with ownership tracking
        """
        
        # Compare with previous frame to detect new wards
        new_wards = []
        
        for ward in current_wards:
            # Check if this is a new ward placement
            is_new_ward = not any(
                self._is_same_ward_location(ward, prev_ward) 
                for prev_ward in self.ward_placement_history
            )
            
            if is_new_ward:
                # Analyze ownership for new ward
                ownership_analysis = self.analyze_ward_ownership(
                    ward.get('region', np.array([])), 
                    ward.get('bbox', {})
                )
                
                ward.update({
                    'placement_timestamp': timestamp,
                    'ownership': ownership_analysis,
                    'is_new_placement': True
                })
                
                new_wards.append(ward)
            else:
                # Existing ward - maintain previous ownership info
                existing_ward = next(
                    (prev_ward for prev_ward in self.ward_placement_history 
                     if self._is_same_ward_location(ward, prev_ward)), 
                    None
                )
                
                if existing_ward:
                    ward['ownership'] = existing_ward.get('ownership', {})
                    ward['placement_timestamp'] = existing_ward.get('placement_timestamp', timestamp)
                    ward['is_new_placement'] = False
        
        # Update history
        self.ward_placement_history = current_wards.copy()
        
        return current_wards
    
    def _is_same_ward_location(self, ward1: Dict, ward2: Dict) -> bool:
        """Check if two wards are at the same location (within tolerance)"""
        
        bbox1 = ward1.get('bbox', {})
        bbox2 = ward2.get('bbox', {})
        
        if not bbox1 or not bbox2:
            return False
        
        # Calculate center points
        center1_x = (bbox1.get('x1', 0) + bbox1.get('x2', 0)) / 2
        center1_y = (bbox1.get('y1', 0) + bbox1.get('y2', 0)) / 2
        
        center2_x = (bbox2.get('x1', 0) + bbox2.get('x2', 0)) / 2
        center2_y = (bbox2.get('y1', 0) + bbox2.get('y2', 0)) / 2
        
        # Check if centers are within 50 pixels (same ward location)
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        return distance < 50
    
    def generate_ward_insights(self, player_wards: List[Dict], 
                             timestamp_range: Tuple[float, float]) -> Dict:
        """
        Generate insights about player's warding patterns
        
        Args:
            player_wards: List of wards placed by the player
            timestamp_range: (start_time, end_time) for analysis
            
        Returns:
            Warding insights and recommendations
        """
        
        start_time, end_time = timestamp_range
        analysis_duration = end_time - start_time
        
        # Filter player wards in time range
        time_filtered_wards = [
            ward for ward in player_wards 
            if start_time <= ward.get('placement_timestamp', 0) <= end_time
            and ward.get('ownership', {}).get('owner') == 'my_ward'
        ]
        
        # Calculate metrics
        total_wards_placed = len(time_filtered_wards)
        wards_per_minute = total_wards_placed / max(analysis_duration / 60, 1)
        
        # Analyze ward positions
        ward_positions = [(ward['position']['x'], ward['position']['y']) 
                         for ward in time_filtered_wards]
        
        # Common warding locations
        position_clusters = self._analyze_ward_positioning(ward_positions)
        
        # Warding timing analysis
        timing_analysis = self._analyze_ward_timing(time_filtered_wards)
        
        return {
            'summary': {
                'total_wards_placed': total_wards_placed,
                'wards_per_minute': round(wards_per_minute, 2),
                'analysis_duration_minutes': round(analysis_duration / 60, 1)
            },
            'positioning': {
                'favorite_locations': position_clusters,
                'coverage_analysis': self._calculate_map_coverage(ward_positions)
            },
            'timing': timing_analysis,
            'recommendations': self._generate_warding_recommendations(
                time_filtered_wards, position_clusters, timing_analysis
            )
        }
    
    def _analyze_ward_positioning(self, positions: List[Tuple[float, float]]) -> List[Dict]:
        """Analyze common ward positioning patterns"""
        
        if not positions:
            return []
        
        # Simple clustering to find common positions
        # For production, use sklearn.cluster.DBSCAN
        
        clusters = []
        cluster_radius = 100  # pixels
        
        for pos in positions:
            # Find if position belongs to existing cluster
            assigned = False
            for cluster in clusters:
                center_x, center_y = cluster['center']
                distance = np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
                
                if distance < cluster_radius:
                    # Add to existing cluster
                    cluster['positions'].append(pos)
                    cluster['count'] += 1
                    # Update center
                    cluster['center'] = (
                        sum(p[0] for p in cluster['positions']) / len(cluster['positions']),
                        sum(p[1] for p in cluster['positions']) / len(cluster['positions'])
                    )
                    assigned = True
                    break
            
            if not assigned:
                # Create new cluster
                clusters.append({
                    'center': pos,
                    'positions': [pos],
                    'count': 1
                })
        
        # Sort by frequency and return top locations
        clusters.sort(key=lambda x: x['count'], reverse=True)
        
        return [
            {
                'location': f"({int(c['center'][0])}, {int(c['center'][1])})",
                'frequency': c['count'],
                'percentage': round(c['count'] / len(positions) * 100, 1)
            }
            for c in clusters[:5]  # Top 5 locations
        ]
    
    def _calculate_map_coverage(self, positions: List[Tuple[float, float]]) -> Dict:
        """Calculate how well wards cover the map"""
        
        if not positions:
            return {'coverage_score': 0, 'analysis': 'No wards placed'}
        
        # Define map regions (approximate League map coordinates)
        map_regions = {
            'river': {'x_range': (400, 1520), 'y_range': (300, 780)},
            'jungle': {'x_range': (200, 1720), 'y_range': (100, 980)},
            'lane_bushes': {'x_range': (300, 1620), 'y_range': (200, 880)}
        }
        
        covered_regions = []
        
        for region_name, bounds in map_regions.items():
            region_wards = [
                pos for pos in positions 
                if bounds['x_range'][0] <= pos[0] <= bounds['x_range'][1]
                and bounds['y_range'][0] <= pos[1] <= bounds['y_range'][1]
            ]
            
            if region_wards:
                covered_regions.append(region_name)
        
        coverage_score = len(covered_regions) / len(map_regions) * 100
        
        return {
            'coverage_score': round(coverage_score, 1),
            'covered_regions': covered_regions,
            'total_regions': len(map_regions)
        }
    
    def _analyze_ward_timing(self, wards: List[Dict]) -> Dict:
        """Analyze ward placement timing patterns"""
        
        if not wards:
            return {'average_interval': 0, 'timing_pattern': 'No data'}
        
        timestamps = sorted([ward['placement_timestamp'] for ward in wards])
        
        # Calculate intervals between ward placements
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        
        # Analyze timing distribution
        timing_analysis = {
            'average_interval_seconds': round(avg_interval, 1),
            'total_placements': len(wards),
            'timing_consistency': 'Consistent' if len(set(
                int(interval/30) for interval in intervals  # 30-second buckets
            )) <= 3 else 'Variable'
        }
        
        return timing_analysis
    
    def _generate_warding_recommendations(self, wards: List[Dict], 
                                        positions: List[Dict], 
                                        timing: Dict) -> List[str]:
        """Generate personalized warding recommendations"""
        
        recommendations = []
        
        # Frequency recommendations
        wards_per_min = len(wards) / max(timing.get('average_interval_seconds', 60) / 60, 1)
        
        if wards_per_min < 0.5:
            recommendations.append("🎯 **Increase Warding Frequency**: Aim for 1+ ward per 2 minutes")
        
        # Position diversity recommendations  
        if len(positions) <= 2:
            recommendations.append("🗺️ **Diversify Ward Placement**: Try warding different areas for better map coverage")
        
        # Timing recommendations
        if timing.get('timing_consistency') == 'Variable':
            recommendations.append("⏰ **Consistent Warding**: Establish regular warding intervals")
        
        return recommendations[:3]  # Top 3 recommendations

# Example usage
def main():
    detector = WardOwnershipDetector()
    
    # Example ward analysis
    sample_ward = {
        'bbox': {'x1': 100, 'y1': 200, 'x2': 120, 'y2': 220},
        'position': {'x': 110, 'y': 210},
        'confidence': 0.9
    }
    
    print("Ward Ownership Detection System initialized!")
    print("Ready to analyze League of Legends ward ownership!")

if __name__ == "__main__":
    main()