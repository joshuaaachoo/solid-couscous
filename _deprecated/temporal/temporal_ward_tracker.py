#!/usr/bin/env python3
"""
Temporal Ward Tracking System
Uses timing analysis and movement correlation to determine ward ownership
"""

import cv2
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import math

@dataclass
class PlayerPosition:
    """Player position at a specific timestamp"""
    x: float
    y: float
    timestamp: float
    frame_number: int

@dataclass
class WardEvent:
    """Ward placement/detection event"""
    ward_id: str
    position: Tuple[float, float]
    timestamp: float
    frame_number: int
    ward_type: str
    confidence: float
    ownership: str = "unknown"  # "player", "teammate", "enemy", "unknown"
    ownership_confidence: float = 0.0

@dataclass
class InventoryState:
    """Player inventory state at timestamp"""
    trinket_ward_charges: int
    trinket_ward_cooldown: float
    control_ward_count: int
    support_item_charges: int
    timestamp: float

class TemporalWardTracker:
    """
    Tracks ward ownership using temporal analysis:
    1. Movement correlation - was player near when ward appeared?
    2. Inventory tracking - did trinket ward count decrease?
    3. Action timing - ward placed during player action?
    4. Pattern analysis - does this match player's warding patterns?
    """
    
    def __init__(self, tracking_window: float = 5.0):
        self.tracking_window = tracking_window  # seconds to look back for correlation
        self.player_positions = deque(maxlen=1000)  # Recent player positions
        self.ward_events = []  # All detected ward events
        self.inventory_states = deque(maxlen=100)  # Recent inventory states
        self.warding_patterns = {}  # Player's warding behavior patterns
        
        # Ownership detection thresholds
        self.movement_threshold = 150.0  # pixels - max distance for ownership
        self.timing_threshold = 2.0      # seconds - max time diff for correlation
        self.inventory_correlation_weight = 0.4
        self.movement_correlation_weight = 0.4
        self.timing_correlation_weight = 0.2
    
    def add_player_position(self, x: float, y: float, timestamp: float, frame_number: int):
        """Track player movement over time"""
        position = PlayerPosition(x, y, timestamp, frame_number)
        self.player_positions.append(position)
    
    def add_inventory_state(self, trinket_charges: int, trinket_cooldown: float, 
                          control_wards: int, support_charges: int, timestamp: float):
        """Track inventory changes over time"""
        state = InventoryState(
            trinket_charges, trinket_cooldown, control_wards, 
            support_charges, timestamp
        )
        self.inventory_states.append(state)
    
    def detect_new_ward(self, ward_position: Tuple[float, float], timestamp: float, 
                       frame_number: int, ward_type: str, confidence: float) -> WardEvent:
        """
        Detect new ward and determine ownership using temporal correlation
        """
        
        ward_event = WardEvent(
            ward_id=f"ward_{timestamp}_{ward_position[0]}_{ward_position[1]}",
            position=ward_position,
            timestamp=timestamp,
            frame_number=frame_number,
            ward_type=ward_type,
            confidence=confidence
        )
        
        # Analyze ownership using multiple temporal factors
        ownership, ownership_confidence = self._analyze_ward_ownership(ward_event)
        ward_event.ownership = ownership
        ward_event.ownership_confidence = ownership_confidence
        
        self.ward_events.append(ward_event)
        self._update_warding_patterns(ward_event)
        
        return ward_event
    
    def _analyze_ward_ownership(self, ward_event: WardEvent) -> Tuple[str, float]:
        """
        Determine ward ownership using temporal correlation analysis
        """
        
        # Factor 1: Movement correlation - was player nearby when ward appeared?
        movement_score = self._calculate_movement_correlation(ward_event)
        
        # Factor 2: Inventory correlation - did trinket/item charges decrease?
        inventory_score = self._calculate_inventory_correlation(ward_event)
        
        # Factor 3: Timing correlation - ward appeared during likely player action?
        timing_score = self._calculate_timing_correlation(ward_event)
        
        # Factor 4: Pattern correlation - matches player's warding patterns?
        pattern_score = self._calculate_pattern_correlation(ward_event)
        
        # Combine all factors with weights
        total_score = (
            movement_score * self.movement_correlation_weight +
            inventory_score * self.inventory_correlation_weight +
            timing_score * self.timing_correlation_weight +
            pattern_score * 0.1  # Lower weight for patterns initially
        )
        
        # Determine ownership based on combined score
        if total_score > 0.7:
            return "player", total_score
        elif total_score > 0.3:
            return "teammate", total_score * 0.7  # Less confident about teammates
        elif total_score < 0.1:
            return "enemy", 1.0 - total_score
        else:
            return "unknown", 0.0
    
    def _calculate_movement_correlation(self, ward_event: WardEvent) -> float:
        """
        Calculate correlation based on player movement near ward placement
        """
        
        ward_x, ward_y = ward_event.position
        ward_time = ward_event.timestamp
        
        # Look for player positions within timing threshold
        relevant_positions = [
            pos for pos in self.player_positions
            if abs(pos.timestamp - ward_time) <= self.timing_threshold
        ]
        
        if not relevant_positions:
            return 0.0
        
        # Find closest position to ward placement
        min_distance = float('inf')
        closest_time_diff = float('inf')
        
        for pos in relevant_positions:
            distance = math.sqrt((pos.x - ward_x)**2 + (pos.y - ward_y)**2)
            time_diff = abs(pos.timestamp - ward_time)
            
            if distance < min_distance:
                min_distance = distance
                closest_time_diff = time_diff
        
        # Score based on distance and timing
        if min_distance <= self.movement_threshold:
            distance_score = 1.0 - (min_distance / self.movement_threshold)
            timing_score = 1.0 - (closest_time_diff / self.timing_threshold)
            return (distance_score * 0.7) + (timing_score * 0.3)
        
        return 0.0
    
    def _calculate_inventory_correlation(self, ward_event: WardEvent) -> float:
        """
        Calculate correlation based on inventory state changes
        """
        
        ward_time = ward_event.timestamp
        ward_type = ward_event.ward_type
        
        # Find inventory states before and after ward placement
        before_states = [
            state for state in self.inventory_states
            if state.timestamp <= ward_time and ward_time - state.timestamp <= 1.0
        ]
        
        after_states = [
            state for state in self.inventory_states
            if state.timestamp > ward_time and state.timestamp - ward_time <= 1.0
        ]
        
        if not before_states or not after_states:
            return 0.0
        
        before_state = before_states[-1]  # Most recent before
        after_state = after_states[0]    # Earliest after
        
        # Check for appropriate inventory decreases
        if ward_type in ["stealth_ward", "zombie_ward"]:
            # Trinket ward or support item ward
            trinket_decreased = before_state.trinket_ward_charges > after_state.trinket_ward_charges
            support_decreased = before_state.support_item_charges > after_state.support_item_charges
            
            if trinket_decreased or support_decreased:
                return 0.9
            elif after_state.trinket_ward_cooldown > before_state.trinket_ward_cooldown:
                return 0.7  # Trinket went on cooldown
        
        elif ward_type == "control_ward":
            # Control ward from inventory
            control_decreased = before_state.control_ward_count > after_state.control_ward_count
            if control_decreased:
                return 0.95
        
        return 0.0
    
    def _calculate_timing_correlation(self, ward_event: WardEvent) -> float:
        """
        Calculate correlation based on timing patterns (movement stops, etc.)
        """
        
        ward_time = ward_event.timestamp
        
        # Look for movement patterns indicating ward placement
        # (player stops moving, changes direction, etc.)
        recent_positions = [
            pos for pos in self.player_positions
            if abs(pos.timestamp - ward_time) <= 2.0
        ]
        
        if len(recent_positions) < 3:
            return 0.0
        
        # Sort by timestamp
        recent_positions.sort(key=lambda p: p.timestamp)
        
        # Calculate movement speeds before and after
        speeds_before = []
        speeds_after = []
        
        for i in range(1, len(recent_positions)):
            curr_pos = recent_positions[i]
            prev_pos = recent_positions[i-1]
            
            distance = math.sqrt(
                (curr_pos.x - prev_pos.x)**2 + (curr_pos.y - prev_pos.y)**2
            )
            time_diff = curr_pos.timestamp - prev_pos.timestamp
            
            if time_diff > 0:
                speed = distance / time_diff
                
                if curr_pos.timestamp < ward_time:
                    speeds_before.append(speed)
                else:
                    speeds_after.append(speed)
        
        # Look for speed reduction (stopping to place ward)
        if speeds_before and speeds_after:
            avg_speed_before = np.mean(speeds_before)
            avg_speed_after = np.mean(speeds_after)
            
            # Player likely stopped to place ward
            if avg_speed_before > 50 and avg_speed_after < 20:
                return 0.8
            elif avg_speed_before > avg_speed_after * 1.5:
                return 0.5
        
        return 0.3  # Neutral timing
    
    def _calculate_pattern_correlation(self, ward_event: WardEvent) -> float:
        """
        Calculate correlation based on player's established warding patterns
        """
        
        ward_x, ward_y = ward_event.position
        ward_type = ward_event.ward_type
        
        if not self.warding_patterns:
            return 0.5  # Neutral when no patterns established
        
        # Check if this location matches player's common warding spots
        common_spots = self.warding_patterns.get('common_locations', [])
        
        for spot in common_spots:
            distance = math.sqrt((ward_x - spot['x'])**2 + (ward_y - spot['y'])**2)
            if distance < 100:  # Within 100 pixels of common spot
                return min(0.8, spot['frequency'] / 10)  # Scale by frequency
        
        # Check ward type preferences
        type_preferences = self.warding_patterns.get('type_preferences', {})
        type_weight = type_preferences.get(ward_type, 0.5)
        
        return type_weight * 0.6
    
    def _update_warding_patterns(self, ward_event: WardEvent):
        """
        Update player's warding behavior patterns based on confirmed placements
        """
        
        if ward_event.ownership != "player":
            return
        
        # Initialize patterns if needed
        if not self.warding_patterns:
            self.warding_patterns = {
                'common_locations': [],
                'type_preferences': {},
                'timing_patterns': []
            }
        
        # Update location patterns
        ward_x, ward_y = ward_event.position
        locations = self.warding_patterns['common_locations']
        
        # Check if this is near an existing common location
        updated_existing = False
        for location in locations:
            distance = math.sqrt((ward_x - location['x'])**2 + (ward_y - location['y'])**2)
            if distance < 50:  # Merge nearby locations
                # Update location with weighted average
                total_freq = location['frequency'] + 1
                location['x'] = (location['x'] * location['frequency'] + ward_x) / total_freq
                location['y'] = (location['y'] * location['frequency'] + ward_y) / total_freq
                location['frequency'] = total_freq
                updated_existing = True
                break
        
        if not updated_existing:
            locations.append({
                'x': ward_x,
                'y': ward_y,
                'frequency': 1
            })
        
        # Update type preferences
        type_prefs = self.warding_patterns['type_preferences']
        ward_type = ward_event.ward_type
        type_prefs[ward_type] = type_prefs.get(ward_type, 0) + 1
        
        # Normalize type preferences
        total_wards = sum(type_prefs.values())
        for ward_type in type_prefs:
            type_prefs[ward_type] = type_prefs[ward_type] / total_wards
    
    def get_ownership_summary(self) -> Dict:
        """
        Get summary of ward ownership detection results
        """
        
        total_wards = len(self.ward_events)
        if total_wards == 0:
            return {"message": "No wards detected yet"}
        
        ownership_counts = {}
        confidence_scores = []
        
        for ward in self.ward_events:
            ownership_counts[ward.ownership] = ownership_counts.get(ward.ownership, 0) + 1
            confidence_scores.append(ward.ownership_confidence)
        
        return {
            "total_wards_detected": total_wards,
            "ownership_breakdown": ownership_counts,
            "average_confidence": np.mean(confidence_scores),
            "player_wards": ownership_counts.get("player", 0),
            "teammate_wards": ownership_counts.get("teammate", 0),
            "enemy_wards": ownership_counts.get("enemy", 0),
            "unknown_wards": ownership_counts.get("unknown", 0),
            "warding_patterns": self.warding_patterns
        }
    
    def export_analysis(self, filepath: str):
        """
        Export detailed ward analysis to JSON file
        """
        
        analysis_data = {
            "timestamp": time.time(),
            "tracking_window": self.tracking_window,
            "ward_events": [
                {
                    "ward_id": ward.ward_id,
                    "position": ward.position,
                    "timestamp": ward.timestamp,
                    "frame_number": ward.frame_number,
                    "ward_type": ward.ward_type,
                    "ownership": ward.ownership,
                    "ownership_confidence": ward.ownership_confidence
                }
                for ward in self.ward_events
            ],
            "summary": self.get_ownership_summary(),
            "parameters": {
                "movement_threshold": self.movement_threshold,
                "timing_threshold": self.timing_threshold,
                "movement_weight": self.movement_correlation_weight,
                "inventory_weight": self.inventory_correlation_weight,
                "timing_weight": self.timing_correlation_weight
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"✅ Ward analysis exported to: {filepath}")

# Example usage and testing
if __name__ == "__main__":
    
    # Initialize tracker
    tracker = TemporalWardTracker(tracking_window=5.0)
    
    # Simulate player movement and ward placement
    print("🎯 Testing Temporal Ward Tracking...")
    
    # Player moves toward river bush
    tracker.add_player_position(800, 600, 10.0, 300)  # Moving toward river
    tracker.add_player_position(820, 580, 10.5, 315)  # Getting closer
    tracker.add_player_position(850, 550, 11.0, 330)  # At river bush
    
    # Inventory state before ward placement
    tracker.add_inventory_state(
        trinket_charges=1, trinket_cooldown=0.0, 
        control_wards=2, support_charges=0, timestamp=10.8
    )
    
    # Player places ward (detected)
    ward_event = tracker.detect_new_ward(
        ward_position=(860, 545), timestamp=11.2, 
        frame_number=336, ward_type="stealth_ward", confidence=0.92
    )
    
    # Inventory state after ward placement  
    tracker.add_inventory_state(
        trinket_charges=0, trinket_cooldown=180.0,
        control_wards=2, support_charges=0, timestamp=11.5
    )
    
    # Player moves away
    tracker.add_player_position(870, 560, 11.8, 354)
    tracker.add_player_position(890, 580, 12.3, 369)
    
    print(f"\\n🔍 Ward Analysis Results:")
    print(f"Ward ID: {ward_event.ward_id}")
    print(f"Ownership: {ward_event.ownership}")
    print(f"Confidence: {ward_event.ownership_confidence:.2f}")
    
    # Test enemy ward (far from player)
    enemy_ward = tracker.detect_new_ward(
        ward_position=(200, 200), timestamp=15.0,
        frame_number=450, ward_type="stealth_ward", confidence=0.88
    )
    
    print(f"\\nEnemy Ward Analysis:")
    print(f"Ownership: {enemy_ward.ownership}")
    print(f"Confidence: {enemy_ward.ownership_confidence:.2f}")
    
    # Get overall summary
    summary = tracker.get_ownership_summary()
    print(f"\\n📊 Overall Summary:")
    print(f"Total Wards: {summary['total_wards_detected']}")
    print(f"Player Wards: {summary['player_wards']}")
    print(f"Enemy Wards: {summary['enemy_wards']}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")
    
    # Export analysis
    tracker.export_analysis("ward_analysis_test.json")
    
    print("\\n✅ Temporal ward tracking system ready for integration!")