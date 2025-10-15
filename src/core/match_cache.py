"""
Match data caching layer using DynamoDB
Caches match details to avoid redundant API calls
"""
import boto3
import json
import time
import os
from typing import Optional, Dict, Any
from decimal import Decimal


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types from DynamoDB"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


class MatchCache:
    """Cache layer for match data using DynamoDB"""
    
    def __init__(self):
        """Initialize DynamoDB client and table"""
        self.dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.table_name = 'riftrewind-matches'
        
        try:
            self.table = self.dynamodb.Table(self.table_name)
        except Exception as e:
            print(f"Warning: Could not access DynamoDB table: {e}")
            self.table = None
    
    def _convert_floats_to_decimal(self, obj: Any) -> Any:
        """Convert float values to Decimal for DynamoDB"""
        if isinstance(obj, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimal(item) for item in obj]
        elif isinstance(obj, float):
            return Decimal(str(obj))
        return obj
    
    def _convert_decimals_to_float(self, obj: Any) -> Any:
        """Convert Decimal values back to float"""
        if isinstance(obj, dict):
            return {k: self._convert_decimals_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimals_to_float(item) for item in obj]
        elif isinstance(obj, Decimal):
            return float(obj)
        return obj
    
    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get match details from cache
        
        Args:
            match_id: Match ID to retrieve
            
        Returns:
            Match data dict or None if not cached
        """
        if not self.table:
            return None
        
        try:
            response = self.table.get_item(Key={'match_id': match_id})
            
            if 'Item' in response:
                item = response['Item']
                # Check if cache is still valid (7 days TTL)
                if 'ttl' in item and item['ttl'] > int(time.time()):
                    match_data = item.get('match_data')
                    return self._convert_decimals_to_float(match_data)
            
            return None
        except Exception as e:
            print(f"Cache read error: {e}")
            return None
    
    def put_match(self, match_id: str, match_data: Dict[str, Any]) -> bool:
        """
        Store match details in cache
        
        Args:
            match_id: Match ID
            match_data: Full match data from Riot API
            
        Returns:
            True if successful, False otherwise
        """
        if not self.table:
            return False
        
        try:
            # Convert floats to Decimal for DynamoDB
            safe_data = self._convert_floats_to_decimal(match_data)
            
            # Set TTL to 7 days from now
            ttl = int(time.time()) + (7 * 24 * 60 * 60)
            
            self.table.put_item(Item={
                'match_id': match_id,
                'match_data': safe_data,
                'ttl': ttl,
                'cached_at': int(time.time())
            })
            
            return True
        except Exception as e:
            print(f"Cache write error: {e}")
            return False
    
    def get_match_batch(self, match_ids: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple matches from cache at once
        
        Args:
            match_ids: List of match IDs
            
        Returns:
            Dict mapping match_id to match_data for cached matches
        """
        if not self.table or not match_ids:
            return {}
        
        cached_matches = {}
        
        try:
            # DynamoDB BatchGetItem supports up to 100 items
            for i in range(0, len(match_ids), 100):
                batch = match_ids[i:i+100]
                
                response = self.dynamodb.batch_get_item(
                    RequestItems={
                        self.table_name: {
                            'Keys': [{'match_id': mid} for mid in batch]
                        }
                    }
                )
                
                for item in response.get('Responses', {}).get(self.table_name, []):
                    # Check TTL
                    if 'ttl' in item and item['ttl'] > int(time.time()):
                        match_id = item['match_id']
                        match_data = self._convert_decimals_to_float(item['match_data'])
                        cached_matches[match_id] = match_data
            
            return cached_matches
        except Exception as e:
            print(f"Batch cache read error: {e}")
            return {}
    
    def put_match_batch(self, matches: Dict[str, Dict[str, Any]]) -> int:
        """
        Store multiple matches in cache at once
        
        Args:
            matches: Dict mapping match_id to match_data
            
        Returns:
            Number of matches successfully cached
        """
        if not self.table or not matches:
            return 0
        
        cached_count = 0
        ttl = int(time.time()) + (7 * 24 * 60 * 60)
        
        try:
            # DynamoDB BatchWriteItem supports up to 25 items
            items = []
            for match_id, match_data in matches.items():
                safe_data = self._convert_floats_to_decimal(match_data)
                items.append({
                    'PutRequest': {
                        'Item': {
                            'match_id': match_id,
                            'match_data': safe_data,
                            'ttl': ttl,
                            'cached_at': int(time.time())
                        }
                    }
                })
            
            # Process in batches of 25
            for i in range(0, len(items), 25):
                batch = items[i:i+25]
                self.dynamodb.batch_write_item(
                    RequestItems={
                        self.table_name: batch
                    }
                )
                cached_count += len(batch)
            
            return cached_count
        except Exception as e:
            print(f"Batch cache write error: {e}")
            return cached_count
