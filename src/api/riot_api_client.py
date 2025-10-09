from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class PlayerInfo:
    puuid: str
    game_name: str
    tag_line: str
    summoner_level: int
    profile_icon_id: int

class RiotApiClient:
    def platform_from_routing(self, routing: str) -> str:
        lookup = {
            "americas": "na1",  # NA
            "europe": "euw1",   # EUW1
            "asia": "kr"        # KR
        }
        return lookup.get(routing, "na1")

    def __init__(self, api_key: str, region: str = "americas"):
        self.api_key = api_key
        self.region = region
        # Endpoint URLs with correct platform/routing
        self.baseurls = {
            "account": f"https://{self.region}.api.riotgames.com",
            "summoner": self.getplatformurl(self.region),
            "match": f"https://{self.region}.api.riotgames.com",
            "league": self.getplatformurl(self.region)
        }
        self.session = None
        self.request_times = []
        self.rate_limits = {
            "personal": {"requests": 100, "seconds": 120},
            "production": {"requests": 3000, "seconds": 10},
        }

    def getplatformurl(self, region: str) -> str:
        # You can update this dict to add more regions as needed
        routing = {
            "americas": "americas.api.riotgames.com",
            "na1": "na1.api.riotgames.com",
            "euw1": "euw1.api.riotgames.com",
        }
        return f"https://{routing.get(region, region + '.api.riotgames.com')}"

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"X-Riot-Token": self.api_key}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _check_rate_limit(self):
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 120]
        if len(self.request_times) >= 90:
            sleep_time = 120 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

    async def _make_request(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                self._check_rate_limit()
                self.request_times.append(time.time())
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        await asyncio.sleep(retry_after)
                        continue
                    elif response.status == 404:
                        return None
                    else:
                        logging.warning(f"API request failed: {response.status} - {url}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                logging.warning(f"Request timeout for {url}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"Request error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        return None

    async def get_match_timeline(self, match_id: str) -> Optional[dict]:
        url = f"{self.baseurls['match']}/lol/match/v5/matches/{match_id}/timeline"
        return await self._make_request(url)

    async def get_player_info(self, game_name: str, tag_line: str) -> Optional[PlayerInfo]:
        account_url = f"{self.baseurls['account']}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        account_data = await self._make_request(account_url)
        if not account_data:
            return None
        puuid = account_data.get('puuid')
        if not puuid:
            return None
        platform_region = self.platform_from_routing(self.region)
        summoner_url = f"https://{platform_region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        summoner_data = await self._make_request(summoner_url)
        if not summoner_data:
            return None
        return PlayerInfo(
            puuid=puuid,
            game_name=game_name,
            tag_line=tag_line,
            summoner_level=summoner_data.get('summonerLevel', 0),
            profile_icon_id=summoner_data.get('profileIconId', 0)
        )

    async def get_match_history(self, puuid: str, count: int = 100, queue_type: int = 420) -> List[str]:
        match_ids = []
        start_index = 0
        batch_size = 100
        while len(match_ids) < count and start_index < 1000:
            current_batch = min(batch_size, count - len(match_ids))
            # If queue_type is None or 0, don't filter by queue (get all game modes)
            if queue_type is None or queue_type == 0:
                url = f"{self.baseurls['match']}/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start_index}&count={current_batch}"
            else:
                url = f"{self.baseurls['match']}/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start_index}&count={current_batch}&queue={queue_type}"
            batch_ids = await self._make_request(url)
            if not batch_ids or len(batch_ids) == 0:
                break
            match_ids.extend(batch_ids)
            start_index += len(batch_ids)
            if len(batch_ids) < current_batch:
                break
        return match_ids[:count]

    async def get_match_details(self, match_id: str) -> Optional[Dict]:
        url = f"{self.baseurls['match']}/lol/match/v5/matches/{match_id}"
        return await self._make_request(url)

    async def get_comprehensive_player_data(self, game_name: str, tag_line: str, match_count: int = 100) -> Optional[Dict]:
        try:
            player_info = await self.get_player_info(game_name, tag_line)
            if not player_info:
                return {"error": f"Player {game_name}#{tag_line} not found"}
            match_ids = await self.get_match_history(player_info.puuid, match_count)
            if not match_ids:
                return {"error": "No matches found"}
            matches = []
            batch_size = 10
            for i in range(0, len(match_ids), batch_size):
                batch_ids = match_ids[i:i + batch_size]
                match_tasks = [self.get_match_details(match_id) for match_id in batch_ids]
                timeline_tasks = [self.get_match_timeline(match_id) for match_id in batch_ids]
                match_results = await asyncio.gather(*match_tasks)
                timeline_results = await asyncio.gather(*timeline_tasks)
                for match_data, timeline in zip(match_results, timeline_results):
                    if match_data and match_data.get('info'):
                        match_data['timeline'] = timeline
                        player_match_data = self._extract_player_match_data(match_data, player_info.puuid)
                        if player_match_data:
                            player_match_data['timeline'] = timeline
                            # Store the full match data for advanced analysis (like death heatmaps)
                            player_match_data['full_match_data'] = match_data
                            matches.append(player_match_data)
                await asyncio.sleep(0.1)
            return {
                "player_info": {
                    "puuid": player_info.puuid,
                    "name": f"{game_name}#{tag_line}",
                    "summoner_level": player_info.summoner_level,
                    "profile_icon": player_info.profile_icon_id
                },
                "matches": matches,
                "total_matches": len(matches),
                "requested_matches": match_count
            }
        except Exception as e:
            logging.error(f"Error getting player data: {e}")
            return {"error": str(e)}

    def _extract_player_match_data(self, match_data: Dict, puuid: str) -> Optional[Dict]:
        info = match_data.get('info', {})
        participants = info.get('participants', [])
        player_data = None
        for participant in participants:
            if participant.get('puuid') == puuid:
                player_data = participant
                break
        if not player_data:
            return None
        return {
            'gameId': match_data.get('metadata', {}).get('matchId'),
            'gameCreation': info.get('gameCreation'),
            'gameDuration': info.get('gameDuration'),
            'gameMode': info.get('gameMode'),
            'queueId': info.get('queueId'),
            'championId': player_data.get('championId'),
            'championName': player_data.get('championName'),
            'teamPosition': player_data.get('teamPosition'),
            'teamId': player_data.get('teamId'),
            'kills': player_data.get('kills'),
            'deaths': player_data.get('deaths'),
            'assists': player_data.get('assists'),
            'totalMinionsKilled': player_data.get('totalMinionsKilled'),
            'neutralMinionsKilled': player_data.get('neutralMinionsKilled'),
            'goldEarned': player_data.get('goldEarned'),
            'totalDamageDealtToChampions': player_data.get('totalDamageDealtToChampions'),
            'totalDamageTaken': player_data.get('totalDamageTaken'),
            'visionScore': player_data.get('visionScore'),
            'wardsPlaced': player_data.get('wardsPlaced'),
            'wardsKilled': player_data.get('wardsKilled'),
            'turretKills': player_data.get('turretKills'),
            'dragonKills': player_data.get('dragonKills'),
            'baronKills': player_data.get('baronKills'),
            'win': player_data.get('win'),
            'teamKills': sum(p.get('kills', 0) for p in participants if p.get('teamId') == player_data.get('teamId')),
            'teamDeaths': sum(p.get('deaths', 0) for p in participants if p.get('teamId') == player_data.get('teamId')),
        }
# python3 -m streamlit run app.py