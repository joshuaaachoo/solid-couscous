from typing import Dict, List


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _aggregate_basic_stats(matches: List[Dict]) -> Dict[str, float]:
    total_games = len(matches)
    wins = sum(1 for m in matches if m.get('win', False))
    kills = sum(m.get('kills', 0) for m in matches)
    deaths = sum(m.get('deaths', 0) for m in matches)
    assists = sum(m.get('assists', 0) for m in matches)
    dmg = sum(m.get('totalDamageDealtToChampions', 0) for m in matches)
    vision = sum(m.get('visionScore', 0) for m in matches)
    cs = sum((m.get('totalMinionsKilled', 0) + m.get('neutralMinionsKilled', 0)) for m in matches)
    time_min = _safe_div(sum(m.get('gameDuration', 1800) for m in matches), 60)

    return {
        'games': float(total_games),
        'wins': float(wins),
        'win_rate': _safe_div(wins, total_games),
        'kills': float(kills),
        'deaths': float(deaths),
        'assists': float(assists),
        'avg_kda': _safe_div(kills + assists, max(deaths, 1)),
        'avg_deaths': _safe_div(deaths, total_games),
        'avg_damage': _safe_div(dmg, max(total_games, 1)),
        'vision_per_min': _safe_div(vision, max(time_min, 1)),
        'cs_per_min': _safe_div(cs, max(time_min, 1)),
    }


def _compute_awards(matches: List[Dict], stats: Dict[str, float]) -> List[str]:
    awards: List[str] = []

    # Vision Goblin
    if stats['vision_per_min'] >= 1.2:
        awards.append("Vision Goblin: you see everything before it sees you")

    # KDA Monster
    if stats['avg_kda'] >= 3.5 and stats['avg_deaths'] <= 4.0:
        awards.append("KDA Monster: clinical fights, minimum donations")

    # CS Machine
    if stats['cs_per_min'] >= 7.0:
        awards.append("CS Machine: money printer goes brrr")

    # Objective Enjoyer (heuristic via damage and wins)
    if stats['win_rate'] >= 0.55 and stats['avg_damage'] >= 22000:
        awards.append("Objective Enjoyer: fights when it matters")

    # Coinflip King (high deaths, volatile)
    if stats['avg_deaths'] >= 7.0 and stats['win_rate'] >= 0.45 and stats['win_rate'] <= 0.55:
        awards.append("Coinflip King: either 1v9 or spectator mode")

    # Tower Tapper (plates proxy via high cs/min and wins)
    if stats['cs_per_min'] >= 6.2 and stats['win_rate'] >= 0.5:
        awards.append("Plate Collector: tower gold enthusiast")

    return awards[:5]


def _compute_clutch_meter(stats: Dict[str, float]) -> Dict[str, object]:
    # Blend survivability and contribution proxies
    low_deaths_factor = max(0.0, 1.0 - stats['avg_deaths'] / 8.0)
    kda_factor = min(1.0, stats['avg_kda'] / 4.0)
    win_factor = stats['win_rate']
    score = int(round(100.0 * (0.4 * kda_factor + 0.4 * low_deaths_factor + 0.2 * win_factor)))
    label = (
        "Ice Cold" if score >= 85 else
        "Closer" if score >= 70 else
        "Sometimes Clutch" if score >= 55 else
        "Work In Progress"
    )
    return {"score": score, "label": label}


def _compute_tilt_risk(stats: Dict[str, float]) -> Dict[str, object]:
    # Heuristic: more deaths and lower WR => higher tilt risk
    death_term = min(1.0, stats['avg_deaths'] / 8.0)
    loss_term = 1.0 - stats['win_rate']
    score = int(round(100.0 * (0.6 * loss_term + 0.4 * death_term)))
    label = (
        "Stable" if score <= 30 else
        "Watchful" if score <= 55 else
        "High" if score <= 75 else
        "Danger"
    )
    return {"score": score, "label": label}


def _meme_line(stats: Dict[str, float]) -> str:
    if stats['cs_per_min'] >= 7.5 and stats['avg_deaths'] <= 4.0:
        return "Lane diff printed a receipt."
    if stats['vision_per_min'] >= 1.4:
        return "Enemy jungler reported: ‘spotted in 4k UHD.’"
    if stats['avg_deaths'] >= 8.0:
        return "You didn’t facecheck the bush. The bush facechecked you."
    if stats['avg_kda'] >= 4.0:
        return "Hands checked: still warm."
    if stats['win_rate'] >= 0.6:
        return "LP delivery service: on-time and insured."
    return "We queue for improvement, we stay for the content."


def generate_fun_insights(player_data: Dict) -> Dict:
    """Compute lighthearted, zero-ML fun insights from recent matches.

    Output:
    {
      "awards": [str, ...],
      "clutch_meter": {"score": int, "label": str},
      "tilt_risk": {"score": int, "label": str},
      "meme": str
    }
    """
    matches = player_data.get('matches', []) or []
    if not matches:
        return {
            'awards': [],
            'clutch_meter': {'score': 50, 'label': 'Sometimes Clutch'},
            'tilt_risk': {'score': 50, 'label': 'Watchful'},
            'meme': 'Play a couple games to unlock fun insights!'
        }

    stats = _aggregate_basic_stats(matches)
    awards = _compute_awards(matches, stats)
    clutch = _compute_clutch_meter(stats)
    tilt = _compute_tilt_risk(stats)
    meme = _meme_line(stats)

    return {
        'awards': awards,
        'clutch_meter': clutch,
        'tilt_risk': tilt,
        'meme': meme,
    }


