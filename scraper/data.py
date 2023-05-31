from dataclasses import dataclass
from typing import List


@dataclass
class Team:
    name: str
    side: str
    picks: List[str]
    picks_order: List[int]
    bans: List[str]
    bans_order: List[int]


@dataclass
class Game:
    league: str
    year: str
    split: str
    date: str
    competition: str
    patch: str
    winner: str
    blue: Team
    red: Team

    def __init__(
        self,
        league: str,
        year: str,
        split: str,
        date: str,
        competition: str,
        patch: str,
        blue: Team,
        red: Team,
        winner: str,
    ):
        self.league = league
        self.year = year
        self.split = split
        self.date = date
        self.competition = competition
        self.patch = patch
        self.winner = winner

        if isinstance(blue, dict):
            blue = Team(**blue)
        self.blue = blue

        if isinstance(red, dict):
            red = Team(**red)
        self.red = red
