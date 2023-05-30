from dataclasses import dataclass
from typing import List


@dataclass
class Team:
    name: str
    side: str
    picks: List[str]
    bans: List[str]


@dataclass
class Game:
    league: str
    date: str
    patch: str
    winner: str
    blue: Team
    red: Team

    def __init__(
        self, league: str, date: str, patch: str, blue: Team, red: Team, winner: str
    ):
        self.league = league
        self.date = date
        self.patch = patch
        self.winner = winner

        if isinstance(blue, dict):
            blue = Team(**blue)
        self.blue = blue

        if isinstance(red, dict):
            red = Team(**red)
        self.red = red
