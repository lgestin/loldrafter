import json, os
from scraper import ProGameScraper
from pathlib import Path

SPLITS = ["Spring", "Summer"]


def save(games, dump_path: Path):
    with open(dump_path, "w") as f:
        f.write(json.dumps(games, default=lambda g: g.__dict__, indent=4))
    return


def get_labels(games: list):
    league_labels = set()
    team_labels = set()
    champion_labels = set()
    patch_labels = set()

    for game in games:
        league_labels.update([game.league])
        team_labels.update(set([game.blue.name, game.blue.name]))
        game_champions = (
            game.blue.picks + game.blue.bans + game.red.picks + game.red.bans
        )
        champion_labels.update(set(game_champions))
        patch_labels.update(set([game.patch]))

    return league_labels, team_labels, champion_labels, patch_labels


def dump_labels(labels, dump_path):
    labels = {l: i + 1 for i, l in enumerate(labels)}
    with open(dump_path, "w") as f:
        f.write(json.dumps(labels, indent=4))
    return labels


def main(dump_path: Path):
    scraper = ProGameScraper()

    games = []
    for league in ["LEC", "LPL", "LCS", "LCK", "LFL"]:
        for year in range(2019, 2024):
            for split in ["Spring", "Summer"]:
                for competition in ["Season", "Playoffs"]:
                    print("scraping", league, year, split, competition)
                    games += scraper.run(
                        league=league, year=year, split=split, competition=competition
                    )

    save(games, dump_path=dump_path / f"data.json")

    league_labels, team_labels, champion_labels, patch_labels = get_labels(games)
    (dump_path / "labels").mkdir(exist_ok=True)
    dump_labels(league_labels, dump_path=dump_path / "labels" / "leagues.json")
    dump_labels(team_labels, dump_path=dump_path / "labels" / "teams.json")
    dump_labels(champion_labels, dump_path=dump_path / "labels" / "champions.json")
    dump_labels(patch_labels, dump_path=dump_path / "labels" / "patchs.json")

    print(len(games))
    return games


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_path", type=Path, required=True)
    options = parser.parse_args()

    main(dump_path=options.dump_path)
