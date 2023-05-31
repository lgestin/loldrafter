import requests
from bs4 import BeautifulSoup

from data import Game, Team


# [ ]
class ProGameScraper:
    def __init__(self) -> None:
        self.url = "https://lol.fandom.com/wiki"

    def run(self, league: str, year: int, split: str, competition: str):
        url = f"{self.url}/{league}/{year}_Season/{split}_{competition}"
        match_page = self.get_page(url=f"{url}/Match_History")
        pickban_page = self.get_page(url=f"{url}/Picks_and_Bans")

        if not match_page.ok or not pickban_page.ok:
            return []

        matches = self.parse_page(match_page)
        pickbans = self.parse_page(pickban_page)

        get_data = lambda col: col.find_all("a")[0].get("title")

        def get_order(pickban, global_order):
            order = [global_order.index(champ) for champ in pickban]
            sorder = sorted(order)
            return [sorder.index(o) for o in order]

        games, unmatched_games = [], []
        for match, pickban in zip(matches[3:], pickbans[2:]):
            matchcols = match.find_all("td")
            pickbancols = pickban.find_all("td")

            pickban_order = self.parse_pickban_row(pickbancols[5:-2])

            bans = self.get_bans_from_cols(matchcols)
            picks = self.get_picks_from_cols(matchcols)

            # chech that both rows contain same match:
            pickban_from_match = set(bans[0] + bans[1]) | set(picks[0] + picks[1])
            if not pickban_from_match == set(pickban_order):
                unmatched_games.append((matchcols, pickbancols))
                # TODO match back unmatched games
                continue

            game = Game(
                league=league,
                year=year,
                split=split,
                date=self.get_date_from_cols(matchcols),
                competition=competition,
                patch=get_data(matchcols[1]),
                blue=Team(
                    name=get_data(matchcols[2]),
                    side="blue",
                    picks=picks[0],
                    picks_order=get_order(picks[0], pickban_order),
                    bans=bans[0],
                    bans_order=get_order(bans[0], pickban_order),
                ),
                red=Team(
                    name=get_data(matchcols[3]),
                    side="red",
                    picks=picks[1],
                    picks_order=get_order(picks[1], pickban_order),
                    bans=bans[1],
                    bans_order=get_order(bans[1], pickban_order),
                ),
                winner=get_data(matchcols[4]),
            )
            games.append(game)

        print(f"{len(games)} found, {len(unmatched_games)} failed to match")

        return games

    def get_page(self, url):
        page = requests.get(url)
        return page

    def parse_page(self, page):
        soup = BeautifulSoup(page.content, "html.parser")

        table = soup.find_all("table")[-1]
        return table.find_all("tr")

    def parse_pickban_header(self, header):
        offset = 0
        order = []
        for i, o in enumerate(header):
            if len(o[2:]) == 1:
                order += [i + offset]
            elif len(o[2:]) > 1:
                n = int(o[-1]) - int(o[-3]) + 1
                order += [[i + offset for _ in range(n)]]
                offset += n - 1
        return order

    def parse_pickban_row(self, row):
        order = []
        for col in row:
            for c in col.find_all("span")[::2]:
                order += [c.find_all("span")[0].get("title")]
        return order

    def get_date_from_cols(self, cols):
        return cols[0].text

    def get_picks_from_cols(self, cols):
        blue, red = cols[7:9]

        red = [span.get("title") for span in red.find_all("span")]
        blue = [span.get("title") for span in blue.find_all("span")]

        return red, blue

    def get_bans_from_cols(self, cols):
        blue, red = cols[5:7]

        red = [span.get("title") for span in red.find_all("span")]
        blue = [span.get("title") for span in blue.find_all("span")]

        return red, blue


if __name__ == "__main__":
    scraper = ProGameScraper()
    scraper.run("LCS", 2019, "Spring")
