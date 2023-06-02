import requests
from bs4 import BeautifulSoup

from data import Game, Team


# [ ]
class ProGameScraper:
    def __init__(self) -> None:
        self.url = "https://lol.fandom.com/wiki"

    def run(self, league: str, year: int, split: str, competition: str):
        # TODO: turn that into pandas database to make it more efficient
        url = f"{self.url}/{league}/{year}_Season/{split}_{competition}"
        match_page = self.get_page(url=f"{url}/Match_History")
        pickban_page = self.get_page(url=f"{url}/Picks_and_Bans")

        if not match_page.ok:
            return []

        match_rows = self.parse_page(match_page)
        pickban_rows = self.parse_page(pickban_page)

        get_data = lambda col: col.find_all("a")[0].get("title")

        # Parse games
        games = []
        for match in match_rows[3:]:
            game = {
                "league": league,
                "year": year,
                "split": split,
                "competition": competition,
            }

            matchcols = match.find_all("td")
            game["date"] = matchcols[0].text
            game["patch"] = get_data(matchcols[1])
            game["winner"] = get_data(matchcols[4])

            teams = {"red": {"side": "red"}, "blue": {"side": "blue"}}

            teams["red"]["name"] = get_data(matchcols[3])
            teams["blue"]["name"] = get_data(matchcols[2])

            picks = self.get_picks_from_cols(matchcols)
            teams["blue"]["picks"], teams["red"]["picks"] = picks

            bans = self.get_bans_from_cols(matchcols)
            teams["blue"]["bans"], teams["red"]["bans"] = bans

            game.update(teams)
            games.append(game)

        # Parse pcikbans order
        pickban_page = self.get_page(url=f"{url}/Picks_and_Bans")
        pickban_rows = self.parse_page(pickban_page)

        pickbans = []
        for match in pickban_rows[2:]:
            pickbancols = match.find_all("td")

            pickban = {}
            pickban["blue"] = pickbancols[1].get("title")
            pickban["red"] = pickbancols[2].get("title")
            pickban["patch"] = get_data(pickbancols[4])
            pickban["order"] = self.parse_pickban_row(pickbancols[5:-2])

            pickbans.append(pickban)

        def get_order(pickban, global_order):
            order = [global_order.index(champ) for champ in pickban]
            sorder = sorted(order)
            return [sorder.index(o) for o in order]

        # Match pickbans to games to add data
        split = []
        for game in games:
            pickban_candidates = []
            for pickban in pickbans:
                if game["blue"]["name"] != pickban["blue"]:
                    continue
                if game["red"]["name"] != pickban["red"]:
                    continue
                if game["patch"] != pickban["patch"]:
                    continue
                match_pickban = (
                    game["blue"]["picks"]
                    + game["red"]["picks"]
                    + game["blue"]["bans"]
                    + game["red"]["bans"]
                )

                if not (set(match_pickban) == set(pickban["order"])):
                    continue
                else:
                    pickban_candidates.append(pickban)

            if len(pickban_candidates) == 1:
                pickban = pickban_candidates[0]["order"]
                game["red"]["picks_order"] = get_order(game["red"]["picks"], pickban)
                game["red"]["bans_order"] = get_order(game["red"]["bans"], pickban)
                game["blue"]["picks_order"] = get_order(game["blue"]["picks"], pickban)
                game["blue"]["bans_order"] = get_order(game["blue"]["bans"], pickban)
            else:
                # If can't be matched or if multiple matching, then consider the data missing
                game["red"]["picks_order"], game["red"]["bans_order"] = None, None
                game["blue"]["picks_order"], game["blue"]["bans_order"] = None, None
            game = Game(**game)
            split.append(game)

        return split

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
