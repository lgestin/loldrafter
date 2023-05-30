import requests
from bs4 import BeautifulSoup

from data import Game, Team


class ProGameScraper:
    def __init__(self) -> None:
        self.url = "https://lol.fandom.com/wiki"

    def run(self, league: str, year: int, split: str):
        url = f"{self.url}/{league}/{year}_Season/{split}_Season/Match_History"
        page = self.get_page(url=url)

        get_data = lambda col: col.find_all("a")[0].get("title")

        games = []
        for row in self.parse_page(page.content):
            cols = row.find_all("td")

            bans = self.get_bans_from_cols(cols)
            picks = self.get_picks_from_cols(cols)
            game = Game(
                league=league,
                date=self.get_date_from_cols(cols),
                patch=get_data(cols[1]),
                blue=Team(
                    name=get_data(cols[2]),
                    side="blue",
                    picks=picks[0],
                    bans=bans[0],
                ),
                red=Team(
                    name=get_data(cols[3]),
                    side="red",
                    picks=picks[1],
                    bans=bans[1],
                ),
                winner=get_data(cols[4]),
            )
            games.append(game)

        return games

    def get_page(self, url):
        page = requests.get(url)
        return page

    def parse_page(self, page):
        soup = BeautifulSoup(page, "html.parser")

        table = soup.find_all("table")[-1]
        for row in table.find_all("tr")[3:]:
            yield row

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
