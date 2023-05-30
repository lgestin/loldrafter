import torch
import json, random
from pathlib import Path
from torch.utils.data import Dataset

from scraper.data import Game


class LabelEncoder:
    def __init__(self):
        pass


class DraftDataset(Dataset):
    def __init__(self, path_to_data: Path):
        with open(path_to_data, "r") as f:
            games = json.load(f)
        self.data = [Game(**game) for game in games]

        self.labels, self.n = {}, {}
        with open(path_to_data.parent / "labels" / "teams.json", "r") as f:
            self.labels["teams"] = json.load(f)
        self.n["teams"] = max(v for v in self.labels["teams"].values())
        with open(path_to_data.parent / "labels" / "champions.json", "r") as f:
            self.labels["champions"] = json.load(f)
        self.n["champions"] = max(v for v in self.labels["champions"].values())
        with open(path_to_data.parent / "labels" / "patchs.json", "r") as f:
            self.labels["patchs"] = json.load(f)
        self.n["patchs"] = max(v for v in self.labels["patchs"].values())

    def __len__(self):
        return 10_000_000

    def __getitem__(self, index):
        # BE CAREFUL everytime this is calle, seeds are reset FOR THE WHOLE PROGRAM
        torch.random.manual_seed(index)
        random.seed(index)

        i = random.randint(0, len(self.data) - 1)
        game = self.data[i]

        sides = ["blue", "red"]
        random.shuffle(sides)
        ally_side, opp_side = sides
        ally_team, opp_team = getattr(game, ally_side), getattr(game, opp_side)

        ally_team_name = self.labels["teams"][ally_team.name]
        ally_picks = [self.labels["champions"][c] for c in ally_team.picks]
        ally_bans = [self.labels["champions"][c] for c in ally_team.bans]

        opp_team_name = self.labels["teams"][opp_team.name]
        opp_picks = [self.labels["champions"][c] for c in opp_team.picks]
        opp_bans = [self.labels["champions"][c] for c in opp_team.bans]

        lane_to_pick = random.randint(0, 4)  # top jg mid adc sup

        ally_picks = torch.as_tensor(ally_picks)
        opp_picks = torch.as_tensor(opp_picks)
        picks = torch.stack([ally_picks, opp_picks])

        ally_bans = torch.as_tensor(ally_bans)
        opp_bans = torch.as_tensor(opp_bans)
        bans = torch.stack([ally_bans, opp_bans], dim=0)
        # MASKS


        # WIP Make training BERT like so that it is easy to sample
        n_champs_masked = random.randint(0, 5)
                


        return {"picks": picks, "bans": bans}


if __name__ == "__main__":
    dataset = DraftDataset(Path("/data/data.json"))

    print(dataset[5], dataset[10])
