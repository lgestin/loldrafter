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

        draft_masks = {
            "blue": torch.zeros(3, 2, 5).bool(),
            "red": torch.zeros(3, 2, 5).bool(),
        }
        arange = torch.arange(5)
        draft_masks["blue"][0] = torch.stack([arange < 0, arange < 0], dim=0)
        draft_masks["blue"][1] = torch.stack([arange < 1, arange < 2], dim=0)
        draft_masks["blue"][2] = torch.stack([arange < 3, arange < 4], dim=0)
        draft_masks["red"][0] = torch.stack([arange < 0, arange < 1], dim=0)
        draft_masks["red"][1] = torch.stack([arange < 2, arange < 3], dim=0)
        draft_masks["red"][2] = torch.stack([arange < 4, arange < 5], dim=0)
        self.draft_masks = draft_masks

    def __len__(self):
        return 10_000_000

    def __getitem__(self, index):
        # BE CAREFUL everytime this is called, seeds are reset FOR THE WHOLE PROGRAM
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

        ally_picks = torch.as_tensor(ally_picks)
        opp_picks = torch.as_tensor(opp_picks)
        picks = torch.stack([ally_picks, opp_picks])

        ally_bans = torch.as_tensor(ally_bans)
        opp_bans = torch.as_tensor(opp_bans)
        bans = torch.stack([ally_bans, opp_bans], dim=0)
        # MASKS

        # WIP Make training BERT like so that it is easy to sample
        # In non pro games 3 possible states when picking champions
        # if red:
        #   0 0 0 0 0
        #   1 1 0 0 0
        #   1 1 1 1 0
        # if blue:
        #   0 0 0 0 0
        #   1 0 0 0 0
        #   1 1 1 0 0
        # TODO: maybe not restrain to draft mask, but any random mask during training
        draft_state = random.randint(0, 2)
        draft_mask = self.draft_masks[ally_side][draft_state].clone()

        # make random which position to chose
        perm = list(range(5))
        for i in range(2):
            random.shuffle(perm)
            draft_mask[i] = draft_mask[i, perm]

        return {"picks": picks, "draft_mask": draft_mask, "bans": bans}


if __name__ == "__main__":
    dataset = DraftDataset(Path("/data/data.json"))

    print(dataset[5], dataset[10])
