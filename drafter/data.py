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
        
        random.seed(0)
        random.shuffle(self.data)

        # filter out sample for which we don't have pick/ban order
        # TODO split data, split on augmentation isn't enough
        games = [game for game in self.data if game.blue.picks_order is not None]
        self.data = games

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

        # DRAFT MASKS
        # DraftPhase has n possible phases
        # 0:
        #   PICKS: 0 0 0 0 0 | 0 0 0 0 0
        #   BANS:  1 1 1 0 0 | 1 1 1 0 0

        # 1:
        #   PICKS: 1 0 0 0 0 | 0 0 0 0 0
        #   BANS:  1 1 1 0 0 | 1 1 1 0 0

        # 2:
        #   PICKS: 1 0 0 0 0 | 1 1 0 0 0
        #   BANS:  1 1 1 0 0 | 1 1 1 0 0

        # 3:
        #   PICKS: 1 1 1 0 0 | 1 1 0 0 0
        #   BANS:  1 1 1 0 0 | 1 1 1 0 0

        # 4:
        #   PICKS: 1 1 1 0 0 | 1 1 1 0 0
        #   BANS:  1 1 1 1 1 | 1 1 1 1 1

        # 5:
        #   PICKS: 1 1 1 0 0 | 1 1 1 1 0
        #   BANS:  1 1 1 1 1 | 1 1 1 1 1

        # 6:
        #   PICKS: 1 1 1 1 1 | 1 1 1 1 0
        #   BANS:  1 1 1 1 1 | 1 1 1 1 1

        blue_picks_state = torch.tensor([0, 1, 1, 3, 3, 3, 5, 5])
        red_picks_state = torch.tensor([0, 0, 2, 2, 3, 4, 4, 5])
        blue_bans_state = torch.tensor([3, 3, 3, 3, 5, 5, 5])
        red_bans_state = torch.tensor([3, 3, 3, 3, 5, 5, 5])
        # number of champions picked at every stage
        self.n_draft = {
            "picks": torch.stack([blue_picks_state, red_picks_state]),
            "bans": torch.stack([blue_bans_state, red_bans_state]),
        }

    def __len__(self):
        return 1_000_000

    def __getitem__(self, index):
        # BE CAREFUL everytime this is called, seeds are reset FOR THE WHOLE PROGRAM

        i = index % len(self.data)
        game = self.data[i]

        random.seed(index)
        teams = [game.blue, game.red]

        picks = [[self.labels["champions"][p] for p in t.picks] for t in teams]
        picks = torch.as_tensor(picks)
        picks_order = torch.as_tensor([t.picks_order for t in teams])

        bans = [[self.labels["champions"][b] for b in t.bans] for t in teams]
        bans = torch.as_tensor(bans)
        bans_order = torch.as_tensor([t.bans_order for t in teams])

        draft_state = random.choice(range(6))

        n_picks = self.n_draft["picks"][:, draft_state][:, None]
        picks_mask = picks_order < n_picks

        n_bans = self.n_draft["bans"][:, draft_state][:, None]
        bans_mask = bans_order < n_bans

        n_next_picks = self.n_draft["picks"][:, draft_state + 1][:, None]
        next_picks_mask = picks_order < n_next_picks

        target_mask = (picks_mask.float() - next_picks_mask.float()).bool()
        assert 0 < target_mask.sum().item() <= 2

        data = {
            "picks": picks,
            "bans": bans,
            "picks_mask": picks_mask,
            "bans_mask": bans_mask,
            "target_mask": target_mask,
        }

        # flip all tensors if side is red
        if draft_state % 2 == 1:
            data = {k: v.flip(0) for k, v in data.items()}

        return data


if __name__ == "__main__":
    dataset = DraftDataset(Path("/data/data.json"))

    print(dataset[5], dataset[10])
