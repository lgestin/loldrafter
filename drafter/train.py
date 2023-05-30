import torch
import torch.nn.functional as F

from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from data import DraftDataset
from drafter import Drafter


def train(
    data_path: Path,
    exp_path: Path,
    d_model: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    val_size: int,
    n_iter_val: int,
    max_iter: int,
):
    cuda = torch.cuda.is_available()

    dataset = DraftDataset(data_path)
    indices = list(range(len(dataset)))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_data = iter(torch.utils.data.DataLoader(train_set, batch_size=batch_size))
    val_data = torch.utils.data.DataLoader(val_set, batch_size=16)

    model = Drafter(d_model=d_model, n_embeddings=dataset.n, dropout=dropout)
    import ipdb

    ipdb.set_trace()

    if torch.cuda.is_available():
        model = model.cuda()

        def move_batch_to_device(batch, device="cuda"):
            return {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(exp_path)

    def loop(batch):
        if cuda:
            batch = move_batch_to_device(batch=batch, device="cuda")

        batch_metrics = {}
        with torch.set_grad_enabled(model.training):
            logits = model(**batch)
            loss_ce = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), batch["picks"].view(-1)
            )
            batch_metrics["loss_ce"] = loss_ce

            if model.training:
                optimizer.zero_grad()
                loss_ce.backward()
                optimizer.step()
            else:
                probs, pred = model.predict_from_logits(logits, bans=batch["bans"])
                accuracy = (pred == batch["picks"]).float().mean(1).mean(0)
                batch_metrics["accuracy"] = accuracy

        return batch_metrics

    for i in range(max_iter):
        model.train()
        batch = next(train_data)

        batch_metrics = loop(batch=batch)

        # log training metrics
        for k, v in batch_metrics.items():
            writer.add_scalar(f"train/{k}", v, i)

        # compute and log val metrics
        if i % n_iter_val == 0:
            model.eval()
            val_metrics = defaultdict(list)

            for val_batch in val_data:
                batch_metrics = loop(batch=val_batch)
                for k, v in batch_metrics.items():
                    val_metrics[k].append(v)

            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", torch.stack(v).mean(), i)

            # Sample from the model
            with torch.no_grad():
                pick_order = torch.stack(
                    [torch.randperm(5) for _ in range(6)], dim=0
                ).to("cuda")
                picks = model.sample(
                    pick_order=pick_order, bans=val_batch["bans"][:6].cuda()
                )
            for k, (p, c, b) in enumerate(
                zip(pick_order, int_to_champ(picks), int_to_champ(val_batch["bans"][:6].view(-1, 10)))
            ):
                text = f'bans: {" ".join(b)}  \npick_order:{" ".join([str(s) for s in p.tolist()])}  \npicks: {" ".join(c)}'
                writer.add_text(f"draft_sample_{k}", text, i)


import json

with open("/data/labels/champions.json", "r") as f:
    d = json.load(f)
d = {v: k for k, v in d.items()}


def int_to_champ(batch):
    champs = []
    for item in batch:
        champs.append([d[i.item()] for i in item])
    return champs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--exp_path", type=Path, required=True)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--val_size", type=int, default=300)
    parser.add_argument("--n_iter_val", type=int, default=1000)
    parser.add_argument("--max_iter", type=int, default=100000)

    options = parser.parse_args()

    train(
        data_path=options.data_path,
        exp_path=options.exp_path,
        d_model=options.d_model,
        batch_size=options.batch_size,
        learning_rate=options.learning_rate,
        dropout=options.dropout,
        val_size=options.val_size,
        n_iter_val=options.n_iter_val,
        max_iter=options.max_iter,
    )
