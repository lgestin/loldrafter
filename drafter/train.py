import json
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
    n_samples = len(dataset.data)
    train_indices = [i for i in indices if i % n_samples > val_size]
    val_indices = [i for i in indices if i % n_samples <= val_size][:val_size]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_data = iter(torch.utils.data.DataLoader(train_set, batch_size=batch_size))
    val_data = torch.utils.data.DataLoader(val_set, batch_size=16)

    model = Drafter(d_model=d_model, n_embeddings=dataset.n, dropout=dropout)

    assert torch.cuda.is_available()
    model = model.cuda()

    def move_batch_to_device(batch, device="cuda"):
        return {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(exp_path)

    def loop(batch):
        if model.training:
            batch = dataset.augment(batch, p=0.15)
        if cuda:
            batch = move_batch_to_device(batch=batch, device="cuda")

        batch_metrics = {}
        with torch.set_grad_enabled(model.training):
            B, _, T = batch["picks"].shape
            target_mask = batch.pop("target_mask")[:, 0]
            logits = model(**batch)
            target = batch["picks"][:, 0]

            # masking
            masked_logits = logits.masked_fill(~target_mask[..., None], 0)
            masked_target = target.masked_fill(~target_mask, 0)

            # loss computation
            loss_ce = F.cross_entropy(
                masked_logits.view(B * T, -1), masked_target.view(B * T)
            )
            batch_metrics["loss_ce"] = loss_ce

            if model.training:
                optimizer.zero_grad()
                loss_ce.backward()
                optimizer.step()
            else:
                # evaluation of the model. Compute accuracy of the predictions
                already_picks = batch["picks"].masked_fill(~batch["picks_mask"], 0)
                bans = batch["bans"].masked_fill(~batch["bans_mask"], 0)
                probs, pred = model.predict_from_logits(
                    logits, already_picks=already_picks, bans=bans
                )
                accuracy = (pred == batch["picks"][:, 0]).float().mean(1).mean(0)
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
            val_batch = move_batch_to_device(val_batch)
            with torch.no_grad():
                pick, picks, updated_picks = model.sample_next(
                    picks=val_batch["picks"],
                    bans=val_batch["bans"],
                    picks_mask=val_batch["picks_mask"],
                    bans_mask=val_batch["bans_mask"],
                )
            champions_dict = get_champions_dict(data_path)
            for k, (b, m, op, ap, up, gt) in enumerate(
                zip(
                    int_to_champ(val_batch["bans"].view(-1, 10), d=champions_dict),
                    val_batch["picks_mask"],
                    int_to_champ(picks[:, 1], d=champions_dict),
                    int_to_champ(picks[:, 0], d=champions_dict),
                    int_to_champ(updated_picks, d=champions_dict),
                    int_to_champ(val_batch["picks"][:, 0], d=champions_dict),
                )
            ):
                text = f'bans: {" ".join(b)}  '
                text += f'\ndraft_mask: {" ".join([str(s) for s in m.tolist()])}  '
                text += f'\nopp_picks: {" ".join(op)}  '
                text += f'\nally_picks: {" ".join(ap)}  '
                text += f'\nupdt_picks: {" ".join(up)}  '
                text += f'\ngt_picks: {" ".join(gt)}'
                writer.add_text(f"draft_sample_{k}", text, i)


def get_champions_dict(data_path: Path):
    with open("/data/labels/champions.json", "r") as f:
        d = json.load(f)
    d["UNK"] = 0
    dinv = {v: k for k, v in d.items()}
    return dinv


def int_to_champ(batch, d: dict):
    champs = []
    for item in batch:
        champs.append([d[i.item()] for i in item])
    return champs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--exp_path", type=Path, required=True)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
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
