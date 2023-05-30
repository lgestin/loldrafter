import torch
import torch.nn as nn

from modules import PositionalEncoding, TransformerLayer


class Drafter(nn.Module):
    def __init__(
        self, d_model: int, n_embeddings: dict, n_layers: int = 3, dropout: float = 0.0
    ):
        super().__init__()

        self.embeddings = nn.ModuleDict()
        for k, n in n_embeddings.items():
            self.embeddings[k] = nn.Embedding(n + 1, d_model)
        self.embeddings["pos"] = nn.Embedding(5, d_model)

        self.positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=5
        )

        self.in_layer = nn.Linear(2 * d_model, d_model)

        bans_encoder = [
            TransformerLayer(d_model=d_model, n_heads=1, dropout=dropout)
            for _ in range(n_layers // 2)
        ]
        self.bans_encoder = nn.Sequential(*bans_encoder)

        draft_decoder = [
            TransformerLayer(
                d_model=d_model, n_heads=1, dropout=dropout, cross_attn=True
            )
            for _ in range(n_layers)
        ]
        self.draft_decoder = nn.ModuleList(draft_decoder)

        self.out_layer = nn.Linear(d_model, n_embeddings["champions"] + 1)

    def forward(self, picks, pick_order, bans):
        # conditioning
        # - Bans
        # - Already picked champs (allied / ennemy)
        # - Lane of the champ to pick
        # - Side, Patch, League, Team

        # input
        # picks [12, 35, 18, 121, 5] champions picked
        # next_pos_picked [4, 1, 2, 0, 3] first pick is adc, 2nd pick is jg...
        B, T = picks.shape

        # Encode bans
        # bans (B, 2, T) 2 is because both team ban
        # TODO: Check if views do as expected
        bans = self.embeddings["champions"](bans).view(B * 2, T, -1)
        bans = self.positional_encoding(bans)
        bans = bans.view(B, 2 * T, -1)

        x_cond = self.bans_encoder(bans)

        # Encode picks
        picks = self.embeddings["champions"](picks)
        input_picks = torch.cat(
            [torch.zeros_like(picks[:, :1, :]), picks[:, :-1, :]], dim=1
        )
        pos = self.embeddings["pos"](pick_order)

        x = torch.cat([pos, input_picks], dim=-1)

        logits = self.model(x, x_cond)

        return logits

    def sample(self, picks=None, pick_order=None, bans=None):
        # Autoregressive Sampling
        B, _, T = bans.shape
        x_cond = self.embeddings["champions"](bans).view(B * 2, T, -1)
        x_cond = self.positional_encoding(x_cond)
        x_cond = x_cond.view(B, 2 * T, -1)

        x_cond = self.bans_encoder(x_cond)

        if picks is None:
            input_picks = torch.zeros(
                (B, 1, self.embeddings["champions"].weight.shape[-1])
            ).to(pick_order.device)
        else:
            picks = self.embeddings["champions"](picks)
            input_picks = torch.cat(
                [torch.zeros_like(picks[:, :1]), picks], dim=1
            )

        assert pick_order is not None
        pos = self.embeddings["pos"](pick_order)

        draft = []
        # pick order contains the picks we want to do during sampling
        for i in range(pos.shape[1]):
            x = torch.cat([pos[:, :i+1], input_picks], dim=-1)
            logits = self.model(x, x_cond)
            _, pred = self.predict_from_logits(logits=logits, bans=bans)
            draft.append(pred[:, -1])
            pick = self.embeddings["champions"](pred[:, -1:])
            input_picks = torch.cat([input_picks, pick], dim=1)

        draft = torch.stack(draft, dim=-1)
        return draft

    def model(self, x, x_cond):
        T, Tc = x.shape[1], x_cond.shape[1]

        mask = torch.ones(1, 1, T, Tc).tril().to(x.device)

        x = self.in_layer(x)
        x = self.positional_encoding(x)
        for layer in self.draft_decoder:
            x = layer(x, x_cond=x_cond, mask=mask)
        logits = self.out_layer(x)

        return logits

    def predict_from_logits(self, logits, bans=None):
        # logits (B, T, Nchamp)
        # bans (B, Nbans)
        if bans is not None:
            ban_mask = self.build_ban_mask(bans=bans)
            logits = logits.masked_fill(ban_mask, float("-inf"))

        probs = logits.softmax(dim=-1)
        return probs, probs.argmax(dim=-1)

    def build_ban_mask(self, bans):
        B = bans.shape[0]
        ban_mask = torch.zeros((B, self.embeddings["champions"].weight.shape[0])).bool()
        ban_mask = ban_mask.to(bans.device)
        ban_mask = torch.scatter(ban_mask, -1, bans.view(B, -1), True)
        return ban_mask.unsqueeze(1)


if __name__ == "__main__":
    drafter = Drafter(8, {"champions": 5})

    x = torch.arange(5)[None]
    pred, logprob = drafter(x, torch.randperm(5)[None])
    print(pred)
    print(logprob)
