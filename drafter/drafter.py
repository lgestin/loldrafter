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

        self.positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=3 * 5
        )

        self.in_layer = nn.Linear(d_model, d_model)

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

    def forward(self, picks, bans, picks_mask=None, bans_mask=None):
        # conditioning
        # - Bans
        # - Already picked champs (allied / ennemy)
        # - Lane of the champ to pick
        # - Side, Patch, League, Team

        # input
        # picks [12, 35, 18, 121, 5] champions picked
        B, _, T = picks.shape
        picks = picks.masked_fill(~picks_mask, 0)
        picks = self.embeddings["champions"](picks)

        # Encode bans
        # bans (B, 2, T) 2 is because both team ban
        # TODO: Check if views do as expected
        bans = bans.masked_fill(~bans_mask, 0)
        bans = self.embeddings["champions"](bans)
        # x_cond is bans and enemy_picks
        x_cond = torch.cat([bans, picks[:, 1:]], dim=1)
        x_cond = self.positional_encoding(x_cond.view(B, 3 * T, -1))
        x_cond = self.bans_encoder(x_cond)

        # Encode picks
        # picks (B, 2, T)
        x = self.positional_encoding(picks[:, 0])

        logits = self.model(x, x_cond)

        return logits

    @torch.no_grad()
    def sample_next(self, picks, bans, picks_mask=None, bans_mask=None):
        # Sample the next n most probable picks given current ones
        B, _, T = bans.shape

        if picks_mask is not None:
            picks = picks.masked_fill(~picks_mask, 0)
        x = self.embeddings["champions"](picks)

        if bans_mask is not None:
            bans = bans.masked_fill(~bans_mask, 0)
        x_cond = self.embeddings["champions"](bans)
        x_cond = torch.cat([x_cond, x[:, 1:]], dim=1)
        x_cond = self.positional_encoding(x_cond.view(B, 3 * T, -1))
        x_cond = self.bans_encoder(x_cond)

        x = self.positional_encoding(x[:, 0])

        logits = self.model(x, x_cond=x_cond)

        already_picks = picks.masked_fill(~picks_mask, 0)
        probs, pred = self.predict_from_logits(
            logits=logits, already_picks=already_picks, bans=bans
        )

        # TODO: mask out probs from already known picks and bans
        max_probs = probs.max(dim=-1).values
        max_probs = max_probs.masked_fill(picks_mask[:, 0], 0)  # Don't predict alreadyknown
        pick_mask = max_probs == max_probs.max(dim=-1, keepdims=True).values
        pick = pred[pick_mask]

        updated_picks = picks[:, 0] * ~pick_mask + pred * pick_mask

        return pick, picks, updated_picks

    def model(self, x, x_cond):
        T, Tc = x.shape[1], x_cond.shape[1]

        x = self.in_layer(x)
        x = self.positional_encoding(x)
        for layer in self.draft_decoder:
            x = layer(x, x_cond=x_cond)
        logits = self.out_layer(x)

        return logits

    def predict_from_logits(self, logits, already_picks=None, bans=None):
        # logits (B, T, Nchamp)
        # bans (B, Nbans)

        if already_picks is None:
            mask = torch.zeros_like(logits[..., 0])
        else:
            mask = self.build_ban_mask(already_picks)
        logits = logits.masked_fill(mask, float("-inf"))

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
