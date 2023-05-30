import math, torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads

        self.qkv = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_cond=None, mask=None):
        if x_cond is None:
            x_cond = x

        b, t, d = x.size()
        bc, tc, dc = x_cond.size()

        q = self.qkv[0](x)
        k = self.qkv[1](x_cond)
        v = self.qkv[2](x_cond)

        q = q.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
        k = k.view(bc, tc, self.n_heads, dc // self.n_heads).transpose(1, 2)
        v = v.view(bc, tc, self.n_heads, dc // self.n_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            attn = attn.masked_fill(mask[:, :, :t, :tc] == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.dropout(out)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        cross_attn: bool = False,
    ):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads=n_heads)
        self.attn_norm = nn.LayerNorm(d_model)

        self.cross_attn = cross_attn
        if cross_attn:
            self.cross_attn = SelfAttention(d_model=d_model, n_heads=n_heads)
            self.cross_attn_norm = nn.LayerNorm(d_model)

        self.linear = nn.Linear(d_model, d_model)
        self.linear_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_cond=None, mask=None):
        if x_cond is not None:
            assert self.cross_attn

        x = x + self.dropout(self.attn(x, mask=mask))
        x = self.attn_norm(x)

        if self.cross_attn:
            assert x_cond is not None
            x = x + self.dropout(self.cross_attn(x, x_cond=x_cond, mask=mask))

        x = x + self.dropout(self.attn(x, x_cond=x_cond, mask=mask))
        x = x + self.dropout(self.linear(x))
        return self.linear_norm(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        n_layers: int = 3,
        max_len: int = 5000,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        n_layers: int = 3,
        max_len: int = 5000,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model, n_heads=n_heads, dropout=dropout, cross_attn=True
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, x_cond, mask=None):
        for layer in self.layers:
            x = layer(x, x_cond=x_cond, mask=mask)
        return x


if __name__ == "__main__":
    sa = TransformerDecoder(8, 2, max_len=5)

    x = torch.randn(3, 5, 8)
    sa(x, x_cond=x)
