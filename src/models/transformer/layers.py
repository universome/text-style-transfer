import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings in log space.
        pe = torch.zeros(config.max_len, config.d_model)

        ln_enumerator = torch.log(torch.arange(0., config.max_len))
        ln_denumerator = torch.arange(0., config.d_model, 2) * (4 * np.log(10) / config.d_model)
        inner_term = torch.exp(ln_enumerator.unsqueeze(1) - ln_denumerator.unsqueeze(0))

        pe[:, 0::2] = torch.sin(inner_term)
        pe[:, 1::2] = torch.cos(inner_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        assert x.dim() == 3 # batch_size * seq_len * emb_size

        return x + self.pe[:x.size(1)].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.config = config

        assert config.d_model % config.n_heads == 0
        self.d_k = self.config.d_model // self.config.n_heads

        # We can run all attentions with a single linear map
        self.attention = ScaledDotProductAttention(config)
        self.dropout = nn.Dropout(p=self.config.dropout)

        self.query_map = nn.Linear(config.d_model, config.d_model)
        self.key_map = nn.Linear(config.d_model, config.d_model)
        self.value_map = nn.Linear(config.d_model, config.d_model)
        self.output_map = nn.Linear(config.d_model, config.d_model)

    def forward(self, query, key, value, mask=None):
        assert query.size(2) == key.size(2) == value.size(2) == self.config.d_model

        if mask is not None:
            assert mask.dim() == 3, "Assuming mask to have 3 dimensions, but got {}".format(mask.dim())
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        head_size = self.config.d_model // self.config.n_heads

        # 1) Do all the linear projections in batch from d_model => n_heads x d_k
        # TODO: Do we have to ".view(...).transpose(..)" it?
        queries = self.query_map(query).view(batch_size, -1, self.config.n_heads, head_size).transpose(1, 2)
        keys = self.key_map(key).view(batch_size, -1, self.config.n_heads, head_size).transpose(1, 2)
        values = self.value_map(value).view(batch_size, -1, self.config.n_heads, head_size).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        out = self.attention(queries, keys, values, mask, self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_heads * head_size)
        out = self.output_map(out)

        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()

        self.config = config
        self.softmax = nn.Softmax(dim=3)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask, dropout):
        assert q.dim() == k.dim() == v.dim() == 4

        head_size = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_size)
        scores.masked_fill_(mask == 0, -1e9)

        weights = self.softmax(scores)
        weights = self.dropout(weights)
        values = torch.matmul(weights, v)

        return values


class SublayerConnection(nn.Module):
    def __init__(self, config):
        super(SublayerConnection, self).__init__()

        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(config.d_model, config.d_ff)
        self.w_2 = nn.Linear(config.d_ff, config.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.relu(self.w_1(x))
        x = self.w_2(self.dropout(x))

        return x
