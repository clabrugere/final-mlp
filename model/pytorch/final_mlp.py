import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, num_hidden, dim_hidden, dim_out=None, dropout=0.0, batch_norm=True):
        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(dim_in, dim_hidden))

            if batch_norm:
                self.layers.append(nn.BatchNorm1d(dim_hidden))

            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        if dim_out:
            self.layers.append(nn.Linear(dim_in, dim_out))
        else:
            self.layers.append(nn.Linear(dim_in, dim_hidden))

    def forward(self, inputs):
        return self.layers(inputs)


class FeatureSelection(nn.Module):
    def __init__(self, dim_feature, dim_gate, num_hidden=1, dim_hidden=64, dropout=0.0):
        super().__init__()

        self.gate_1 = MLP(
            dim_in=dim_gate,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_feature,
            dropout=dropout,
            batch_norm=False,
        )
        self.gate_1_bias = nn.Parameter(torch.ones(1, dim_gate))

        self.gate_2 = MLP(
            dim_in=dim_gate,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_feature,
            dropout=dropout,
            batch_norm=False,
        )
        self.gate_2_bias = nn.Parameter(torch.ones(1, dim_gate))

    def forward(self, embeddings):
        print(embeddings.size())
        print(self.gate_1_bias.size())

        # embeddings is of shape (bs, dim_feature)
        gate_score_1 = self.gate_1(self.gate_1_bias)  # (1, dim_feature)
        print(gate_score_1.size())
        out_1 = 2.0 * F.sigmoid(gate_score_1) * embeddings  # (bs, dim_feature)

        gate_score_2 = self.gate_2(self.gate_2_bias)  # (1, dim_feature)
        out_2 = 2.0 * F.sigmoid(gate_score_2) * embeddings  # (bs, dim_feature)

        return out_1, out_2  # (bs, dim_feature), (bs, dim_feature)


class Aggregation(nn.Module):
    def __init__(self, dim_in_1, dim_in_2, dim_latent_1, dim_latent_2, num_heads=1):
        super().__init__()

        self.num_heads = num_heads
        self.dim_head_1 = dim_latent_1 // num_heads
        self.dim_head_2 = dim_latent_2 // num_heads

        self.w_1 = nn.Linear(dim_in_1, 1)
        self.w_2 = nn.Linear(dim_in_2, 1)

        self.w_1 = nn.Parameter(torch.empty(self.dim_head_1, num_heads, 1))
        self.w_2 = nn.Parameter(torch.empty(self.dim_head_2, num_heads, 1))
        self.w_12 = nn.Parameter(torch.empty(num_heads, self.dim_head_1, self.dim_head_2, 1))
        self.bias = nn.Parameter(torch.ones(1, num_heads, 1))

        self._reset_weights()

    def _reset_weights(self):
        nn.init.xavier_uniform_(self.w_1)
        nn.init.xavier_uniform_(self.w_2)
        nn.init.xavier_uniform_(self.w_12)

    def forward(self, latent_1, latent_2):
        # bilinear aggregation of the two latent representations
        # y = b + w_1.T o_1 + w_2.T o_2 + o_1.T W_3 o_2
        latent_1 = torch.reshape(latent_1, (-1, self.num_heads, self.dim_head_1))  # (bs, num_heads, dim_head_1)
        latent_2 = torch.reshape(latent_2, (-1, self.num_heads, self.dim_head_2))  # (bs, num_heads, dim_head_2)

        first_order = torch.einsum("bhi,iho->bho", latent_1, self.w_1)  # (bs, num_heads, 1)
        first_order += torch.einsum("bhi,iho->bho", latent_2, self.w_2)  # (bs, num_heads, 1)
        second_order = torch.einsum("bhi,hijo,bhj->bho", latent_1, self.w_12, latent_2)  # (bs, num_heads, 1)

        out = torch.sum(first_order + second_order + self.bias, 1)  # (bs, 1)

        return out


class FinalMLP(nn.Module):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=32,
        dim_hidden_fs=64,
        num_hidden_1=2,
        dim_hidden_1=64,
        num_hidden_2=2,
        dim_hidden_2=64,
        num_heads=1,
        dropout=0.0,
    ):
        super().__init__()

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=dim_embedding)

        # feature selection layer that projects a learnable vector to the flatened embedded feature space
        self.feature_selection = FeatureSelection(
            dim_feature=dim_input * dim_embedding,
            dim_gate=dim_input,
            dim_hidden=dim_hidden_fs,
            dropout=dropout,
        )

        # branch 1
        self.interaction_1 = MLP(
            dim_in=dim_input * dim_embedding,
            num_hidden=num_hidden_1,
            dim_hidden=dim_hidden_1,
            dropout=dropout,
        )
        # branch 2
        self.interaction_2 = MLP(
            dim_in=dim_input * dim_embedding,
            num_hidden=num_hidden_2,
            dim_hidden=dim_hidden_2,
            dropout=dropout,
        )

        # final aggregation layer
        self.aggregation = Aggregation(
            dim_in_1=dim_hidden_1,
            dim_in_2=dim_hidden_2,
            dim_latent_1=dim_hidden_1,
            dim_latent_2=dim_hidden_2,
            num_heads=num_heads,
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # (bs, num_emb, dim_emb)
        embeddings = torch.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)

        # weight features of the two streams using a gating mechanism
        emb_1, emb_2 = self.feature_selection(embeddings)  # (bs, num_emb * dim_emb), (bs, num_emb * dim_emb)

        # get interactions from the two branches
        # (bs, dim_hidden_1), (bs, dim_hidden_1)
        latent_1, latent_2 = self.interaction_1(emb_1), self.interaction_2(emb_2)

        # merge the representations using an aggregation scheme
        logits = self.aggregation(latent_1, latent_2)  # (bs, 1)
        outputs = F.sigmoid(logits)  # (bs, 1)

        return outputs
