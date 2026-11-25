import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GRSN(nn.Module):
    def __init__(self, network, normal_x, config):
        super().__init__()
        endpoints = torch.tensor(network.endpoints.copy(), dtype=torch.long)
        self.register_buffer("endpoints", endpoints, persistent=False)
        self.register_buffer("edge_index", torch.cat((endpoints, endpoints.flip(0)), dim=1), persistent=False)
        self.register_buffer("length_m", torch.tensor(network.length_m.copy(), dtype=torch.float32), persistent=False)
        self.register_buffer("importance", torch.tensor(network.importance.copy(), dtype=torch.float32), persistent=False)
        self.register_buffer("normal_x", torch.tensor(normal_x.copy(), dtype=torch.float32), persistent=False)
        channels = (1, *config.gcn_widths)
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(channels[:-1], channels[1:]))
        width = 3 * channels[-1] + 7
        layers = []
        for hidden in config.mlp_widths:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.scorer = nn.Sequential(*layers)

    def candidate_scores(self, x, damage, mask):
        if x.ndim != 2 or x.shape[1] != self.normal_x.numel():
            raise ValueError("x must have shape (batch_size, node_count)")
        if damage.shape != mask.shape or mask.shape != (x.shape[0], self.length_m.numel()):
            raise ValueError("damage and mask must have shape (batch_size, road_count)")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean")
        h = x.unsqueeze(-1)
        for conv in self.convs:
            h = torch.tanh(conv(h, self.edge_index))
        batches, roads = mask.nonzero(as_tuple=True)
        u, v = self.endpoints[:, roads]
        hu, hv = h[batches, u], h[batches, v]
        xu, xv = x[batches, u], x[batches, v]
        du, dv = self.normal_x[u] - xu, self.normal_x[v] - xv
        attributes = torch.stack((self.length_m[roads], damage[batches, roads], xu + xv,
                                  (xu - xv).abs(), du + dv, (du - dv).abs(),
                                  self.importance[roads]), dim=1)
        z = torch.cat((hu + hv, (hu - hv).abs(), h.mean(dim=1)[batches], attributes), dim=1)
        return batches, roads, self.scorer(z).squeeze(-1)

    def forward(self, x, damage, mask):
        batches, roads, scores = self.candidate_scores(x, damage, mask)
        return x.new_full(mask.shape, -torch.inf).index_put((batches, roads), scores)
