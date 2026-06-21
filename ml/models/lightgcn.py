"""
LightGCN model for user-food collaborative filtering.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]

        num_nodes = self.num_users + self.num_items
        adj = self._build_sparse_adj(edge_index, num_nodes)
        adj_norm = self._normalize_adj(adj, num_nodes)

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_norm, all_emb)
            embs.append(all_emb)

        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        users, items = final_emb.split([self.num_users, self.num_items], dim=0)
        return users, items

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.forward(edge_index)
        u = user_emb[user_ids]
        i = item_emb[item_ids]
        return (u * i).sum(dim=-1)

    def bpr_loss(self, user_ids, pos_ids, neg_ids, edge_index):
        user_emb, item_emb = self.forward(edge_index)

        u = user_emb[user_ids]
        pos = item_emb[pos_ids]
        neg = item_emb[neg_ids]

        pos_score = (u * pos).sum(dim=-1)
        neg_score = (u * neg).sum(dim=-1)

        loss = -F.logsigmoid(pos_score - neg_score).mean()

        reg = (u ** 2).sum() + (pos ** 2).sum() + (neg ** 2).sum()
        reg = reg / (3 * len(user_ids))

        return loss + 1e-4 * reg

    def _build_sparse_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.Tensor:
        user_idx = edge_index[0]
        item_idx = edge_index[1] + self.num_users

        rows = torch.cat([user_idx, item_idx])
        cols = torch.cat([item_idx, user_idx])
        indices = torch.stack([rows, cols])

        values = torch.ones(indices.shape[1], device=edge_index.device)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        return adj

    def _normalize_adj(self, adj: torch.sparse.Tensor, num_nodes: int) -> torch.sparse.Tensor:
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        deg_diag = torch.sparse_coo_tensor(
            torch.arange(num_nodes).unsqueeze(0).repeat(2, 1),
            deg_inv_sqrt,
            (num_nodes, num_nodes),
        )
        return torch.sparse.mm(torch.sparse.mm(deg_diag, adj), deg_diag)
