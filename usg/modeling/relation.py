# usg/modeling/relation.py
"""
Relation Proposal Constructor (RPC) and Relation Decoder for USG-Par.

RPC:
1. Project queries: S = MLP_s(q), O = MLP_o(q)
2. Two-way RAC: S, O = RAC(S, O)
3. Pair confidence: C[i,j] = cosine_similarity(S[i], O[j])
4. Select Top-K pairs

Relation Decoder:
- Transformer with cross-attention to pixel features
- Predicate classification via dot-product with text embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP with ReLU activation."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(d_in if i == 0 else d_hidden, d_hidden),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Linear(d_hidden if num_layers > 1 else d_in, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RelationAwareCrossAttention(nn.Module):
    """
    Two-way Relation-Aware Cross-Attention (RAC).
    S = S + CrossAttn(S, O)
    O = O + CrossAttn(O, S)
    """
    def __init__(self, d: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "sub2obj": nn.MultiheadAttention(d, nhead, batch_first=True),
                "obj2sub": nn.MultiheadAttention(d, nhead, batch_first=True),
                "ln_s": nn.LayerNorm(d),
                "ln_o": nn.LayerNorm(d),
            })
            for _ in range(num_layers)
        ])

    def forward(self, sub: torch.Tensor, obj: torch.Tensor):
        """
        Args:
            sub: (B, N, D) subject features
            obj: (B, N, D) object features
        Returns:
            sub, obj: refined features
        """
        for layer in self.layers:
            s2o, _ = layer["sub2obj"](sub, obj, obj, need_weights=False)
            o2s, _ = layer["obj2sub"](obj, sub, sub, need_weights=False)
            sub = layer["ln_s"](sub + s2o)
            obj = layer["ln_o"](obj + o2s)
        return sub, obj


class RelationProposalConstructor(nn.Module):
    """
    Relation Proposal Constructor (RPC).

    Steps:
    1. Project queries: S = MLP_s(q), O = MLP_o(q)
    2. Two-way RAC: S, O = RAC(S, O)
    3. Pair confidence: C[i,j] = cosine_similarity(S[i], O[j])
    4. Select Top-K pairs

    Returns:
        sub_idx, obj_idx: (B, K) selected pair indices
        pair_scores: (B, K) confidence scores for selected pairs
        pair_conf: (B, Q, Q) full confidence matrix (for L_pair)
    """
    def __init__(self, d: int = 256, nhead: int = 8, rac_layers: int = 2, topk: int = 200):
        super().__init__()
        self.sub_proj = MLP(d, d, d, num_layers=2)
        self.obj_proj = MLP(d, d, d, num_layers=2)
        self.rac = RelationAwareCrossAttention(d=d, nhead=nhead, num_layers=rac_layers)
        self.topk = topk

    def forward(self, q: torch.Tensor):
        """
        Args:
            q: (B, Q, D) query features

        Returns:
            sub_idx: (B, K) subject indices
            obj_idx: (B, K) object indices
            pair_scores: (B, K) pair confidence scores
            pair_conf: (B, Q, Q) full confidence matrix (logits, for L_pair)
        """
        # 1. Project
        sub = self.sub_proj(q)  # (B, Q, D)
        obj = self.obj_proj(q)  # (B, Q, D)

        # 2. RAC
        sub, obj = self.rac(sub, obj)

        # 3. Pair confidence matrix (cosine similarity)
        sub_n = F.normalize(sub, dim=-1)
        obj_n = F.normalize(obj, dim=-1)
        C = torch.einsum("bnd,bmd->bnm", sub_n, obj_n)  # (B, Q, Q)

        B, Q, _ = C.shape

        # Exclude self-edges (use -1e4 for float16 compatibility)
        diag_mask = torch.eye(Q, device=C.device, dtype=torch.bool).unsqueeze(0)
        C_masked = C.masked_fill(diag_mask, -1e4)

        # 4. Top-K pairs
        flat = C_masked.view(B, -1)
        k = min(self.topk, flat.shape[1])
        vals, idx = torch.topk(flat, k=k, dim=-1)

        sub_idx = idx // Q
        obj_idx = idx % Q

        return sub_idx, obj_idx, vals, C


class RelationQueryBuilder(nn.Module):
    """
    Build relation queries from selected pairs.

    r_ij = MLP(concat(q_i, q_j))
    """
    def __init__(self, d: int = 256):
        super().__init__()
        self.fusion = MLP(d * 2, d, d, num_layers=2)

    def forward(self, q: torch.Tensor, sub_idx: torch.Tensor, obj_idx: torch.Tensor):
        """
        Args:
            q: (B, Q, D) query features
            sub_idx: (B, K) subject indices
            obj_idx: (B, K) object indices

        Returns:
            rel_q: (B, K, D) relation queries
        """
        B, K = sub_idx.shape
        D = q.size(-1)

        # Gather subject and object features
        q_sub = torch.gather(q, 1, sub_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, K, D)
        q_obj = torch.gather(q, 1, obj_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, K, D)

        # Concatenate and fuse
        rel_q = self.fusion(torch.cat([q_sub, q_obj], dim=-1))  # (B, K, D)

        return rel_q


class RelationDecoder(nn.Module):
    """
    Transformer-based Relation Decoder.

    - Self-attention on relation queries
    - Cross-attention to pixel decoder features
    - Predicate classification via linear layer (or dot-product with text embeddings)
    """
    def __init__(self, d: int = 256, nhead: int = 8, num_layers: int = 6, num_predicates: int = 56):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(d, nhead, batch_first=True),
                "cross_attn": nn.MultiheadAttention(d, nhead, batch_first=True),
                "ln1": nn.LayerNorm(d),
                "ln2": nn.LayerNorm(d),
                "ffn": nn.Sequential(
                    nn.Linear(d, 4 * d),
                    nn.ReLU(inplace=True),
                    nn.Linear(4 * d, d),
                ),
                "ln3": nn.LayerNorm(d),
            })
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d, num_predicates)

    def forward(self, rel_q: torch.Tensor, context: torch.Tensor):
        """
        Args:
            rel_q: (B, K, D) relation queries
            context: (B, HW, D) pixel decoder features

        Returns:
            logits: (B, K, P) predicate logits
        """
        x = rel_q
        for layer in self.layers:
            # Self-attention
            sa, _ = layer["self_attn"](x, x, x, need_weights=False)
            x = layer["ln1"](x + sa)

            # Cross-attention to context
            ca, _ = layer["cross_attn"](x, context, context, need_weights=False)
            x = layer["ln2"](x + ca)

            # FFN
            x = layer["ln3"](x + layer["ffn"](x))

        logits = self.classifier(x)  # (B, K, P)
        return logits
