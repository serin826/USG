# usg/modeling/usgpar_image.py
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from .backbone_openclip import OpenCLIPConvNeXtBackbone
from .m2f_adapter import Mask2FormerAdapter
from .relation import RelationProposalConstructor, RelationQueryBuilder, RelationDecoder


class USGParImageOnly(nn.Module):
    """
    이미지 전용 USG-Par 흐름(논문 구조에 맞춘 연결):
      - Frozen OpenCLIP ConvNeXt backbone
      - Mask2Former pixel+mask decoder: masks + query_features 추출
      - Object logits: query · text_emb (open-vocab)
      - Relation: RPC(top-k pair) + relation decoder -> predicate logits
    """
    def __init__(
        self,
        obj_class_names: List[str],
        pred_class_names: List[str],
        clip_model: str = "convnext_base",
        clip_pretrained: str = "laion400m_s13b_b51k",
        freeze_backbone: bool = True,
        num_queries: int = 100,
        topk_pairs: int = 200,
        rel_layers: int = 6,
        d_model: int = 256,
    ):
        super().__init__()
        self.obj_names = obj_class_names
        self.pred_names = pred_class_names

        self.backbone = OpenCLIPConvNeXtBackbone(
            model_name=clip_model,
            pretrained=clip_pretrained,
            freeze=freeze_backbone,
        )

        in_channels = self.backbone.stage_channels
        strides = self.backbone.strides

        print("[USG] stage_channels:", in_channels)
        print("[USG] strides:", strides)

        self.m2f = Mask2FormerAdapter(
            in_channels=in_channels,
            strides=strides,
            conv_dim=d_model,
            mask_dim=d_model,
            num_queries=num_queries,
            num_classes_closed=len(obj_class_names),
            dec_layers=9,
            nheads=8,
        )

        # open-vocab text embeddings
        self.register_buffer("obj_text_emb", torch.empty(0), persistent=False)
        self.register_buffer("pred_text_emb", torch.empty(0), persistent=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # projection: query features (d_model) -> text embedding space (clip_dim)
        # OpenCLIP convnext_base text embedding is 512-dim
        self._clip_text_dim = 512
        self.query_to_text_proj = nn.Linear(d_model, self._clip_text_dim)

        # relation modules (논문: RPC + relation query builder + relation decoder)
        self.rpc = RelationProposalConstructor(d=d_model, nhead=8, rac_layers=2, topk=topk_pairs)
        self.rel_query_builder = RelationQueryBuilder(d=d_model)
        self.rel_decoder = RelationDecoder(d=d_model, nhead=8, num_layers=rel_layers, num_predicates=len(pred_class_names))

        # text encoder(임베딩 생성용) – inference 시에도 필요
        self._text_model_name = clip_model
        self._text_pretrained = clip_pretrained

    @torch.no_grad()
    def build_text_embeddings(self, device: Optional[torch.device] = None):
        """
        논문처럼: class-name text embedding을 사용.
        logits = inner_product(query, text_emb)
        """
        device = device or torch.device("cpu")
        model, _, _ = open_clip.create_model_and_transforms(self._text_model_name, pretrained=self._text_pretrained)
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(self._text_model_name)

        def encode(names: List[str]) -> torch.Tensor:
            # 프롬프트는 간단히 "a photo of {name}" 사용(원하면 개선 가능)
            texts = [f"a photo of {n}" for n in names]
            tokens = tokenizer(texts).to(device)
            feat = model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
            return feat

        self.obj_text_emb = encode(self.obj_names)
        self.pred_text_emb = encode(self.pred_names)

    def forward(
        self,
        images: torch.Tensor,
        gt_pairs: Optional[List[torch.Tensor]] = None,
        gt_indices: Optional[List[tuple]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) input images
            gt_pairs: (training only) list of (R, 2) GT relation pairs per image
            gt_indices: (training only) list of (pred_idx, gt_idx) from Hungarian matching

        Returns:
            - mask_logits: (B,Q,Hm,Wm)
            - obj_logits_open: (B,Q,Cobj)
            - query_features: (B,Q,D)
            - pair_sub_idx/pair_obj_idx: (B,K)
            - rel_logits: (B,K,Cpred)
            - gt_rel_logits: (training only) list of (Ri, Cpred) for GT pairs
        """
        if self.obj_text_emb.numel() == 0:
            raise RuntimeError("build_text_embeddings(device)를 먼저 호출해야 합니다.")

        feats = self.backbone(images)  # list stage feats
        m2f_out = self.m2f(feats)

        q = m2f_out.query_features  # (B,Q,D)

        # project query features to text embedding space for open-vocab classification
        q_proj = self.query_to_text_proj(q)  # (B,Q,clip_dim)
        qn = F.normalize(q_proj, dim=-1)
        scale = self.logit_scale.exp().clamp(1e-3, 100.0)

        # object open-vocab logits
        obj_logits_open = scale * torch.einsum("bqd,cd->bqc", qn, self.obj_text_emb)

        # relation: RPC -> RelationQueryBuilder -> RelationDecoder
        sub_idx, obj_idx, pair_scores, pair_conf = self.rpc(q)  # (B,K), (B,K), (B,K), (B,Q,Q)

        # Build relation queries: r_ij = MLP(concat(q_i, q_j))
        rel_q = self.rel_query_builder(q, sub_idx, obj_idx)  # (B,K,D)

        # context tokens from mask2former features
        ctx = m2f_out.context_tokens  # (B,HW,D)
        rel_logits = self.rel_decoder(rel_q, ctx)  # (B,K,Cpred)

        result = {
            "mask_logits": m2f_out.pred_masks,
            "obj_logits_open": obj_logits_open,
            "query_features": q,
            "context_tokens": ctx,  # (B,HW,D) for relation decoder
            "pair_sub_idx": sub_idx,
            "pair_obj_idx": obj_idx,
            "pair_scores": pair_scores,
            "pair_conf": pair_conf,  # (B,Q,Q) for L_pair
            "rel_logits": rel_logits,
        }

        # Training: compute relation logits for GT pairs directly
        if gt_pairs is not None and gt_indices is not None and self.training:
            gt_rel_logits = []
            B = images.shape[0]

            for b in range(B):
                if len(gt_pairs[b]) == 0 or len(gt_indices[b][0]) == 0:
                    gt_rel_logits.append(None)
                    continue

                pred_idx, gt_idx = gt_indices[b]

                # Map GT object indices to matched query indices
                gt_to_query = {}
                for i in range(len(gt_idx)):
                    gt_to_query[gt_idx[i].item()] = pred_idx[i].item()

                # Collect valid GT pairs (both endpoints matched)
                valid_sub_q = []
                valid_obj_q = []
                valid_rel_idx = []

                for r in range(len(gt_pairs[b])):
                    sub_gt = gt_pairs[b][r, 0].item()
                    obj_gt = gt_pairs[b][r, 1].item()

                    if sub_gt in gt_to_query and obj_gt in gt_to_query:
                        valid_sub_q.append(gt_to_query[sub_gt])
                        valid_obj_q.append(gt_to_query[obj_gt])
                        valid_rel_idx.append(r)

                if len(valid_sub_q) == 0:
                    gt_rel_logits.append(None)
                    continue

                # Build relation queries for GT pairs
                sub_q_tensor = torch.tensor(valid_sub_q, device=q.device)
                obj_q_tensor = torch.tensor(valid_obj_q, device=q.device)

                gt_rel_q = self.rel_query_builder(
                    q[b:b+1],
                    sub_q_tensor.unsqueeze(0),
                    obj_q_tensor.unsqueeze(0)
                )  # (1, R', D)

                # Decode GT relation queries
                gt_logits = self.rel_decoder(gt_rel_q, ctx[b:b+1])  # (1, R', P)
                gt_rel_logits.append({
                    'logits': gt_logits[0],  # (R', P)
                    'valid_idx': valid_rel_idx,  # which GT relations are valid
                })

            result["gt_rel_logits"] = gt_rel_logits

        return result
