# usg/modeling/losses.py
"""
USG-Par Losses for PSG SGDet
- L_obj = L_cls + L_mask_bce + L_mask_dice
- L_rel = L_predicate_cls + L_pair
- L_total = L_obj + L_rel
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss for binary masks.
    pred: (N, H, W) logits
    target: (N, H, W) binary masks
    """
    pred = pred.sigmoid().flatten(1)  # (N, H*W)
    target = target.flatten(1).float()  # (N, H*W)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def sigmoid_ce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid cross-entropy loss for masks.
    pred: (N, H, W) logits
    target: (N, H, W) binary masks
    """
    return F.binary_cross_entropy_with_logits(pred, target.float(), reduction='mean')


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for query-GT assignment.

    Matching cost = 1.0 * cls_cost + 5.0 * bce_cost + 5.0 * dice_cost
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask_bce: float = 5.0,
        cost_mask_dice: float = 5.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask_bce = cost_mask_bce
        self.cost_mask_dice = cost_mask_dice

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, Q, C)
        pred_masks: torch.Tensor,   # (B, Q, H, W)
        gt_classes: List[torch.Tensor],  # list of (Ni,)
        gt_masks: List[torch.Tensor],    # list of (Ni, H, W)
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns list of (pred_indices, gt_indices) for each batch element.
        """
        B, Q, H, W = pred_masks.shape
        device = pred_masks.device
        indices = []

        for b in range(B):
            tgt_cls = gt_classes[b]  # (N,)
            tgt_mask = gt_masks[b]   # (N, Hgt, Wgt)
            N = tgt_cls.shape[0]

            if N == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device)
                ))
                continue

            # Move to device and resize if needed
            tgt_mask = tgt_mask.to(device).float()
            if tgt_mask.shape[1:] != (H, W):
                tgt_mask = F.interpolate(
                    tgt_mask.unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )[0]

            # Cast to float32 for numerical stability (important for AMP)
            out_logits = pred_logits[b].float()  # (Q, C)
            out_mask = pred_masks[b].float()  # (Q, H, W)

            # Classification cost: -prob[target_class]
            out_prob = out_logits.sigmoid()  # (Q, C)
            tgt_cls_dev = tgt_cls.to(device)
            cost_class = -out_prob[:, tgt_cls_dev]  # (Q, N)

            # Mask costs
            out_mask_flat = out_mask.flatten(1)  # (Q, H*W)
            tgt_mask_flat = tgt_mask.flatten(1)  # (N, H*W)

            # BCE cost (clamp for numerical stability)
            out_mask_sig = out_mask_flat.sigmoid().clamp(1e-6, 1 - 1e-6)
            cost_bce = -(tgt_mask_flat.unsqueeze(0) * torch.log(out_mask_sig.unsqueeze(1)) +
                        (1 - tgt_mask_flat.unsqueeze(0)) * torch.log(1 - out_mask_sig.unsqueeze(1)))
            cost_bce = cost_bce.mean(dim=2)  # (Q, N)

            # Dice cost
            numerator = 2 * torch.einsum("qh,nh->qn", out_mask_sig, tgt_mask_flat)
            denominator = out_mask_sig.sum(1)[:, None] + tgt_mask_flat.sum(1)[None, :] + 1e-6
            cost_dice = 1 - numerator / denominator  # (Q, N)

            # Total cost
            C = (
                self.cost_class * cost_class +
                self.cost_mask_bce * cost_bce +
                self.cost_mask_dice * cost_dice
            )

            # Check for NaN/Inf and replace with large value
            C = torch.where(torch.isfinite(C), C, torch.full_like(C, 1e4))
            C = C.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(C)
            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=device),
                torch.tensor(gt_idx, dtype=torch.long, device=device)
            ))

        return indices


class USGLoss(nn.Module):
    """
    Combined loss for USG-Par model.

    L_obj = lambda_cls * L_cls + lambda_bce * L_bce + lambda_dice * L_dice
    L_rel = L_predicate_cls + L_pair
    L_total = L_obj + L_rel
    """

    def __init__(
        self,
        num_classes: int = 133,
        num_predicates: int = 56,
        lambda_cls: float = 1.0,
        lambda_bce: float = 5.0,
        lambda_dice: float = 5.0,
        pair_pos_weight: float = 5.0,
        no_object_weight: float = 0.1,
        rel_query_builder=None,
        rel_decoder=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.lambda_cls = lambda_cls
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.pair_pos_weight = pair_pos_weight

        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_mask_bce=5.0,
            cost_mask_dice=5.0,
        )

        # No-object weight for classification
        self.no_object_weight = no_object_weight

        # Relation modules for computing GT pair losses (set via set_relation_modules)
        self.rel_query_builder = rel_query_builder
        self.rel_decoder = rel_decoder

    def set_relation_modules(self, rel_query_builder, rel_decoder):
        """Set relation modules for computing GT pair losses."""
        self.rel_query_builder = rel_query_builder
        self.rel_decoder = rel_decoder

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_classes: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        rel_pairs: List[torch.Tensor],
        rel_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: model outputs containing
                - obj_logits: (B, Q, C) object classification logits
                - mask_logits: (B, Q, H, W) mask logits
                - rel_logits: (B, K, P) relation predicate logits
                - pair_conf: (B, Q, Q) pair confidence matrix from RPC
                - pair_sub_idx, pair_obj_idx: (B, K) selected pair indices
        """
        pred_logits = outputs['obj_logits_open']  # (B, Q, C)
        pred_masks = outputs['mask_logits']  # (B, Q, H, W)
        B, Q, H, W = pred_masks.shape
        device = pred_masks.device

        # Hungarian matching
        indices = self.matcher(pred_logits, pred_masks, gt_classes, gt_masks)

        # ============ Object Loss ============
        loss_cls = self._compute_cls_loss(pred_logits, gt_classes, indices)
        loss_bce, loss_dice = self._compute_mask_loss(pred_masks, gt_masks, indices, H, W)

        L_obj = (
            self.lambda_cls * loss_cls +
            self.lambda_bce * loss_bce +
            self.lambda_dice * loss_dice
        )

        # ============ Relation Loss ============
        loss_pred_cls, loss_pair = self._compute_relation_loss(
            outputs, indices, rel_pairs, rel_labels, gt_classes, device
        )

        L_rel = loss_pred_cls + loss_pair

        # Total loss
        total_loss = L_obj + L_rel

        return {
            'loss': total_loss,
            'L_obj': L_obj,
            'L_rel': L_rel,
            'loss_cls': loss_cls,
            'loss_bce': loss_bce,
            'loss_dice': loss_dice,
            'loss_pred_cls': loss_pred_cls,
            'loss_pair': loss_pair,
        }

    def _compute_cls_loss(
        self,
        pred_logits: torch.Tensor,
        gt_classes: List[torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute classification loss with sigmoid CE."""
        B, Q, C = pred_logits.shape
        device = pred_logits.device

        # Build target: one-hot for matched, all zeros for unmatched
        target = torch.zeros(B, Q, C, device=device)

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_idx_cpu = pred_idx.cpu()
                gt_idx_cpu = gt_idx.cpu()
                gt_cls = gt_classes[b][gt_idx_cpu].to(device)
                for i, (pi, ci) in enumerate(zip(pred_idx, gt_cls)):
                    if ci < C:
                        target[b, pi, ci] = 1.0

        # Sigmoid CE loss
        loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')
        return loss

    def _compute_mask_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: List[torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        H: int, W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mask BCE and Dice loss."""
        device = pred_masks.device
        total_bce = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)
        num_masks = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            pred_idx_cpu = pred_idx.cpu()
            gt_idx_cpu = gt_idx.cpu()

            src_masks = pred_masks[b, pred_idx_cpu]  # (M, H, W)
            tgt_masks = gt_masks[b][gt_idx_cpu].to(device).float()  # (M, Hgt, Wgt)

            if tgt_masks.shape[1:] != (H, W):
                tgt_masks = F.interpolate(
                    tgt_masks.unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )[0]

            total_bce = total_bce + sigmoid_ce_loss(src_masks, tgt_masks)
            total_dice = total_dice + dice_loss(src_masks, tgt_masks)
            num_masks += 1

        if num_masks > 0:
            total_bce = total_bce / num_masks
            total_dice = total_dice / num_masks

        return total_bce, total_dice

    def _compute_relation_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        rel_pairs: List[torch.Tensor],
        rel_labels: List[torch.Tensor],
        gt_classes: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relation loss = L_predicate_cls + L_pair
        """
        # ============ L_pair: Train RPC confidence matrix ============
        loss_pair = self._compute_pair_loss(outputs, indices, rel_pairs, device)

        # ============ L_predicate_cls: Train predicate classifier ============
        loss_pred_cls = self._compute_predicate_loss(outputs, indices, rel_pairs, rel_labels, device)

        return loss_pred_cls, loss_pair

    def _compute_pair_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        rel_pairs: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        L_pair: Weighted BCE on pair confidence matrix C.
        G[i,j] = 1 if (i,j) is a GT relation pair, else 0.
        """
        if 'pair_conf' not in outputs:
            return torch.tensor(0.0, device=device)

        pair_conf = outputs['pair_conf']  # (B, Q, Q)
        B, Q, _ = pair_conf.shape

        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0

        for b in range(B):
            pred_idx, gt_idx = indices[b]
            if len(pred_idx) == 0 or len(rel_pairs[b]) == 0:
                continue

            # Map GT object indices to matched query indices
            gt_to_query = {}
            for i in range(len(gt_idx)):
                gt_to_query[gt_idx[i].item()] = pred_idx[i].item()

            # Build target matrix G
            G = torch.zeros(Q, Q, device=device)
            gt_rel = rel_pairs[b]  # (R, 2)

            for r in range(len(gt_rel)):
                sub_gt, obj_gt = gt_rel[r, 0].item(), gt_rel[r, 1].item()
                if sub_gt in gt_to_query and obj_gt in gt_to_query:
                    sub_q = gt_to_query[sub_gt]
                    obj_q = gt_to_query[obj_gt]
                    G[sub_q, obj_q] = 1.0

            # Weighted BCE loss
            C = pair_conf[b]  # (Q, Q)
            pos_weight = torch.tensor([self.pair_pos_weight], device=device)
            loss = F.binary_cross_entropy_with_logits(
                C.view(-1), G.view(-1),
                pos_weight=pos_weight.expand_as(G.view(-1)),
                reduction='mean'
            )
            total_loss = total_loss + loss
            num_valid += 1

        if num_valid > 0:
            total_loss = total_loss / num_valid

        return total_loss

    def _compute_predicate_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        rel_pairs: List[torch.Tensor],
        rel_labels: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        L_predicate_cls: Sigmoid CE on predicate logits for GT pairs.

        Computes relation logits directly from GT pairs using relation modules.
        This ensures relation decoder gets trained regardless of RPC quality.
        """
        # Compute GT relation logits directly if relation modules are available
        if self.rel_query_builder is not None and self.rel_decoder is not None:
            return self._compute_predicate_loss_direct(outputs, indices, rel_pairs, rel_labels, device)

        # Use GT relation logits if available (computed in model forward)
        if 'gt_rel_logits' in outputs:
            return self._compute_predicate_loss_from_gt(outputs, rel_labels, device)

        # Fallback: try to use RPC-selected pairs (may fail if GT not in Top-K)
        return self._compute_predicate_loss_from_rpc(outputs, indices, rel_pairs, rel_labels, device)

    def _compute_predicate_loss_direct(
        self,
        outputs: Dict[str, torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        rel_pairs: List[torch.Tensor],
        rel_labels: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute predicate loss by directly processing GT pairs through relation modules."""
        query_features = outputs['query_features']  # (B, Q, D)
        # Get context tokens - may be flattened pixel features
        if 'context_tokens' in outputs:
            context = outputs['context_tokens']
        else:
            # Flatten mask features as context (simplified)
            mask_logits = outputs['mask_logits']  # (B, Q, H, W)
            B, Q, H, W = mask_logits.shape
            D = query_features.shape[-1]
            # Use query features as context (self-attention style)
            context = query_features  # (B, Q, D)

        P = self.num_predicates
        total_loss = torch.tensor(0.0, device=device)
        num_rels = 0

        B = query_features.shape[0]
        for b in range(B):
            pred_idx, gt_idx = indices[b]
            if len(pred_idx) == 0 or len(rel_pairs[b]) == 0:
                continue

            # Map GT object indices to matched query indices
            gt_to_query = {}
            for i in range(len(gt_idx)):
                gt_to_query[gt_idx[i].item()] = pred_idx[i].item()

            gt_rel = rel_pairs[b].to(device)
            gt_lbl = rel_labels[b].to(device)

            # Collect valid GT pairs
            valid_sub_q = []
            valid_obj_q = []
            valid_labels = []

            for r in range(len(gt_rel)):
                sub_gt, obj_gt = gt_rel[r, 0].item(), gt_rel[r, 1].item()
                pred_id = gt_lbl[r].item()

                if sub_gt in gt_to_query and obj_gt in gt_to_query:
                    valid_sub_q.append(gt_to_query[sub_gt])
                    valid_obj_q.append(gt_to_query[obj_gt])
                    valid_labels.append(pred_id)

            if len(valid_sub_q) == 0:
                continue

            # Build relation queries for GT pairs
            sub_q_tensor = torch.tensor(valid_sub_q, device=device, dtype=torch.long)
            obj_q_tensor = torch.tensor(valid_obj_q, device=device, dtype=torch.long)

            # Get query features for batch b
            q_b = query_features[b:b+1]  # (1, Q, D)
            ctx_b = context[b:b+1]  # (1, ?, D)

            # Build relation queries
            gt_rel_q = self.rel_query_builder(
                q_b,
                sub_q_tensor.unsqueeze(0),
                obj_q_tensor.unsqueeze(0)
            )  # (1, R', D)

            # Decode through relation decoder
            gt_logits = self.rel_decoder(gt_rel_q, ctx_b)  # (1, R', P)
            gt_logits = gt_logits[0]  # (R', P)

            # Compute loss for each GT relation
            for i, pred_id in enumerate(valid_labels):
                logit = gt_logits[i]  # (P,)

                target = torch.zeros(P, device=device)
                if pred_id < P:
                    target[pred_id] = 1.0
                loss = F.binary_cross_entropy_with_logits(logit, target, reduction='mean')
                total_loss = total_loss + loss
                num_rels += 1

        if num_rels > 0:
            total_loss = total_loss / num_rels

        return total_loss

    def _compute_predicate_loss_from_gt(
        self,
        outputs: Dict[str, torch.Tensor],
        rel_labels: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute predicate loss using GT relation logits."""
        gt_rel_logits = outputs['gt_rel_logits']
        P = self.num_predicates

        total_loss = torch.tensor(0.0, device=device)
        num_rels = 0

        for b, gt_data in enumerate(gt_rel_logits):
            if gt_data is None:
                continue

            logits = gt_data['logits']  # (R', P)
            valid_idx = gt_data['valid_idx']  # indices into rel_labels[b]
            gt_lbl = rel_labels[b].to(device)

            for i, r in enumerate(valid_idx):
                pred_id = gt_lbl[r].item()
                logit = logits[i]  # (P,)

                # Sigmoid CE (one-hot target)
                target = torch.zeros(P, device=device)
                if pred_id < P:
                    target[pred_id] = 1.0
                loss = F.binary_cross_entropy_with_logits(logit, target, reduction='mean')
                total_loss = total_loss + loss
                num_rels += 1

        if num_rels > 0:
            total_loss = total_loss / num_rels

        return total_loss

    def _compute_predicate_loss_from_rpc(
        self,
        outputs: Dict[str, torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        rel_pairs: List[torch.Tensor],
        rel_labels: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute predicate loss from RPC-selected pairs (fallback)."""
        if 'rel_logits' not in outputs:
            return torch.tensor(0.0, device=device)

        rel_logits = outputs['rel_logits']
        pair_sub_idx = outputs['pair_sub_idx']
        pair_obj_idx = outputs['pair_obj_idx']
        B, K, P = rel_logits.shape

        total_loss = torch.tensor(0.0, device=device)
        num_rels = 0

        for b in range(B):
            pred_idx, gt_idx = indices[b]
            if len(pred_idx) == 0 or len(rel_pairs[b]) == 0:
                continue

            gt_to_query = {}
            for i in range(len(gt_idx)):
                gt_to_query[gt_idx[i].item()] = pred_idx[i].item()

            pred_sub = pair_sub_idx[b]
            pred_obj = pair_obj_idx[b]
            gt_rel = rel_pairs[b].to(device)
            gt_lbl = rel_labels[b].to(device)

            for r in range(len(gt_rel)):
                sub_gt, obj_gt = gt_rel[r, 0].item(), gt_rel[r, 1].item()
                pred_id = gt_lbl[r].item()

                if sub_gt not in gt_to_query or obj_gt not in gt_to_query:
                    continue

                sub_q = gt_to_query[sub_gt]
                obj_q = gt_to_query[obj_gt]

                match_mask = (pred_sub == sub_q) & (pred_obj == obj_q)
                match_indices = match_mask.nonzero(as_tuple=True)[0]

                if len(match_indices) > 0:
                    k = match_indices[0]
                    logit = rel_logits[b, k]

                    target = torch.zeros(P, device=device)
                    if pred_id < P:
                        target[pred_id] = 1.0
                    loss = F.binary_cross_entropy_with_logits(logit, target, reduction='mean')
                    total_loss = total_loss + loss
                    num_rels += 1

        if num_rels > 0:
            total_loss = total_loss / num_rels

        return total_loss
