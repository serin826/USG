# usg/eval/sgdet_metrics.py
"""
SGDet Evaluation Metrics for PSG.

For each predicted relation:
    final_score = subject_score * object_score * pair_score * predicate_score

Metrics:
- Recall@K: fraction of GT relations with a matching prediction in top-K
- Mean Recall@K: average per-predicate recall (for class balance)

K values: 10, 20, 50, 100
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SGDetEvaluator:
    """
    Scene Graph Detection Evaluator.

    Computes Recall@K and mean Recall@K for K in {10, 20, 50, 100}.
    """

    def __init__(self, num_predicates: int = 56, iou_threshold: float = 0.5):
        self.num_predicates = num_predicates
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset accumulated results."""
        self.all_gt_triplets = []  # List of (sub_mask, obj_mask, pred_id) per image
        self.all_pred_triplets = []  # List of (sub_mask, obj_mask, pred_id, score) per image

        # For mean recall (per-predicate)
        self.per_predicate_gt_count = defaultdict(int)
        self.per_predicate_hit_count = defaultdict(lambda: defaultdict(int))  # pred_id -> K -> count

    def add_batch(
        self,
        pred_masks: torch.Tensor,  # (B, Q, H, W) predicted masks (logits)
        pred_obj_scores: torch.Tensor,  # (B, Q) object confidence scores
        pred_obj_classes: torch.Tensor,  # (B, Q) predicted object classes
        pred_rel_logits: torch.Tensor,  # (B, K, P) predicate logits
        pred_pair_scores: torch.Tensor,  # (B, K) pair confidence scores
        pred_sub_idx: torch.Tensor,  # (B, K) subject indices
        pred_obj_idx: torch.Tensor,  # (B, K) object indices
        gt_masks: List[torch.Tensor],  # list of (N, H, W) GT masks
        gt_classes: List[torch.Tensor],  # list of (N,) GT classes
        rel_pairs: List[torch.Tensor],  # list of (R, 2) GT relation pairs
        rel_labels: List[torch.Tensor],  # list of (R,) GT predicate labels
    ):
        """
        Add a batch of predictions and ground truths for evaluation.
        """
        B = pred_masks.shape[0]
        device = pred_masks.device

        # Convert mask logits to binary masks
        pred_masks_binary = (pred_masks.sigmoid() > 0.5).float()

        for b in range(B):
            # Get predictions for this image
            masks_b = pred_masks_binary[b]  # (Q, H, W)
            obj_scores_b = pred_obj_scores[b]  # (Q,)
            rel_logits_b = pred_rel_logits[b]  # (K, P)
            pair_scores_b = pred_pair_scores[b]  # (K,)
            sub_idx_b = pred_sub_idx[b]  # (K,)
            obj_idx_b = pred_obj_idx[b]  # (K,)

            # Get GT for this image
            gt_masks_b = gt_masks[b].to(device).float()  # (N, H, W)
            gt_classes_b = gt_classes[b]  # (N,)
            gt_pairs_b = rel_pairs[b]  # (R, 2)
            gt_labels_b = rel_labels[b]  # (R,)

            if len(gt_pairs_b) == 0:
                continue

            # Resize GT masks if needed
            H, W = masks_b.shape[1], masks_b.shape[2]
            if gt_masks_b.shape[1:] != (H, W):
                gt_masks_b = F.interpolate(
                    gt_masks_b.unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )[0]
            gt_masks_b = (gt_masks_b > 0.5).float()

            # Compute predicate scores (softmax over predicates)
            pred_scores_b = rel_logits_b.sigmoid()  # (K, P)

            # Build list of predicted triplets with scores
            # final_score = sub_score * obj_score * pair_score * pred_score
            pred_triplets = []
            K = sub_idx_b.shape[0]

            for k in range(K):
                sub_i = sub_idx_b[k].item()
                obj_i = obj_idx_b[k].item()

                sub_score = obj_scores_b[sub_i].item()
                obj_score = obj_scores_b[obj_i].item()
                pair_score = pair_scores_b[k].item()

                sub_mask = masks_b[sub_i]  # (H, W)
                obj_mask = masks_b[obj_i]  # (H, W)

                for p in range(self.num_predicates):
                    pred_score = pred_scores_b[k, p].item()
                    final_score = sub_score * obj_score * pair_score * pred_score

                    pred_triplets.append({
                        'sub_mask': sub_mask,
                        'obj_mask': obj_mask,
                        'predicate': p,
                        'score': final_score,
                        'sub_idx': sub_i,
                        'obj_idx': obj_i,
                    })

            # Sort by score
            pred_triplets.sort(key=lambda x: x['score'], reverse=True)

            # Build GT triplets
            gt_triplets = []
            for r in range(len(gt_pairs_b)):
                sub_gt_idx = gt_pairs_b[r, 0].item()
                obj_gt_idx = gt_pairs_b[r, 1].item()
                pred_id = gt_labels_b[r].item()

                gt_triplets.append({
                    'sub_mask': gt_masks_b[sub_gt_idx],
                    'obj_mask': gt_masks_b[obj_gt_idx],
                    'predicate': pred_id,
                })

                # Count per-predicate GT
                self.per_predicate_gt_count[pred_id] += 1

            # Evaluate at different K values
            for K_val in [10, 20, 50, 100]:
                top_k_preds = pred_triplets[:K_val]

                # Check which GT triplets are hit
                gt_hit = [False] * len(gt_triplets)

                for pred in top_k_preds:
                    for gt_idx, gt in enumerate(gt_triplets):
                        if gt_hit[gt_idx]:
                            continue

                        # Check predicate match
                        if pred['predicate'] != gt['predicate']:
                            continue

                        # Check mask IoU
                        sub_iou = self._compute_iou(pred['sub_mask'], gt['sub_mask'])
                        obj_iou = self._compute_iou(pred['obj_mask'], gt['obj_mask'])

                        if sub_iou >= self.iou_threshold and obj_iou >= self.iou_threshold:
                            gt_hit[gt_idx] = True
                            # Record per-predicate hit
                            self.per_predicate_hit_count[gt['predicate']][K_val] += 1
                            break

            self.all_gt_triplets.append(gt_triplets)
            self.all_pred_triplets.append(pred_triplets)

    def _compute_iou(self, mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """Compute IoU between two binary masks."""
        intersection = (mask1 * mask2).sum().item()
        union = (mask1 + mask2).clamp(0, 1).sum().item()

        if union == 0:
            return 0.0
        return intersection / union

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dict with R@K and mR@K for K in {10, 20, 50, 100}
        """
        results = {}

        # Total GT relations
        total_gt = sum(self.per_predicate_gt_count.values())

        for K in [10, 20, 50, 100]:
            # Recall@K: total hits / total GT
            total_hits = sum(
                self.per_predicate_hit_count[p][K]
                for p in self.per_predicate_gt_count.keys()
            )
            recall_k = total_hits / total_gt if total_gt > 0 else 0.0
            results[f'R@{K}'] = recall_k

            # Mean Recall@K: average per-predicate recall
            predicate_recalls = []
            for p in range(self.num_predicates):
                gt_count = self.per_predicate_gt_count[p]
                if gt_count > 0:
                    hit_count = self.per_predicate_hit_count[p][K]
                    predicate_recalls.append(hit_count / gt_count)

            mean_recall_k = np.mean(predicate_recalls) if predicate_recalls else 0.0
            results[f'mR@{K}'] = mean_recall_k

        return results

    def print_results(self):
        """Print evaluation results."""
        metrics = self.compute_metrics()

        print("\n" + "=" * 50)
        print("SGDet Evaluation Results")
        print("=" * 50)

        print("\nRecall@K:")
        for K in [10, 20, 50, 100]:
            print(f"  R@{K}: {metrics[f'R@{K}']:.4f}")

        print("\nMean Recall@K:")
        for K in [10, 20, 50, 100]:
            print(f"  mR@{K}: {metrics[f'mR@{K}']:.4f}")

        print("=" * 50 + "\n")

        return metrics
