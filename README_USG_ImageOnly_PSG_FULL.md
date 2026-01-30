# Image-only USG (USG-Par) on PSG — Full-Reproduction Engineering README (v2)

This document is a **complete, self-contained engineering specification** for reproducing
the **image-only USG-Par** model on the **Panoptic Scene Graph (PSG)** dataset with
**paper-grade performance**, without requiring direct reading of the paper.

Every architectural choice, training target, loss, and evaluation rule that materially
affects performance is explicitly stated.

---

# 1. Task Definition

## Input
- Single RGB image I.

## Output (SGDet)
- Objects:
  - mask (H×W)
  - category_id
  - confidence score
- Relations:
  - (subject_index, object_index)
  - predicate_id
  - confidence score

## Metrics
- Recall@K and mean Recall@K
- K ∈ {10, 20, 50, 100}
- Track R@50 and mR@50 during development.

---

# 2. High-Level Architecture

Pipeline:

Image  
→ Frozen OpenCLIP ConvNeXt Backbone  
→ Multi-scale Pixel Decoder (Mask2Former style)  
→ Shared Mask Decoder (100 queries)  
→ Object Predictions (class + mask)  
→ Relation Proposal Constructor (RPC)  
→ Top-K Object Pairs  
→ Relation Decoder (Transformer)  
→ Predicate Predictions  

---

# 3. Backbone and Pixel Decoder

## Backbone
- OpenCLIP ConvNeXt-Large
- Frozen parameters

## Pixel Decoder
- Deformable-Attention Multi-scale Encoder (Mask2Former style)
- Feature levels: 3
- Output dimension: 256

Rationale:
- Provides strong dense features for mask prediction.
- Matches Mask2Former behavior used in high-performance segmentation.

---

# 4. Shared Mask Decoder

- Number of queries Q = 100
- Layers: 6
- Each layer:
  - Self-attention on queries
  - Masked cross-attention (queries → pixel features)
  - FFN

Outputs per query:
- q_i ∈ R^256
- mask_logits_i ∈ R^{H×W}

---

# 5. Open-Vocabulary Classification Head

Instead of FC classifier:

1. Encode class names once using CLIP text encoder.
2. Normalize embeddings.

For each query:

logits = normalize(q) · normalize(E_class)^T

Loss:
- Sigmoid Cross-Entropy

PSG:
- 133 object classes
- 56 predicate classes

Cache text embeddings and reuse.

---

# 6. Hungarian Matching (Mandatory)

For each image, match Q predicted queries to N_gt ground-truth objects.

### Matching Cost

cost =  
1.0 * classification_cost  
+ 5.0 * mask_bce_cost  
+ 5.0 * mask_dice_cost  

### After Matching

Matched queries:
- Supervised with GT label + GT mask

Unmatched queries:
- Labeled as no-object

---

# 7. Object Loss

L_obj =
1.0 * L_cls  
+ 5.0 * L_mask_bce  
+ 5.0 * L_mask_dice  

Where:
- L_cls: sigmoid CE
- L_mask_bce: binary CE
- L_mask_dice: Dice loss

---

# 8. Relation Proposal Constructor (RPC)

Goal: Identify promising object pairs.

## Steps

1. Project queries:

S = MLP_s(q)  
O = MLP_o(q)

2. Two-way Relation-Aware Cross-Attention (RAC)

Repeat 2 layers:

S = S + CrossAttn(S, O)  
O = O + CrossAttn(O, S)

3. Pair Confidence Matrix

C[i,j] = cosine_similarity(S[i], O[j])

4. Select Top-K pairs per image

Recommended:
- K = 200

---

# 9. Pair Loss

Binary matrix G:

G[i,j] = 1 if (i,j) corresponds to any GT relation pair  
Else 0

Loss:

L_pair = Weighted BCE(C, G)

Positive weight:
- pos_weight = 5.0

---

# 10. Relation Query Construction

For each selected pair (i,j):

r_ij = MLP( concat(q_i , q_j) )

Dimension: 256

---

# 11. Relation Decoder

Transformer Decoder:

- Layers: 6
- d_model = 256
- Self-attention on relation queries
- Cross-attention to pixel decoder features

Output:

p_ij ∈ R^256

Predicate logits:

logits = normalize(p_ij) · normalize(E_predicate)^T

Loss:
- Sigmoid CE

---

# 12. Relation Loss

L_rel = L_predicate_cls + L_pair

---

# 13. Total Loss

L = 1.0 * L_obj + 1.0 * L_rel

---

# 14. Training Hyperparameters

Optimizer: AdamW  
Learning rate: 1e-3  
Weight decay: 1e-2  
Batch size: as large as GPU allows  
AMP: enabled  
Gradient clip: 1.0  

LR Schedule:
- Linear warmup 1000 iters
- Then cosine decay

---

# 15. PSG Dataset Loader

Each sample must return:

image  
gt_labels [N]  
gt_masks [N,H,W]  
gt_relations: list of (sub_idx, obj_idx, predicate_id)

All indices refer to GT object ordering.

---

# 16. Evaluation (SGDet)

For each predicted relation:

final_score =
subject_score *
object_score *
pair_score *
predicate_score

Ranking:

Sort relations by final_score.

Compute Recall@K and mean Recall@K.

---

# 17. Required File Structure

usg_psg/
  usg/
    models/
      image_backbone.py
      pixel_decoder.py
      mask_decoder.py
      openvocab_head.py
      matcher_hungarian.py
      rpc.py
      relation_decoder.py
      losses.py
      usg_par_image.py
    datasets/
      psg_dataset.py
    eval/
      sgdet_metrics.py
  train.py
  eval.py

---

# 18. Sanity Checks

Object Stage:
- #matched queries == #GT objects
- Masks visually align with GT

Relation Stage:
- Oracle recall of GT pairs inside top-K > 80%

Training:
- L_obj decreases first
- L_rel decreases after RPC is added

---

# 19. Common Failure Modes

- No Hungarian matching → collapse
- No L_pair → relations fail
- FC classifier instead of text-dot → lower generalization
- Predicting all Q×Q pairs → noisy + slow

---

# 20. Implementation Contract

If the implementation follows this document exactly,
the resulting system is structurally equivalent to
image-only USG-Par and capable of paper-grade PSG SGDet performance.
