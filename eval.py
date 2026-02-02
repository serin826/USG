# eval.py
"""
USG-Par Evaluation Script for PSG Dataset (SGDet)

Usage:
    python eval.py --checkpoint checkpoints/usg_best.pth --data_root /path/to/OpenPSG/data
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add Mask2Former to path (check multiple possible locations)
for mask2former_path in [
    '/mnt/d/JSR/scene_graph/USG/Mask2Former',
    '/home/keti/Mask2Former',
    '/home/keti/dev_sr/USG/Mask2Former',  # Remote server path
    '/home/keti/dev_sr/usg-simple/mask2former',
    os.path.join(os.path.dirname(__file__), 'Mask2Former'),
    os.path.join(os.path.dirname(__file__), 'mask2former'),
]:
    if os.path.exists(mask2former_path) and mask2former_path not in sys.path:
        sys.path.insert(0, mask2former_path)
        break

from usg.data.vocab import PSG_OBJECTS, PSG_PREDICATES, NUM_OBJECT_CLASSES, NUM_PREDICATE_CLASSES
from usg.data.psg_dataset import PSGDataset, psg_collate_fn, PSGTransform
from usg.modeling.usgpar_image import USGParImageOnly
from usg.eval.sgdet_metrics import SGDetEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate USG-Par on PSG dataset')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='/mnt/d/JSR/scene_graph/USG/OpenPSG/data',
                        help='Path to OpenPSG data directory')
    parser.add_argument('--ann_file', type=str, default='psg/psg_train_val.json',
                        help='Annotation file relative to data_root')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')

    # Model
    parser.add_argument('--clip_model', type=str, default='convnext_base')
    parser.add_argument('--clip_pretrained', type=str, default='laion400m_s13b_b51k')
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--topk_pairs', type=int, default=200)
    parser.add_argument('--d_model', type=int, default=256)

    # Eval
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--iou_threshold', type=float, default=0.5)

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def build_dataloader(args):
    """Build PSG dataloader for evaluation."""
    ann_path = os.path.join(args.data_root, args.ann_file)
    img_dir = os.path.join(args.data_root, 'coco/coco')
    seg_dir = os.path.join(args.data_root, 'coco/coco')

    transform = PSGTransform(size=args.img_size, training=False)

    dataset = PSGDataset(
        ann_file=ann_path,
        img_dir=img_dir,
        seg_dir=seg_dir,
        split=args.split,
        transform=transform,
        num_object_classes=NUM_OBJECT_CLASSES,
        num_predicate_classes=NUM_PREDICATE_CLASSES,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=psg_collate_fn,
        pin_memory=True,
    )

    return loader


def build_model(args, device):
    """Build and load USG-Par model."""
    model = USGParImageOnly(
        obj_class_names=PSG_OBJECTS,
        pred_class_names=PSG_PREDICATES,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        freeze_backbone=True,
        num_queries=args.num_queries,
        topk_pairs=args.topk_pairs,
        d_model=args.d_model,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded successfully")

    # Build text embeddings
    model.build_text_embeddings(device=device)

    return model


@torch.no_grad()
def evaluate(model, loader, evaluator, device, args):
    """Run evaluation."""
    model.eval()

    for batch in tqdm(loader, desc='Evaluating'):
        images = batch['images'].to(device)
        gt_masks = batch['gt_masks']
        gt_classes = batch['gt_classes']
        rel_pairs = batch['rel_pairs']
        rel_labels = batch['rel_labels']

        # Forward pass
        outputs = model(images)

        # Get object scores from classification logits
        obj_logits = outputs['obj_logits_open']  # (B, Q, C)
        obj_scores = obj_logits.sigmoid().max(dim=-1).values  # (B, Q)
        obj_classes = obj_logits.argmax(dim=-1)  # (B, Q)

        # Add to evaluator
        evaluator.add_batch(
            pred_masks=outputs['mask_logits'],
            pred_obj_scores=obj_scores,
            pred_obj_classes=obj_classes,
            pred_rel_logits=outputs['rel_logits'],
            pred_pair_scores=outputs['pair_scores'],
            pred_sub_idx=outputs['pair_sub_idx'],
            pred_obj_idx=outputs['pair_obj_idx'],
            gt_masks=gt_masks,
            gt_classes=gt_classes,
            rel_pairs=rel_pairs,
            rel_labels=rel_labels,
        )

    # Compute and print results
    metrics = evaluator.print_results()
    return metrics


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build dataloader
    print(f"Building {args.split} dataloader...")
    loader = build_dataloader(args)
    print(f"Dataset: {len(loader.dataset)} samples, {len(loader)} batches")

    # Build model
    print("Building model...")
    model = build_model(args, device)

    # Create evaluator
    evaluator = SGDetEvaluator(
        num_predicates=NUM_PREDICATE_CLASSES,
        iou_threshold=args.iou_threshold,
    )

    # Run evaluation
    print(f"\nEvaluating on {args.split} split...")
    metrics = evaluate(model, loader, evaluator, device, args)

    # Save results
    result_file = args.checkpoint.replace('.pth', f'_{args.split}_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"IoU Threshold: {args.iou_threshold}\n\n")
        f.write("Results:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
    print(f"Results saved to: {result_file}")


if __name__ == '__main__':
    main()
