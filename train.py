# train.py
"""
USG-Par Training Script for PSG Dataset

Based on USG (Universal Scene Graph Generation) paper settings:
- OpenCLIP ConvNeXt backbone (frozen)
- Mask2Former-style decoder
- Relation Proposal Constructor + Relation Decoder
- Hungarian matching for query-GT assignment

Usage:
    python train.py --data_root /path/to/OpenPSG/data --epochs 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

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
from usg.modeling.losses import USGLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train USG-Par on PSG dataset')

    # Data
    parser.add_argument('--data_root', type=str, default='/mnt/d/JSR/scene_graph/USG/OpenPSG/data',
                        help='Path to OpenPSG data directory')
    parser.add_argument('--ann_file', type=str, default='psg/psg_train_val.json',
                        help='Annotation file relative to data_root')

    # Model
    parser.add_argument('--clip_model', type=str, default='convnext_base')
    parser.add_argument('--clip_pretrained', type=str, default='laion400m_s13b_b51k')
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--topk_pairs', type=int, default=200)
    parser.add_argument('--d_model', type=int, default=256)

    # Training (README spec: lr=1e-3, weight_decay=1e-2, warmup=1000 iters, grad_clip=1.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)  # backbone frozen anyway
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=640)

    # Loss weights (README spec: L_obj = 1.0*cls + 5.0*bce + 5.0*dice, L_rel = pred_cls + pair)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_bce', type=float, default=5.0)
    parser.add_argument('--lambda_dice', type=float, default=5.0)
    parser.add_argument('--pair_pos_weight', type=float, default=5.0)

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision training')

    return parser.parse_args()


def build_dataloader(args, split='train'):
    """Build PSG dataloader"""
    ann_path = os.path.join(args.data_root, args.ann_file)
    img_dir = os.path.join(args.data_root, 'coco/coco')
    seg_dir = os.path.join(args.data_root, 'coco/coco')

    transform = PSGTransform(size=args.img_size, training=(split == 'train'))

    dataset = PSGDataset(
        ann_file=ann_path,
        img_dir=img_dir,
        seg_dir=seg_dir,
        split=split,
        transform=transform,
        num_object_classes=NUM_OBJECT_CLASSES,
        num_predicate_classes=NUM_PREDICATE_CLASSES,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.num_workers,
        collate_fn=psg_collate_fn,
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return loader


def build_model(args, device):
    """Build USG-Par model"""
    model = USGParImageOnly(
        obj_class_names=PSG_OBJECTS,
        pred_class_names=PSG_PREDICATES,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        freeze_backbone=True,  # Freeze backbone initially
        num_queries=args.num_queries,
        topk_pairs=args.topk_pairs,
        d_model=args.d_model,
    ).to(device)

    # Build text embeddings
    model.build_text_embeddings(device=device)

    return model


def build_optimizer(model, args):
    """Build optimizer with different LR for backbone"""
    # Separate parameters
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': other_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': args.backbone_lr},
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    return optimizer


def build_scheduler(optimizer, args, steps_per_epoch):
    """Build learning rate scheduler with warmup (README: 1000 iter warmup + cosine decay)"""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_iters  # 1000 iterations, not epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch, args):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_obj = 0
    total_rel = 0
    total_cls = 0
    total_bce = 0
    total_dice = 0
    total_pred_cls = 0
    total_pair = 0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        images = batch['images'].to(device)
        gt_masks = batch['gt_masks']
        gt_classes = batch['gt_classes']
        rel_pairs = batch['rel_pairs']
        rel_labels = batch['rel_labels']

        optimizer.zero_grad()

        if args.amp:
            with autocast('cuda'):
                outputs = model(images)
                losses = criterion(outputs, gt_classes, gt_masks, rel_pairs, rel_labels)
                loss = losses['loss']

            scaler.scale(loss).backward()

            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            losses = criterion(outputs, gt_classes, gt_masks, rel_pairs, rel_labels)
            loss = losses['loss']

            loss.backward()

            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

        scheduler.step()

        # Accumulate losses (matching USGLoss output keys)
        total_loss += loss.item()
        total_obj += losses['L_obj'].item()
        total_rel += losses['L_rel'].item()
        total_cls += losses['loss_cls'].item()
        total_bce += losses['loss_bce'].item()
        total_dice += losses['loss_dice'].item()
        total_pred_cls += losses['loss_pred_cls'].item()
        total_pair += losses['loss_pair'].item()
        num_batches += 1

        # Logging
        if (batch_idx + 1) % args.log_freq == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_batches
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f} | L_obj: {total_obj/num_batches:.4f} "
                  f"(cls:{total_cls/num_batches:.4f}, bce:{total_bce/num_batches:.4f}, dice:{total_dice/num_batches:.4f}) | "
                  f"L_rel: {total_rel/num_batches:.4f} "
                  f"(pred:{total_pred_cls/num_batches:.4f}, pair:{total_pair/num_batches:.4f}) | "
                  f"LR: {lr:.6f} | Time: {elapsed:.1f}s")

    return {
        'loss': total_loss / num_batches,
        'L_obj': total_obj / num_batches,
        'L_rel': total_rel / num_batches,
        'loss_cls': total_cls / num_batches,
        'loss_bce': total_bce / num_batches,
        'loss_dice': total_dice / num_batches,
        'loss_pred_cls': total_pred_cls / num_batches,
        'loss_pair': total_pair / num_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, args):
    """Validate model"""
    model.eval()

    total_loss = 0
    num_batches = 0

    for batch in loader:
        images = batch['images'].to(device)
        gt_masks = batch['gt_masks']
        gt_classes = batch['gt_classes']
        rel_pairs = batch['rel_pairs']
        rel_labels = batch['rel_labels']

        outputs = model(images)
        losses = criterion(outputs, gt_classes, gt_masks, rel_pairs, rel_labels)

        total_loss += losses['loss'].item()
        num_batches += 1

    return {'loss': total_loss / num_batches}


def save_checkpoint(model, optimizer, scheduler, epoch, args, filename):
    """Save checkpoint"""
    os.makedirs(args.save_dir, exist_ok=True)
    filepath = os.path.join(args.save_dir, filename)

    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': vars(args),
    }

    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, args):
    """Load checkpoint"""
    ckpt = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)

    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])

    start_epoch = ckpt.get('epoch', 0) + 1
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch


def main():
    args = parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print("Building dataloaders...")
    train_loader = build_dataloader(args, split='train')
    val_loader = build_dataloader(args, split='val')
    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # Model
    print("Building model...")
    model = build_model(args, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params / 1e6:.2f}M")

    # Loss (README spec)
    criterion = USGLoss(
        num_classes=NUM_OBJECT_CLASSES,
        num_predicates=NUM_PREDICATE_CLASSES,
        lambda_cls=args.lambda_cls,
        lambda_bce=args.lambda_bce,
        lambda_dice=args.lambda_dice,
        pair_pos_weight=args.pair_pos_weight,
    )
    # Set relation modules for direct GT pair loss computation
    criterion.set_relation_modules(model.rel_query_builder, model.rel_decoder)

    # Optimizer & Scheduler
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, len(train_loader))

    # AMP scaler
    scaler = GradScaler('cuda') if args.amp else None

    # Resume
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"{'='*60}\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, args
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, args)
        print(f"Val Loss: {val_metrics['loss']:.4f}")

        # Save checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, args, f'usg_epoch_{epoch}.pth')

        # Save best
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            save_checkpoint(model, optimizer, scheduler, epoch, args, 'usg_best.pth')
            print(f"New best model saved (loss: {best_loss:.4f})")

    # Save final
    save_checkpoint(model, optimizer, scheduler, args.epochs, args, 'usg_final.pth')
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
