# usg/data/psg_dataset.py
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PSGDataset(Dataset):
    """
    PSG (Panoptic Scene Graph) Dataset

    Annotation structure:
    - segments_info: list of {id, category_id, iscrowd, isthing, area}
    - relations: list of [subject_idx, object_idx, predicate_idx]
    - pan_seg_file_name: panoptic segmentation mask file
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        seg_dir: str,
        split: str = "train",
        transform=None,
        max_relations: int = 100,
        num_object_classes: int = 133,
        num_predicate_classes: int = 56,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.split = split
        self.transform = transform
        self.max_relations = max_relations
        self.num_object_classes = num_object_classes
        self.num_predicate_classes = num_predicate_classes

        # Load annotations
        with open(ann_file, 'r') as f:
            raw = json.load(f)

        self.thing_classes = raw['thing_classes']
        self.stuff_classes = raw['stuff_classes']
        self.predicate_classes = raw['predicate_classes']
        self.all_classes = self.thing_classes + self.stuff_classes

        # Filter by split
        all_data = raw['data']
        test_ids = set(raw.get('test_image_ids', []))

        if split == "train":
            # Train: exclude test images, require relations
            self.data = [d for d in all_data if d['image_id'] not in test_ids and len(d.get('relations', [])) > 0]
        elif split == "val":
            # Val: only test images with relations (for SGDet evaluation)
            self.data = [d for d in all_data if d['image_id'] in test_ids and len(d.get('relations', [])) > 0]
        elif split == "test":
            # Test: same as val (PSG uses test_image_ids for validation)
            self.data = [d for d in all_data if d['image_id'] in test_ids and len(d.get('relations', [])) > 0]
        else:
            self.data = all_data

        print(f"[PSGDataset] Loaded {len(self.data)} samples for split='{split}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Load image
        img_path = os.path.join(self.img_dir, item['file_name'])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load panoptic segmentation
        seg_path = os.path.join(self.seg_dir, item['pan_seg_file_name'])
        pan_seg = np.array(Image.open(seg_path))

        # Convert RGB to segment ID: R + G*256 + B*256*256
        if pan_seg.ndim == 3:
            pan_seg_id = pan_seg[:, :, 0].astype(np.int32) + \
                         pan_seg[:, :, 1].astype(np.int32) * 256 + \
                         pan_seg[:, :, 2].astype(np.int32) * 256 * 256
        else:
            pan_seg_id = pan_seg.astype(np.int32)

        # Parse segments
        segments_info = item['segments_info']
        relations = item.get('relations', [])

        # Build segment ID to index mapping
        seg_id_to_idx = {}
        gt_classes = []
        gt_masks = []

        H, W = pan_seg_id.shape

        for i, seg in enumerate(segments_info):
            seg_id = seg['id']
            cat_id = seg['category_id']
            seg_id_to_idx[seg_id] = i
            gt_classes.append(cat_id)

            # Extract binary mask for this segment
            mask = (pan_seg_id == seg_id).astype(np.float32)
            gt_masks.append(mask)

        num_objects = len(segments_info)

        # Parse relations: [sub_idx, obj_idx, pred_idx]
        rel_pairs = []
        rel_labels = []
        for rel in relations[:self.max_relations]:
            sub_idx, obj_idx, pred_idx = rel
            if sub_idx < num_objects and obj_idx < num_objects:
                rel_pairs.append([sub_idx, obj_idx])
                rel_labels.append(pred_idx)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, masks=gt_masks)
            image = transformed['image']
            gt_masks = transformed['masks']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            gt_masks = [torch.from_numpy(m) for m in gt_masks]

        # Stack masks
        if len(gt_masks) > 0:
            gt_masks = torch.stack(gt_masks, dim=0)  # (N, H, W)
        else:
            gt_masks = torch.zeros(0, H, W)

        gt_classes = torch.tensor(gt_classes, dtype=torch.long)

        # Relation tensors
        if len(rel_pairs) > 0:
            rel_pairs = torch.tensor(rel_pairs, dtype=torch.long)  # (R, 2)
            rel_labels = torch.tensor(rel_labels, dtype=torch.long)  # (R,)
        else:
            rel_pairs = torch.zeros(0, 2, dtype=torch.long)
            rel_labels = torch.zeros(0, dtype=torch.long)

        return {
            'image': image,  # (3, H, W)
            'gt_masks': gt_masks,  # (N, H, W)
            'gt_classes': gt_classes,  # (N,)
            'rel_pairs': rel_pairs,  # (R, 2)
            'rel_labels': rel_labels,  # (R,)
            'image_id': item['image_id'],
            'num_objects': num_objects,
        }


def psg_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for PSG dataset.
    Handles variable number of objects and relations per image.
    """
    images = torch.stack([b['image'] for b in batch], dim=0)

    # Variable length data - keep as lists
    gt_masks = [b['gt_masks'] for b in batch]
    gt_classes = [b['gt_classes'] for b in batch]
    rel_pairs = [b['rel_pairs'] for b in batch]
    rel_labels = [b['rel_labels'] for b in batch]
    image_ids = [b['image_id'] for b in batch]
    num_objects = [b['num_objects'] for b in batch]

    return {
        'images': images,  # (B, 3, H, W)
        'gt_masks': gt_masks,  # list of (Ni, H, W)
        'gt_classes': gt_classes,  # list of (Ni,)
        'rel_pairs': rel_pairs,  # list of (Ri, 2)
        'rel_labels': rel_labels,  # list of (Ri,)
        'image_ids': image_ids,
        'num_objects': num_objects,
    }


class PSGTransform:
    """Transforms for PSG dataset with mask support"""

    def __init__(self, size: int = 640, training: bool = True):
        self.size = size
        self.training = training

    def __call__(self, image: np.ndarray, masks: List[np.ndarray]) -> Dict:
        H, W = image.shape[:2]

        # Resize
        scale = self.size / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)

        image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        resized_masks = []
        for m in masks:
            rm = cv2.resize(m, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(rm)

        # Pad to square
        pad_H = self.size - new_H
        pad_W = self.size - new_W

        image = np.pad(image, ((0, pad_H), (0, pad_W), (0, 0)), mode='constant', constant_values=0)
        padded_masks = []
        for m in resized_masks:
            pm = np.pad(m, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
            padded_masks.append(pm)

        # Random horizontal flip (training only)
        if self.training and np.random.rand() > 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            padded_masks = [np.ascontiguousarray(m[:, ::-1]) for m in padded_masks]

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        masks_tensor = [torch.from_numpy(m.astype(np.float32)) for m in padded_masks]

        # Normalize (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return {'image': image, 'masks': masks_tensor}
