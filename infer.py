# infer.py
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open_clip

# Add Mask2Former to path (check multiple possible locations)
for mask2former_path in [
    '/mnt/d/JSR/scene_graph/USG/Mask2Former',
    '/home/keti/Mask2Former',
    '/home/keti/dev_sr/usg-simple/mask2former',
    os.path.join(os.path.dirname(__file__), 'mask2former'),
]:
    if os.path.exists(mask2former_path) and mask2former_path not in sys.path:
        sys.path.insert(0, mask2former_path)
        break

from usg.data.vocab import PSG_OBJECTS, PSG_PREDICATES
from usg.modeling.usgpar_image import USGParImageOnly


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def preprocess_with_openclip(img_bgr: np.ndarray, clip_model="convnext_base", pretrained="laion400m_s13b_b51k") -> torch.Tensor:
    _, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil = Image.fromarray(img_rgb)
    x = preprocess(pil).unsqueeze(0)  # (1,3,H,W)
    return x


def color_from_id(i: int) -> Tuple[int, int, int]:
    # deterministic BGR color
    rng = (i * 2654435761) & 0xFFFFFFFF
    b = 50 + (rng & 127)
    g = 50 + ((rng >> 8) & 127)
    r = 50 + ((rng >> 16) & 127)
    return int(b), int(g), int(r)


def overlay_masks(
    base_bgr: np.ndarray,
    masks: np.ndarray,          # (Q,H,W) in {0,1}
    labels: List[str],
    scores: List[float],
    alpha: float = 0.45,
) -> np.ndarray:
    out = base_bgr.copy()
    H, W = out.shape[:2]
    for qi, m in enumerate(masks):
        if m.sum() < 10:
            continue
        color = color_from_id(qi)
        colored = np.zeros_like(out, dtype=np.uint8)
        colored[m.astype(bool)] = color
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)

        # label text at mask centroid
        ys, xs = np.where(m.astype(bool))
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        text = f"{labels[qi]} {scores[qi]:.2f}"
        cv2.putText(out, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    return out


def save_graph_image(nodes: List[str], edges: List[Tuple[int,int,str,float]], out_path: str):
    """
    논문 스타일의 깔끔한 Scene Graph 시각화
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import to_rgba
    except Exception as e:
        raise RuntimeError("그래프 저장엔 networkx, matplotlib가 필요합니다. pip install networkx matplotlib\n" + str(e))

    if len(nodes) == 0:
        print("No nodes to visualize.")
        return

    G = nx.DiGraph()

    # 노드 추가 (라벨에서 객체 이름만 추출)
    node_labels = {}
    for i, n in enumerate(nodes):
        # "0:person(0.85)" -> "person"
        parts = n.split(":")
        if len(parts) > 1:
            obj_name = parts[1].split("(")[0]
        else:
            obj_name = n.split("(")[0]
        G.add_node(i)
        node_labels[i] = obj_name

    # 엣지 추가
    edge_labels = {}
    for s, o, p, sc in edges:
        if s < len(nodes) and o < len(nodes):
            G.add_edge(s, o)
            edge_labels[(s, o)] = p  # predicate만 표시

    # 색상 팔레트 (파스텔 톤)
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8B500', '#00CED1', '#FF69B4', '#32CD32', '#FFD700'
    ]
    node_colors = [colors[i % len(colors)] for i in range(len(nodes))]

    # Figure 설정
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_facecolor('#FAFAFA')

    # 레이아웃 선택 (노드 수에 따라)
    if len(nodes) <= 5:
        pos = nx.circular_layout(G, scale=2)
    else:
        try:
            pos = nx.kamada_kawai_layout(G, scale=3)
        except:
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

    # 노드 그리기 (둥근 사각형 스타일)
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=3000,
        node_shape='o',
        edgecolors='#333333',
        linewidths=2,
        alpha=0.9,
        ax=ax
    )

    # 노드 라벨 (객체 이름)
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=11,
        font_weight='bold',
        font_color='#1a1a1a',
        ax=ax
    )

    # 엣지 그리기 (곡선 화살표)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#666666',
        width=2,
        alpha=0.7,
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        min_source_margin=30,
        min_target_margin=30,
        ax=ax
    )

    # 엣지 라벨 (관계)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=9,
        font_color='#E74C3C',
        font_weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E74C3C', alpha=0.8),
        ax=ax
    )

    # 타이틀
    ax.set_title('Scene Graph', fontsize=16, fontweight='bold', color='#333333', pad=20)

    # 축 제거
    ax.axis('off')

    # 여백 조정
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="images.jpg")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None, help="Path to trained checkpoint (optional)")
    ap.add_argument("--top_obj", type=int, default=12)
    ap.add_argument("--top_edges", type=int, default=20)
    ap.add_argument("--mask_thr", type=float, default=0.5)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)

    # model
    model = USGParImageOnly(
        obj_class_names=PSG_OBJECTS,
        pred_class_names=PSG_PREDICATES,
        num_queries=100,
        topk_pairs=200,
        clip_model="convnext_base",
        clip_pretrained="laion400m_s13b_b51k",
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully")

    model.eval()
    model.build_text_embeddings(device=device)

    x = preprocess_with_openclip(img).to(device)

    with torch.no_grad():
        out = model(x)

    # object scores/labels
    obj_logits = out["obj_logits_open"][0]  # (Q,Cobj)
    obj_prob = obj_logits.softmax(-1)
    obj_score, obj_cls = obj_prob.max(-1)   # (Q,)

    # top objects (선정된 query만 오버레이/그래프에 사용)
    topk = min(args.top_obj, obj_score.numel())
    topv, topi = torch.topk(obj_score, k=topk)

    # masks
    mask_logits = out["mask_logits"][0]  # (Q,Hm,Wm)
    mask_prob = torch.sigmoid(mask_logits)

    # 원본 크기로 리사이즈
    H, W = img.shape[:2]
    mask_prob_up = F.interpolate(mask_prob.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]  # (Q,H,W)

    # 선택된 query만 마스크 추출
    sel_masks = []
    sel_labels = []
    sel_scores = []
    sel_qids = topi.tolist()

    for qi in sel_qids:
        m = (mask_prob_up[qi].cpu().numpy() >= args.mask_thr).astype(np.uint8)
        sel_masks.append(m)
        cls = int(obj_cls[qi].item())
        sel_labels.append(PSG_OBJECTS[cls])
        sel_scores.append(float(obj_score[qi].item()))

    overlay = overlay_masks(img, np.stack(sel_masks, 0), sel_labels, sel_scores, alpha=0.45)
    out_img_path = os.path.join(args.outdir, "overlay.png")
    cv2.imwrite(out_img_path, overlay)
    print("Saved:", out_img_path)

    # relations -> 그래프 edge 만들기
    # out["rel_logits"]: (B,K,Cpred) - one logit per pair
    rel_logits = out["rel_logits"][0]  # (K,Cpred)
    sub_idx = out["pair_sub_idx"][0]   # (K,)
    obj_idx = out["pair_obj_idx"][0]   # (K,)
    pair_scores = out["pair_scores"][0]# (K,)

    K = sub_idx.numel()
    rel_prob = rel_logits.sigmoid()    # Use sigmoid for multi-label
    rel_sc, rel_cls = rel_prob.max(-1) # (K,)

    # Debug info
    print(f"\n=== Debug Info ===")
    print(f"Selected top {len(sel_qids)} objects: {sel_qids}")
    print(f"Object labels: {sel_labels}")
    print(f"Object scores: {[f'{s:.3f}' for s in sel_scores]}")
    print(f"Number of relation pairs (K): {K}")
    print(f"Pair sub_idx range: {sub_idx.min().item()} ~ {sub_idx.max().item()}")
    print(f"Pair obj_idx range: {obj_idx.min().item()} ~ {obj_idx.max().item()}")
    print(f"Pair scores range: {pair_scores.min().item():.3f} ~ {pair_scores.max().item():.3f}")
    print(f"Relation scores range: {rel_sc.min().item():.3f} ~ {rel_sc.max().item():.3f}")

    # 그래프에는 "선정된 query(top objects)"만 노드로 씀
    qid_to_node = {qid: ni for ni, qid in enumerate(sel_qids)}
    nodes = [f"{i}:{sel_labels[i]}({sel_scores[i]:.2f})" for i in range(len(sel_qids))]

    # edges 후보 중에서, 양끝이 모두 선택된 노드인 것만
    edges = []
    for t in range(K):
        qs = int(sub_idx[t].item())
        qo = int(obj_idx[t].item())
        if qs not in qid_to_node or qo not in qid_to_node:
            continue
        pred = PSG_PREDICATES[int(rel_cls[t].item())]
        sc = float(rel_sc[t].item())
        edges.append((qid_to_node[qs], qid_to_node[qo], pred, sc))

    print(f"Edges found (both endpoints in top objects): {len(edges)}")

    # score 상위만 저장
    edges = sorted(edges, key=lambda x: x[3], reverse=True)[:args.top_edges]

    if len(edges) > 0:
        print(f"Top edges: {edges[:5]}")
        out_graph_path = os.path.join(args.outdir, "graph.png")
        save_graph_image(nodes, edges, out_graph_path)
        print("Saved:", out_graph_path)
    else:
        print("Edges not found - RPC pairs와 top objects가 겹치지 않습니다.")
        print("Hint: --top_obj 값을 늘리거나, 학습을 더 진행해보세요.")


if __name__ == "__main__":
    main()
