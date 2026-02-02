# usg/modeling/m2f_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

# Detectron2
try:
    from detectron2.layers import ShapeSpec
except Exception as e:
    raise ImportError(
        "detectron2 import 실패. detectron2를 먼저 설치해야 합니다.\n"
        f"원인: {e}"
    )

# Mask2Former (설치/경로에 따라 import가 달라질 수 있음)
_M2F_IMPORT_ERROR = None
try:
    # 보통: pip로 설치된 mask2former 패키지
    from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
    from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
        MultiScaleMaskedTransformerDecoder,
    )
except Exception as e1:
    _M2F_IMPORT_ERROR = e1
    try:
        # detectron2 projects/Mask2Former를 PYTHONPATH에 올린 경우
        from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
        from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
            MultiScaleMaskedTransformerDecoder,
        )
    except Exception as e2:
        raise ImportError(
            "Mask2Former import 실패.\n"
            "해결: (1) detectron2와 Mask2Former를 설치하거나, (2) Mask2Former 프로젝트 경로를 PYTHONPATH에 추가하세요.\n"
            f"첫 시도 에러: {e1}\n"
            f"두번째 시도 에러: {e2}"
        )


@dataclass
class M2FOutputs:
    pred_masks: torch.Tensor                 # (B, Q, Hm, Wm)
    pred_logits_closed: Optional[torch.Tensor]  # (B, Q, C+1)  (Mask2Former 기본 head)
    query_features: torch.Tensor             # (B, Q, D)
    mask_features: torch.Tensor              # (B, D, Hm, Wm)
    multi_scale_features: List[torch.Tensor] # list[(B, D, Hi, Wi)] length=3
    context_tokens: torch.Tensor             # (B, HW, D)  (relation decoder용)


class Mask2FormerAdapter(nn.Module):
    """
    - MSDeformAttnPixelDecoder + MultiScaleMaskedTransformerDecoder를 사용
    - 핵심: pred_masks 뿐 아니라 "query_features(B,Q,D)"를 꺼내서 open-vocab / relation에 사용
    """

    def __init__(
        self,
        in_channels: List[int],
        strides: List[int],
        conv_dim: int = 256,
        mask_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dec_layers: int = 9,
        num_classes_closed: int = 133,  # closed-set head용 (원하면 무시 가능)
    ):
        super().__init__()
        assert len(in_channels) == len(strides), "in_channels/strides 길이가 같아야 합니다."

        # Detectron2-style feature dict
        self.in_features = [f"res{i}" for i in range(2, 2 + len(in_channels))]  # res2,res3,res4,res5
        input_shape = {
            name: ShapeSpec(channels=c, stride=s)
            for name, c, s in zip(self.in_features, in_channels, strides)
        }

        # Pixel decoder (MSDeformAttn)
        # Pixel decoder (MSDeformAttn) — 버전 호환 안전판
        sig = inspect.signature(MSDeformAttnPixelDecoder.__init__)

        kwargs_pd = dict(
            input_shape=input_shape,
            conv_dim=conv_dim,
            mask_dim=mask_dim,
            transformer_in_features=self.in_features,
            common_stride=4,
            transformer_dropout=0.0,
            transformer_nheads=nheads,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
        )

        # ⚠️ num_feature_levels는 지원될 때만 넣는다
        if "num_feature_levels" in sig.parameters:
            kwargs_pd["num_feature_levels"] = len(self.in_features)

        # 기타 버전별 옵션
        if "norm" in sig.parameters:
            kwargs_pd["norm"] = "GN"
        if "dropout" in sig.parameters:
            kwargs_pd["dropout"] = 0.0
        if "deformable_transformer_encoder_in_features" in sig.parameters:
            kwargs_pd["deformable_transformer_encoder_in_features"] = self.in_features

        self.pixel_decoder = MSDeformAttnPixelDecoder(**kwargs_pd)

        # Mask2Former transformer decoder (기본 closed-set head도 같이 돌릴 수 있음)
        self.mask_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=conv_dim,
            mask_classification=True,
            num_classes=num_classes_closed,
            hidden_dim=conv_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=2048,
            dec_layers=dec_layers,
            pre_norm=False,
            mask_dim=mask_dim,
            enforce_input_project=False,
        )

        self._last_query_features: Optional[torch.Tensor] = None
        self._patch_mask_decoder()

    def _patch_mask_decoder(self):
        """
        Mask2Former의 decoder_norm에 hook을 걸어서 query features를 캡처합니다.
        decoder_norm의 출력이 normalized query features입니다.
        forward_prediction_heads가 여러 번 호출되므로 마지막 호출의 결과가 저장됩니다.
        """
        adapter_self = self

        # decoder_norm에 hook 등록 (forward_prediction_heads에서 호출됨)
        if hasattr(self.mask_decoder, 'decoder_norm'):
            def hook_fn(module, inp, out):
                # out: normalized transformer output (Q, B, D)
                if torch.is_tensor(out) and out.dim() == 3:
                    # (Q, B, D) -> (B, Q, D)
                    if out.shape[0] > out.shape[1]:  # Q > B인 경우 (일반적)
                        adapter_self._last_query_features = out.transpose(0, 1)
                    else:
                        adapter_self._last_query_features = out

            self.mask_decoder.decoder_norm.register_forward_hook(hook_fn)
        else:
            # decoder_norm이 없는 경우: 마지막 FFN layer에 hook
            if hasattr(self.mask_decoder, 'transformer_ffn_layers'):
                last_ffn = self.mask_decoder.transformer_ffn_layers[-1]

                def hook_fn(module, inp, out):
                    if torch.is_tensor(out) and out.dim() == 3:
                        # (Q, B, D) -> (B, Q, D)
                        if out.shape[0] > out.shape[1]:
                            adapter_self._last_query_features = out.transpose(0, 1)
                        else:
                            adapter_self._last_query_features = out

                last_ffn.register_forward_hook(hook_fn)

    def forward(self, feats_list: List[torch.Tensor]) -> M2FOutputs:
        assert len(feats_list) == len(self.in_features), (
            f"backbone stage feature 개수({len(feats_list)})와 in_features({len(self.in_features)})가 다릅니다."
        )

        features: Dict[str, torch.Tensor] = {
            name: t for name, t in zip(self.in_features, feats_list)
        }

        # pixel decoder
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)

        # context tokens(관계 디코더용): 가장 낮은 해상도(보통 multi_scale_features[-1])를 flatten
        ctx_map = multi_scale_features[-1]  # (B,D,H,W)
        context_tokens = ctx_map.flatten(2).transpose(1, 2)  # (B,HW,D)

        # mask decoder
        self._last_query_features = None
        out = self.mask_decoder(multi_scale_features, mask_features)

        pred_masks = out["pred_masks"]
        pred_logits = out.get("pred_logits", None)

        # query feature 확보
        q = None
        # 1) dict에서 직접 제공되는 경우
        for k in ["pred_embeds", "query_features"]:
            if k in out and torch.is_tensor(out[k]):
                q = out[k]
                break
        # 2) hook에서 잡힌 경우
        if q is None and self._last_query_features is not None:
            q = self._last_query_features

        if q is None:
            raise RuntimeError(
                "Mask2Former에서 query features(B,Q,D)를 추출하지 못했습니다.\n"
                "현재 mask2former 구현이 hook 추출 패턴과 다를 수 있습니다.\n"
                "해결: mask2former의 transformer decoder forward에서 hs(queries)를 반환하도록 1줄 패치가 필요합니다."
            )

        # q를 (B,Q,D)로 강제
        if q.dim() != 3:
            raise RuntimeError(f"query feature 차원이 예상과 다릅니다: {tuple(q.shape)}")
        # pred_masks의 query 차원과 일치 확인
        if q.shape[1] != pred_masks.shape[1]:
            # 어떤 구현은 (Q,B,D)로 나올 수 있어 transpose 시도
            if q.shape[0] == pred_masks.shape[1] and q.shape[1] == pred_masks.shape[0]:
                q = q.transpose(0, 1)
            else:
                # 마지막 수단: 자르거나 맞추기(권장X)
                minq = min(q.shape[1], pred_masks.shape[1])
                q = q[:, :minq]
                pred_masks = pred_masks[:, :minq]
                if pred_logits is not None:
                    pred_logits = pred_logits[:, :minq]

        return M2FOutputs(
            pred_masks=pred_masks,
            pred_logits_closed=pred_logits,
            query_features=q,
            mask_features=mask_features,
            multi_scale_features=multi_scale_features,
            context_tokens=context_tokens,
        )
