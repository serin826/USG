# usg/modeling/backbone_openclip.py
import torch
import torch.nn as nn
import open_clip

class OpenCLIPConvNeXtBackbone(nn.Module):
    def __init__(
        self,
        model_name="convnext_base",
        pretrained="laion400m_s13b_b51k",
        freeze=True,
        input_size=224,
    ):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = self.model.visual
        self.trunk = self.visual.trunk  # timm convnext trunk

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.stage_channels, self.strides = self._infer_stage_meta(input_size)

    @torch.no_grad()
    def _infer_stage_meta(self, input_size=224):
        self.eval()
        device = next(self.model.parameters()).device
        x = torch.zeros(1, 3, input_size, input_size, device=device)
        feats = self.forward(x)  # list[(B,C,H,W)]
        stage_channels = [f.shape[1] for f in feats]
        strides = [int(round(input_size / f.shape[2])) for f in feats]
        return stage_channels, strides

    def _forward_stages(self, images: torch.Tensor):
        """
        timm ConvNeXt trunk에서 stem -> stages 순으로 통과시키며
        stage별 feature (B,C,H,W) 4개를 수집
        """
        trunk = self.trunk

        # 1) stem 통과 (3ch -> C)
        if hasattr(trunk, "stem"):
            x = trunk.stem(images)
        elif hasattr(trunk, "stem_0"):  # 혹시 다른 네이밍
            x = trunk.stem_0(images)
        else:
            raise RuntimeError("ConvNeXt trunk에서 stem을 찾지 못했습니다. trunk 구조 확인 필요.")

        feats = []
        if not hasattr(trunk, "stages"):
            raise RuntimeError("ConvNeXt trunk에서 stages를 찾지 못했습니다. trunk 구조 확인 필요.")

        # 2) stage 순회하며 feature 수집
        for s in trunk.stages:
            x = s(x)
            if torch.is_tensor(x) and x.dim() == 4:
                feats.append(x)

        if len(feats) == 0:
            raise RuntimeError("stages 순회에서 4D feature를 수집하지 못했습니다.")

        return feats

    def forward(self, images: torch.Tensor):
        # 1) 우선 forward_features 시도
        if hasattr(self.trunk, "forward_features"):
            out = self.trunk.forward_features(images)

            # ✅ 어떤 open_clip/timm 조합은 마지막 feature 텐서만 반환함
            if torch.is_tensor(out):
                return self._forward_stages(images)

            if isinstance(out, (list, tuple)):
                return list(out)

            if isinstance(out, dict):
                feats = [v for v in out.values() if torch.is_tensor(v) and v.dim() == 4]
                if len(feats) > 0:
                    return feats

            # 예상 못한 타입이면 fallback
            return self._forward_stages(images)

        # 2) forward_features 자체가 없으면 fallback
        return self._forward_stages(images)
