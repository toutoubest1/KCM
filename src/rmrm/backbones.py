"""Frozen pretrained vision encoders returning multi-layer CLS + patch tokens.

Every backbone exposes ``forward_layers(x) -> dict[layer] = (cls[B,D], patches[B,N,D])``
and ``grid`` (patch-grid H, W) for the current input size.  Layers are 1-indexed
transformer-block outputs; for the last block the model's final LayerNorm is
applied (as in standard feature extraction), intermediate blocks are raw
residual-stream outputs (their scale differences are handled by the robust
per-coordinate standardisation in the monitor).
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

WEIGHTS_DIR = os.environ.get("RMRM_WEIGHTS", os.path.join(os.path.dirname(__file__), "..", "..", "weights"))


class TimmViT(nn.Module):
    def __init__(self, name: str, layers, pretrained_file: str | None = None, hub_pretrained: bool = False,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, patch: int = 14):
        super().__init__()
        import timm
        kw = dict(dynamic_img_size=True)
        if hub_pretrained:
            self.m = timm.create_model(name, pretrained=True, **kw)
        else:
            self.m = timm.create_model(name, pretrained=False, **kw)
            if pretrained_file is not None:
                sd = torch.load(pretrained_file, map_location="cpu", weights_only=False)
                if "model" in sd:
                    sd = sd["model"]
                sd = _remap_dinov2(sd) if "dinov2" in name else sd
                missing, unexpected = self.m.load_state_dict(sd, strict=False)
                missing = [k for k in missing if not k.startswith("head")]
                assert not missing, f"missing keys: {missing[:10]}"
        self.m.eval()
        self.layers = sorted(layers)
        self.n_prefix = getattr(self.m, "num_prefix_tokens", 1)
        self.patch = patch
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.dim = self.m.embed_dim

    @torch.inference_mode()
    def forward_layers(self, x: torch.Tensor):
        """x: float tensor [B,3,H,W] in [0,1]."""
        x = (x - self.mean) / self.std
        m = self.m
        h = m.patch_embed(x)
        h = m._pos_embed(h)
        h = m.patch_drop(h) if hasattr(m, "patch_drop") else h
        h = m.norm_pre(h) if hasattr(m, "norm_pre") else h
        out = {}
        last = len(m.blocks)
        for i, blk in enumerate(m.blocks, start=1):
            h = blk(h)
            if i in self.layers:
                t = m.norm(h) if i == last else h
                out[i] = (t[:, 0].float(), t[:, self.n_prefix:].float())
            if i >= self.layers[-1]:
                break
        return out


def _remap_dinov2(sd):
    """Official facebookresearch DINOv2 checkpoints already use timm-compatible key names
    (cls_token, pos_embed, patch_embed.proj, blocks.i.{norm1,attn,ls1.gamma,norm2,mlp,ls2.gamma}, norm);
    only the SSL-only ``mask_token`` must be dropped.  UNVERIFIED until real weights are loaded:
    the assert on missing keys in TimmViT will catch any mismatch."""
    return {k: v for k, v in sd.items() if k != "mask_token"}


class OpenCLIPViT(nn.Module):
    def __init__(self, arch: str, weight_file: str, layers, openai_jit: bool = False):
        super().__init__()
        import open_clip
        if openai_jit:                       # official OpenAI CLIP release (TorchScript archive)
            from open_clip.openai import load_openai_model
            model = load_openai_model(weight_file, device="cpu").float()
        else:
            model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=weight_file)
        self.v = model.visual.eval()
        self.layers = sorted(layers)
        self.mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.patch = self.v.conv1.kernel_size[0]
        self.dim = self.v.conv1.out_channels

    @torch.inference_mode()
    def forward_layers(self, x):
        import torch.nn.functional as F
        v = self.v
        x = (x - self.mean) / self.std
        B, _, H, W = x.shape
        x = v.conv1(x)
        gh, gw = x.shape[-2:]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        pe = v.positional_embedding
        n0 = int((pe.shape[0] - 1) ** 0.5)
        pe_patch = pe[1:].reshape(1, n0, n0, -1).permute(0, 3, 1, 2)
        pe_patch = F.interpolate(pe_patch, size=(gh, gw), mode="bicubic", align_corners=False)
        pe_patch = pe_patch.permute(0, 2, 3, 1).reshape(gh * gw, -1)
        x = x + torch.cat([pe[:1], pe_patch], 0).to(x.dtype)
        x = v.patch_dropout(x)
        x = v.ln_pre(x)
        out = {}
        blocks = v.transformer.resblocks
        last = len(blocks)
        batch_first = getattr(v.transformer, "batch_first", True)
        if not batch_first:
            x = x.permute(1, 0, 2)
        for i, blk in enumerate(blocks, start=1):
            x = blk(x)
            if i in self.layers:
                t = x if batch_first else x.permute(1, 0, 2)
                if i == last:
                    t = v.ln_post(t)
                out[i] = (t[:, 0].float(), t[:, 1:].float())
            if i >= self.layers[-1]:
                break
        return out


def build_backbone(name: str, layers):
    """Supported names: dinov2_s14, clip_b32, vit_b16, clip_b16_openai, deit_s16."""
    if name == "dinov2_s14":
        f = os.path.join(WEIGHTS_DIR, "dinov2_vits14_pretrain.pth")
        if os.path.exists(f):
            return TimmViT("vit_small_patch14_dinov2", layers, pretrained_file=f, patch=14)
        return TimmViT("vit_small_patch14_dinov2.lvd142m", layers, hub_pretrained=True, patch=14)
    if name == "clip_b32":
        f = os.path.join(WEIGHTS_DIR, "vit_b_32-quickgelu-laion400m_e32-46683a32.pt")
        return OpenCLIPViT("ViT-B-32-quickgelu", f, layers)
    if name == "vit_b16":
        f = os.path.join(WEIGHTS_DIR, "jx_vit_base_p16_224-80ecf9dd.pth")
        return TimmViT("vit_base_patch16_224", layers, pretrained_file=f, patch=16)
    if name == "clip_b16_openai":          # stage-5 confirmatory backbone (OpenAI CLIP ViT-B/16, sha256 5806e77c...)
        return OpenCLIPViT("ViT-B-16", os.path.join(WEIGHTS_DIR, "clip_openai_ViT-B-16.pt"), layers, openai_jit=True)
    if name == "deit_s16":                 # stage-5 confirmatory backbone (supervised DeiT-S/16, ImageNet-1k, cd65a155)
        return TimmViT("deit_small_patch16_224", layers, pretrained_file=os.path.join(WEIGHTS_DIR, "deit_small_patch16_224-cd65a155.pth"), patch=16)
    raise ValueError(name)
