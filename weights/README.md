# Pretrained checkpoints (not redistributed)

Put these files in this folder, or point `RMRM_WEIGHTS` at a folder containing them. `src/rmrm/backbones.py` loads
them by file name. All encoders are frozen, and none is fine-tuned.

| backbone (role) | file | sha256 | source |
|---|---|---|---|
| DINOv2 ViT-S/14 (**primary**) | `dinov2_vits14_pretrain.pth` | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | official DINOv2 release (dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) |
| DeiT-S/16 (robustness) | `deit_small_patch16_224-cd65a155.pth` | `cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f` | official DeiT release (dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| OpenAI CLIP ViT-B/16 (robustness) | `clip_openai_ViT-B-16.pt` | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` | OpenAI CLIP release (`ViT-B-16.pt` from openaipublic.azureedge.net/clip/models/…), loaded as TorchScript with open_clip |

If `dinov2_vits14_pretrain.pth` is absent, the code falls back to timm `vit_small_patch14_dinov2.lvd142m` from the
Hugging Face hub. Those are the same weights, but this fallback has not been verified byte-for-byte. Verify the
checksums with `sha256sum *`.
