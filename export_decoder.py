"""
Export the SAM 2 prompt encoder + mask decoder to ONNX for the browser.

Runs at Docker build time (CPU is fine). The browser loads the .onnx with
onnxruntime-web, feeds it the embedding from the `embed` action plus click
points, and gets mask logits back in ~tens of ms.

Inputs (all float32 unless noted):
  image_embed       [1, 256, 64, 64]
  high_res_feats_0  [1, 32, 256, 256]
  high_res_feats_1  [1, 64, 128, 128]
  point_coords      [1, N, 2]   click points in 1024x1024 input space (x, y)
  point_labels      [1, N]      1 = include, 0 = exclude
  mask_input        [1, 1, 256, 256]  previous low-res logits, or zeros
  has_mask_input    [1]         1.0 if mask_input is used, else 0.0
Outputs:
  masks             [1, 3, 256, 256]  low-res mask logits (> 0 = inside)
  iou_predictions   [1, 3]            pick the highest
"""

import argparse

import torch
import torch.nn as nn

from sam2.build_sam import build_sam2


class SAM2DecoderOnnx(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.mask_decoder = sam_model.sam_mask_decoder
        self.image_size = sam_model.image_size

    def _embed_points(self, coords, labels):
        coords = coords + 0.5
        pad_pt = torch.zeros((coords.shape[0], 1, 2), dtype=coords.dtype)
        pad_lb = -torch.ones((labels.shape[0], 1), dtype=labels.dtype)
        coords = torch.cat([coords, pad_pt], dim=1)
        labels = torch.cat([labels, pad_lb], dim=1)
        coords = coords / self.image_size
        emb = self.prompt_encoder.pe_layer._pe_encoding(coords)
        lab = labels.unsqueeze(-1).expand_as(emb)
        emb = emb * (lab != -1)
        emb = emb + self.prompt_encoder.not_a_point_embed.weight * (lab == -1)
        for i in range(self.prompt_encoder.num_point_embeddings):
            emb = emb + self.prompt_encoder.point_embeddings[i].weight * (lab == i)
        return emb

    def _embed_mask(self, mask_input, has_mask_input):
        dense = has_mask_input * self.prompt_encoder.mask_downscaling(mask_input)
        dense = dense + (1 - has_mask_input) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return dense

    def forward(self, image_embed, high_res_feats_0, high_res_feats_1,
                point_coords, point_labels, mask_input, has_mask_input):
        sparse = self._embed_points(point_coords, point_labels)
        dense = self._embed_mask(mask_input, has_mask_input)
        masks, iou, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )
        # Index 0 is the single-mask output; 1..3 are the multimask candidates.
        return masks[:, 1:, :, :], iou[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--checkpoint", default="/models/sam2.1_hiera_base_plus.pt")
    ap.add_argument("--out", default="/models/sam2_decoder.onnx")
    args = ap.parse_args()

    model = build_sam2(args.config, args.checkpoint, device="cpu").eval()
    dec = SAM2DecoderOnnx(model).eval()

    dummy = (
        torch.randn(1, 256, 64, 64),
        torch.randn(1, 32, 256, 256),
        torch.randn(1, 64, 128, 128),
        torch.randint(0, 1024, (1, 2, 2)).float(),
        torch.ones(1, 2),
        torch.zeros(1, 1, 256, 256),
        torch.zeros(1),
    )
    names_in = ["image_embed", "high_res_feats_0", "high_res_feats_1",
                "point_coords", "point_labels", "mask_input", "has_mask_input"]
    with torch.no_grad():
        torch.onnx.export(
            dec, dummy, args.out,
            input_names=names_in,
            output_names=["masks", "iou_predictions"],
            dynamic_axes={"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}},
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
