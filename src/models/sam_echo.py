from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry


class SamEcho:
    """Thin inference wrapper matching the authors' public SAM-Echo notebook."""

    def __init__(
        self,
        checkpoint: str | Path,
        model_type: str = "vit_b",
        device: str = "cuda",
        amp: bool = True,
        input_size: int = 1024,
    ) -> None:
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.amp = bool(amp and self.device.type == "cuda")
        self.input_size = input_size
        model = sam_model_registry[model_type](checkpoint=None)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
        self.checkpoint_metadata = {
            key: payload.get(key) for key in ("epoch", "dice_score") if isinstance(payload, dict) and key in payload
        }
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp)

    @torch.inference_mode()
    def encode(self, image_chw: np.ndarray | torch.Tensor) -> torch.Tensor:
        image = torch.as_tensor(image_chw, dtype=torch.float32, device=self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        with self._autocast():
            embedding = self.model.image_encoder(image)
        return embedding

    @torch.inference_mode()
    def decode(self, embedding: torch.Tensor, box_1024: np.ndarray | torch.Tensor, output_shape: tuple[int, int]) -> np.ndarray:
        boxes = torch.as_tensor(box_1024, dtype=torch.float32, device=self.device)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        embedding = embedding.to(self.device)
        with self._autocast():
            sparse, dense = self.model.prompt_encoder(points=None, boxes=boxes, masks=None)
            logits, _ = self.model.mask_decoder(
                image_embeddings=embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            logits = F.interpolate(logits, size=output_shape, mode="bilinear", align_corners=False)
        return (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()

    def decode_logits_with_grad(self, embeddings: torch.Tensor, boxes_1024: torch.Tensor) -> torch.Tensor:
        """Decode a batch while preserving gradients to box coordinates.

        SAM's reference mask decoder assumes one image with multiple prompts,
        so independent image/box pairs are decoded in a short loop.
        All SAM parameters remain frozen; only gradients to boxes are retained.
        """
        outputs = []
        for index in range(embeddings.shape[0]):
            box = boxes_1024[index : index + 1].to(self.device, dtype=torch.float32)
            embedding = embeddings[index : index + 1].to(self.device, dtype=torch.float32)
            sparse, dense = self.model.prompt_encoder(points=None, boxes=box, masks=None)
            logits, _ = self.model.mask_decoder(
                image_embeddings=embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            outputs.append(logits)
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def predict(self, image_chw: np.ndarray, box_1024: np.ndarray, output_shape: tuple[int, int]) -> tuple[np.ndarray, torch.Tensor]:
        embedding = self.encode(image_chw)
        mask = self.decode(embedding, box_1024, output_shape)
        return mask, embedding
