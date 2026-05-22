"""OCR supervision loss for glyph fidelity.

Run a fixed OCR model (easyocr / paddleocr) on decoded predictions, compare to
the expected text strings from the prompt. We use a character-level CTC-style
edit-distance approximation (differentiable surrogate: cross-entropy over
per-region character logits when an OCR head is co-trained).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class OcrLossConfig:
    char_ce_weight: float = 1.0
    region_presence_weight: float = 0.5


class GlyphHeadLoss:
    """Loss for a small character-classification head trained alongside the glyph branch.

    Head outputs logits over a fixed character vocabulary for each glyph region
    token sequence. We supervise with ground-truth character IDs derived from
    the prompt's brand/text spans.
    """

    def __init__(self, config: OcrLossConfig, num_classes: int, blank_id: int = 0) -> None:
        self.config = config
        self.num_classes = num_classes
        self.blank_id = blank_id

    def __call__(
        self,
        char_logits: Tensor,
        target_char_ids: Tensor,
        target_lengths: Tensor,
        input_lengths: Tensor,
        region_presence_logits: Optional[Tensor] = None,
        region_presence_target: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        log_probs = F.log_softmax(char_logits, dim=-1).transpose(0, 1)
        ctc = F.ctc_loss(
            log_probs,
            target_char_ids,
            input_lengths,
            target_lengths,
            blank=self.blank_id,
            zero_infinity=True,
            reduction="mean",
        )
        total = self.config.char_ce_weight * ctc
        out = {"ocr_ctc": ctc}

        if region_presence_logits is not None and region_presence_target is not None:
            pres = F.binary_cross_entropy_with_logits(
                region_presence_logits, region_presence_target.float()
            )
            total = total + self.config.region_presence_weight * pres
            out["ocr_presence"] = pres

        out["ocr_total"] = total
        return out
