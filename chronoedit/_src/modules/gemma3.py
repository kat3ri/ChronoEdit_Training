# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gemma 3 text encoder for ChronoEdit.

Drop-in replacement for UMT5-XXL: both produce 4096-dimensional embeddings
with 512 max tokens, so text_dim in the network config does not need to change.

Usage (offline pre-extraction)::

    python scripts/extract_gemma3.py --csv_path <metadata.csv>

Usage (online inference)::

    from chronoedit._src.modules.gemma3 import get_gemma3_embedding
    emb = get_gemma3_embedding("Add sunglasses to the person")  # [1, 512, 4096]
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn

from chronoedit._ext.imaginaire.utils import distributed, log

__all__ = ["Gemma3EncoderModel", "get_gemma3_embedding"]


class Gemma3EncoderModel:
    """Wrapper around Gemma 3 12B encoder for text embedding extraction.

    Produces embeddings with dimension 4096, matching UMT5-XXL output.
    Embeddings are zero-padded or truncated to ``text_len`` tokens.

    Parameters
    ----------
    text_len : int
        Maximum sequence length (default 512, matching UMT5).
    dtype : torch.dtype
        Precision for model weights and output.
    device : torch.device | str
        Target device.
    model_name : str
        HuggingFace model identifier for Gemma 3.
    """

    def __init__(
        self,
        text_len: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[torch.device, str] = "cuda",
        model_name: str = "google/gemma-3-12b",
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.model_name = model_name

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers is required for Gemma 3 encoder. "
                "Install with: pip install transformers>=4.49.0"
            )

        log.info(f"Loading Gemma 3 tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        log.info(f"Loading Gemma 3 model from {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device} if isinstance(device, str) else None,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        if not isinstance(device, str):
            self.model.to(device)

        # Gemma 3 12B hidden size is 4096 — matching UMT5-XXL
        self._hidden_size = self.model.config.hidden_size
        assert self._hidden_size == 4096, (
            f"Expected Gemma 3 hidden_size=4096, got {self._hidden_size}. "
            "If using a different Gemma variant, update text_dim in network config."
        )
        log.info(
            f"Gemma 3 encoder ready: hidden_size={self._hidden_size}, "
            f"text_len={self.text_len}, dtype={self.dtype}"
        )

    @torch.inference_mode()
    def __call__(
        self,
        texts: Union[str, List[str]],
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """Encode text prompts into embeddings.

        Parameters
        ----------
        texts : str | list[str]
            Input prompts.
        device : torch.device | str | None
            Override device for output tensor.

        Returns
        -------
        torch.Tensor
            Shape ``[B, text_len, 4096]`` (zero-padded if shorter).
        """
        if device is None:
            device = self.device
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.text_len,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use last hidden state
        hidden_states = outputs.last_hidden_state  # [B, seq_len, 4096]

        # Zero out padding positions
        seq_lens = attention_mask.sum(dim=1).long()

        stack_emb = []
        for emb, length in zip(hidden_states, seq_lens):
            length = min(length.item(), self.text_len)
            if length >= self.text_len:
                stack_emb.append(emb[: self.text_len].to(dtype=self.dtype))
            else:
                zeros = torch.zeros(
                    self.text_len - length,
                    emb.shape[1],
                    device=emb.device,
                    dtype=self.dtype,
                )
                stack_emb.append(torch.cat([emb[:length].to(dtype=self.dtype), zeros], dim=0))

        result = torch.stack(stack_emb)
        return result.to(device)


# ---------------------------------------------------------------------------
# Global singleton (matches UMT5 pattern from umt5.py)
# ---------------------------------------------------------------------------

_gemma3_encoder: Optional[Gemma3EncoderModel] = None


def get_gemma3_embedding(
    prompts: Union[str, List[str]],
    device: str = "cuda",
    max_length: int = 512,
) -> torch.Tensor:
    """Get Gemma 3 text embeddings (global singleton, lazy init).

    Parameters
    ----------
    prompts : str | list[str]
        Text prompts to encode.
    device : str
        Target device.
    max_length : int
        Maximum token length.

    Returns
    -------
    torch.Tensor
        Shape ``[B, max_length, 4096]``.
    """
    global _gemma3_encoder
    if _gemma3_encoder is None:
        _gemma3_encoder = Gemma3EncoderModel(text_len=max_length, device=device)
    return _gemma3_encoder(prompts, device=device)


def free_gemma3_encoder():
    """Release the global Gemma 3 encoder and free GPU memory."""
    global _gemma3_encoder
    if _gemma3_encoder is not None:
        log.info("Deleting global Gemma 3 text encoder...")
        del _gemma3_encoder.tokenizer
        del _gemma3_encoder.model
        del _gemma3_encoder
        _gemma3_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Gemma 3 encoder freed.")
