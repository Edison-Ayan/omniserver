"""Qwen2-VL tokenizer + chat template on the standalone `tokenizers` library
(no transformers). Loads tokenizer.json, applies the chat template, and expands
each image placeholder to its vision-token count.
"""

from __future__ import annotations

from typing import List

from tokenizers import AddedToken, Tokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
SYSTEM = "You are a helpful assistant."
MERGE_SIZE = 2


class Qwen2VLTokenizer:
    def __init__(self, tokenizer_json_path: str):
        self.tok = Tokenizer.from_file(tokenizer_json_path)
        # tokenizer.json ships ids up to 151654; image/video pad live in
        # tokenizer_config.json (which the raw library doesn't read). Append them
        # in id order so they land on 151655 / 151656.
        if self.tok.token_to_id(IMAGE_PAD) is None:
            self.tok.add_special_tokens([
                AddedToken("<|image_pad|>", special=True),
                AddedToken("<|video_pad|>", special=True),
            ])

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    @staticmethod
    def _n_vision_tokens(grid_thw) -> int:
        t, h, w = (int(x) for x in grid_thw)
        return t * (h // MERGE_SIZE) * (w // MERGE_SIZE)

    def build_prompt(self, text: str, image_grids: List = ()) -> str:
        """Single user turn: system + (vision blocks) + text, with the assistant
        generation prefix. Each image expands to its vision-token count."""
        vision = ""
        for grid in image_grids:
            n = self._n_vision_tokens(grid)
            vision += VISION_START + IMAGE_PAD * n + VISION_END
        return (f"{IM_START}system\n{SYSTEM}{IM_END}\n"
                f"{IM_START}user\n{vision}{text}{IM_END}\n"
                f"{IM_START}assistant\n")

    def encode_prompt(self, text: str, image_grids: List = ()) -> List[int]:
        return self.encode(self.build_prompt(text, image_grids))
