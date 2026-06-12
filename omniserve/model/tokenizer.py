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
        # 踩坑:tokenizer.json 只到 id 151654,<|image_pad|>(151655)/<|video_pad|>
        # 在 tokenizer_config.json 的 added_tokens_decoder 里(transformers 会读,
        # raw tokenizers 库不读)。必须手动补,且按 id 顺序加才能落到 151655/151656,
        # 否则 image_pad 会被当字面字符拆开,input_ids 完全对不上。
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
