"""OpenAI-compatible HTTP server for omniserve.

Exposes `POST /v1/chat/completions` (streaming + non-streaming) backed by the
continuous-batching engine, so it behaves like a drop-in multimodal endpoint —
the same shape you'd hit on vLLM's server. Images are accepted in the OpenAI
vision format (`image_url` with a base64 data URI).

The engine is not thread-safe and runs the model on the GPU, so it lives on a
single background thread. HTTP handlers submit a Request and drain that
request's output via a thread-safe queue, bridged to asyncio with run_in_executor.

Run:
    python -m omniserve.server --port 8000 [--fp16] [--prefix-cache]
Then:
    curl localhost:8000/v1/chat/completions -d '{"model":"qwen2-vl","messages":[...]}'
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import queue
import threading
import time
import uuid
from typing import List, Tuple

from fastapi import FastAPI
from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from .engine import LLMEngine
from .request import Request, SamplingParams
from .scheduler import SchedulerConfig

MODEL_NAME = "omniserve-qwen2-vl"


# --------------------------------------------------------------------------- #
# Engine service: drives the step loop on a background thread.
# --------------------------------------------------------------------------- #
class EngineService:
    def __init__(self, runner, sched_config: SchedulerConfig):
        self.engine = LLMEngine(runner, sched_config)
        self._submissions: "queue.Queue" = queue.Queue()
        self._out: dict = {}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, request: Request) -> "queue.Queue":
        out_q: "queue.Queue" = queue.Queue()
        self._submissions.put((request, out_q))
        return out_q

    def _drain_submissions(self, block: bool) -> None:
        if block:
            req, out_q = self._submissions.get()
            self._out[self.engine.add_request(req)] = out_q
        try:
            while True:
                req, out_q = self._submissions.get_nowait()
                self._out[self.engine.add_request(req)] = out_q
        except queue.Empty:
            pass

    def _loop(self) -> None:
        while True:
            # If idle, block until a request arrives; otherwise just poll.
            self._drain_submissions(block=not self.engine.has_unfinished())
            if not self.engine.has_unfinished():
                continue
            for delta in self.engine.step():
                out_q = self._out.get(delta.request_id)
                if out_q is not None:
                    out_q.put(delta)
                    if delta.finished:
                        self._out.pop(delta.request_id, None)


# --------------------------------------------------------------------------- #
# OpenAI request parsing / response shaping.
# --------------------------------------------------------------------------- #
def _decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        url = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")


def parse_messages(messages: List[dict]) -> Tuple[str, List[Image.Image]]:
    """Flatten OpenAI chat messages into (prompt_text, images). The engine's
    runner formats a single user turn, so we concatenate text and collect images.
    """
    texts: List[str] = []
    images: List[Image.Image] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    images.append(_decode_image(part["image_url"]["url"]))
    return "\n".join(t for t in texts if t), images


def _completion(text: str, finish_reason: str = "stop") -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text}}],
    }


def _chunk(delta_text: str, finish_reason=None) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "delta": {"content": delta_text} if delta_text else {}}],
    }


# --------------------------------------------------------------------------- #
# FastAPI app.
# --------------------------------------------------------------------------- #
def build_app(service: EngineService):
    app = FastAPI(title="omniserve")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(http_req: HTTPRequest):
        body = await http_req.json()
        prompt, images = parse_messages(body.get("messages", []))
        sampling = SamplingParams(
            max_new_tokens=int(body.get("max_tokens", 128)),
            temperature=float(body.get("temperature", 0.0)),
        )
        out_q = service.submit(Request(prompt=prompt, images=images, sampling=sampling))
        loop = asyncio.get_event_loop()

        if body.get("stream"):
            async def sse():
                while True:
                    delta = await loop.run_in_executor(None, out_q.get)
                    if delta.text:
                        yield f"data: {json.dumps(_chunk(delta.text))}\n\n"
                    if delta.finished:
                        yield f"data: {json.dumps(_chunk('', 'stop'))}\n\n"
                        yield "data: [DONE]\n\n"
                        return
            return StreamingResponse(sse(), media_type="text/event-stream")

        text = ""
        while True:
            delta = await loop.run_in_executor(None, out_q.get)
            text += delta.text
            if delta.finished:
                break
        return JSONResponse(_completion(text))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--vision-cache", action="store_true")
    ap.add_argument("--max-running", type=int, default=16)
    args = ap.parse_args()

    from .runners.qwen_vl import QwenVLRunner

    vc = pc = None
    if args.vision_cache:
        from .cache import VisionEmbeddingCache
        vc = VisionEmbeddingCache(max_entries=64)
    if args.prefix_cache:
        from .cache import PrefixKVCache
        pc = PrefixKVCache(max_entries=64)

    print("loading model ...")
    runner = QwenVLRunner(load_in_4bit=not args.fp16, vision_cache=vc, prefix_cache=pc)
    service = EngineService(runner, SchedulerConfig(max_running=args.max_running,
                                                    max_prefill_per_step=1))
    app = build_app(service)

    import uvicorn
    print(f"omniserve listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
