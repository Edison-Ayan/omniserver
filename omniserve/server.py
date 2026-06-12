"""omniserve 的 OpenAI 兼容 HTTP server。

暴露 `POST /v1/chat/completions`(流式 + 非流式),底层是 continuous-batching 引擎,
所以它表现得像一个 drop-in 的多模态端点——和 vLLM server 一样的接口形态。图像以
OpenAI vision 格式接收(`image_url` 里放 base64 data URI)。

引擎不是线程安全的,而且在 GPU 上跑模型,所以它跑在单个后台线程上。HTTP handler 提交
一个 Request,通过线程安全队列取出该请求的输出,再用 run_in_executor 桥接到 asyncio。

运行:
    python -m omniserve.server --port 8000 [--fp16] [--prefix-cache]
然后:
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

# 踩坑:本文件用了 `from __future__ import annotations`,类型注解会变成字符串,
# FastAPI 解析 handler 的 `HTTPRequest` 注解时只在「模块全局」里找。所以这些
# fastapi 导入必须放模块级——放在 build_app() 内部(局部)会导致 FastAPI 解析不到
# Request 类型,把它当成 query 参数,请求直接 422。
from fastapi import FastAPI
from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from .engine import LLMEngine
from .request import Request, SamplingParams
from .scheduler import SchedulerConfig

MODEL_NAME = "omniserve-qwen2-vl"


# --------------------------------------------------------------------------- #
# Engine service:在后台线程上驱动步进循环。
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
            # 空闲时阻塞等请求到来;否则只是轮询。
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
# OpenAI 请求解析 / 响应组装。
# --------------------------------------------------------------------------- #
def _decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        url = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")


def parse_messages(messages: List[dict]) -> Tuple[str, List[Image.Image]]:
    """把 OpenAI chat messages 摊平成 (prompt_text, images)。引擎的 runner 只格式化
    单轮 user 对话,所以这里把文本拼起来、把图收集起来。
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
