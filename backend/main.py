"""AuraSight FastAPI application.

REST for control/settings/analysis, a WebSocket for live events, and a
low-rate MJPEG preview for the local UI. Binds to 127.0.0.1 only.

Run:  python -m backend.main      (or scripts/dev.ps1 / scripts/dev.sh)
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import hardware
from .camera.service import CameraState
from .config.settings import REPO_ROOT, SettingsStore
from .events import EventBus
from .pipeline import CameraNotRunning, NothingToReplay, Pipeline
from .speech.service import Priority
from .vision.analyzer import AnalyzerBusy

log = logging.getLogger("aurasight")


class CameraStartBody(BaseModel):
    index: int | None = Field(None, ge=0, le=16)


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    speak: bool = True


class SayBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class DescribeBody(BaseModel):
    speak: bool = True


def create_app(pipeline: Pipeline | None = None, start_pipeline: bool = True) -> FastAPI:
    holder: dict[str, Pipeline] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        p = pipeline or Pipeline(SettingsStore(), EventBus())
        holder["p"] = p
        if start_pipeline:
            await run_in_threadpool(p.start)
        yield
        await run_in_threadpool(p.shutdown)

    app = FastAPI(title="AuraSight", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"],
    )

    def P() -> Pipeline:
        return holder["p"]

    # -- system ----------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/status")
    def status() -> dict:
        return P().status()

    @app.get("/api/hardware")
    def hw() -> dict:
        return hardware.check()

    @app.get("/api/events")
    def events(limit: int = 50) -> list[dict]:
        return P().bus.history(min(max(limit, 1), 150))

    # -- settings ----------------------------------------------------------------
    @app.get("/api/settings")
    def get_settings() -> dict:
        return P().store.current.model_dump()

    @app.put("/api/settings")
    def put_settings(patch: dict[str, Any]) -> dict:
        try:
            return P().store.update(patch).model_dump()
        except (ValueError, ValidationError) as e:
            raise HTTPException(422, str(e)) from e

    # -- camera --------------------------------------------------------------------
    @app.get("/api/cameras")
    async def cameras() -> list[dict]:
        return await run_in_threadpool(P().cameras)

    @app.post("/api/camera/start")
    async def camera_start(body: CameraStartBody | None = None) -> dict:
        ok = await run_in_threadpool(P().start_camera, body.index if body else None)
        st = P().camera.status()
        if not ok:
            raise HTTPException(503, st.get("error") or "Camera failed to start")
        return st

    @app.post("/api/camera/pause")
    def camera_pause() -> dict:
        P().pause_camera()
        return P().camera.status()

    @app.post("/api/camera/resume")
    def camera_resume() -> dict:
        P().resume_camera()
        return P().camera.status()

    @app.post("/api/camera/stop")
    async def camera_stop() -> dict:
        await run_in_threadpool(P().stop_camera)
        return P().camera.status()

    @app.get("/api/stream.mjpg")
    async def stream(request: Request):
        p = P()
        if not p.camera.is_active:
            raise HTTPException(409, "Camera is not running")
        boundary = "frame"

        async def gen():
            last = -1
            while not await request.is_disconnected():
                if not p.camera.is_active:
                    break
                f = p.camera.latest()
                s = p.store.current
                if f is not None and f.id != last:
                    last = f.id
                    ok, buf = await run_in_threadpool(
                        cv2.imencode, ".jpg", f.image, [cv2.IMWRITE_JPEG_QUALITY, s.preview_quality])
                    if ok:
                        data = buf.tobytes()
                        yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                               f"Content-Length: {len(data)}\r\n\r\n").encode() + data + b"\r\n"
                await asyncio.sleep(1.0 / s.preview_fps)

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}",
                                 headers={"Cache-Control": "no-store"})

    # -- understanding -------------------------------------------------------------------
    @app.get("/api/objects")
    def objects(include_unverified: bool = False) -> list[dict]:
        p = P()
        objs = p.all_objects() if include_unverified else p.objects()
        return [o.to_dict() for o in objs]

    @app.post("/api/scene/describe")
    async def describe(body: DescribeBody | None = None) -> dict:
        try:
            return await run_in_threadpool(P().describe_scene, body.speak if body else True)
        except CameraNotRunning as e:
            raise HTTPException(409, str(e)) from e
        except AnalyzerBusy as e:
            raise HTTPException(429, str(e)) from e

    @app.post("/api/scene/replay")
    def replay() -> dict:
        try:
            return P().replay_description()
        except NothingToReplay as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/ask")
    async def ask(body: AskBody) -> dict:
        return await run_in_threadpool(P().ask, body.question, body.speak)

    @app.post("/api/summary")
    def summary() -> dict:
        return {"text": P().summary(speak=True)}

    @app.get("/api/vlm/status")
    def vlm_status() -> dict:
        return P().analyzer.status()

    @app.get("/api/vlm/debug")
    def vlm_debug() -> dict:
        """Last prompt, raw model output and image size sent - for diagnosing answers."""
        return P().analyzer.debug()

    @app.post("/api/vlm/start")
    def vlm_start() -> dict:
        P().analyzer.start_background()
        return P().analyzer.status()

    # -- speech -------------------------------------------------------------------------------
    @app.post("/api/speech/say")
    def say(body: SayBody) -> dict:
        return {"queued": P().speech.say(body.text, Priority.USER, kind="user")}

    @app.post("/api/speech/stop")
    def speech_stop() -> dict:
        P().stop_speech()
        return {"stopped": True}

    @app.post("/api/speech/test")
    def speech_test() -> dict:
        s = P().store.current
        text = f"This is AuraSight. Speech rate {s.speech_rate:.1f}, volume {round(s.speech_volume * 100)} percent."
        return {"queued": P().speech.say(text, Priority.USER, kind="test")}

    @app.get("/api/speech/voices")
    async def voices() -> list[dict]:
        sp = P().speech
        if sp.state not in ("ready", "speaking"):
            return []
        try:
            return await run_in_threadpool(sp.run_on_engine, lambda e: e.voices(), 10.0)
        except Exception:
            return []

    # -- websocket ----------------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        p = P()
        q = p.bus.subscribe()
        try:
            await websocket.send_json({
                "type": "hello", "status": p.status(), "settings": p.store.current.model_dump(),
                "history": p.bus.history(40),
            })

            async def receiver():
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

            recv = asyncio.create_task(receiver())
            try:
                while not recv.done():
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        ev = {"type": "keepalive"}
                    await websocket.send_json(ev)
            finally:
                recv.cancel()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            p.bus.unsubscribe(q)

    # -- built frontend (production mode) ---------------------------------------------------------
    dist = Path(os.environ.get("AURASIGHT_FRONTEND_DIST", REPO_ROOT / "frontend" / "dist"))
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

    return app


def run() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("AURASIGHT_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("AURASIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("AURASIGHT_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
