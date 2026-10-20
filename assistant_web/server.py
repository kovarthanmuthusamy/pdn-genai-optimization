"""Web page of the design assistant (thesis defense demo): FastAPI server + one static page.

What: loads the thesis model once (``assistant_web.engine``) and serves the page at http://HOST:PORT.
    Open it in the Windows browser (WSL forwards localhost). The page asks:
      GET  /api/setup   board geometry, parts, targets with margins, frequencies, examples, model info
      POST /api/check   {codes, target}                    -> verdict (safe + direct), spectrum, additions
      POST /api/map     {codes, index}                     -> predicted map at frequency point `index`
      POST /api/fixes   {codes, target, mode, max_radius}  -> nearest passing placements (thesis assistant)
Usage: python -m assistant_web.server            (stop with Ctrl+C)
Config keys: HOST, PORT, GPUS (env ASSIST_WEB_PORT, ASSIST_WEB_GPUS="0,1" override).
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
HOST = "127.0.0.1"
PORT = int(os.getenv("ASSIST_WEB_PORT", "8050"))
GPUS = tuple(int(g) for g in os.getenv("ASSIST_WEB_GPUS", "0,1").split(",") if g.strip())
# =============================================================================

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STATE: dict = {}


class CheckReq(BaseModel):
    codes: list[int]
    target: str


class MapReq(BaseModel):
    codes: list[int]
    index: int


class FixReq(BaseModel):
    codes: list[int]
    target: str
    mode: str = "safe"
    max_radius: int = 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    from assistant_web.board import geometry
    from assistant_web.engine import PARTS, Engine
    eng = Engine(gpus=GPUS)
    ex_path = HERE / "examples.json"
    examples = json.loads(ex_path.read_text()) if ex_path.is_file() else []
    STATE.update(engine=eng, setup={
        "board": geometry(), "parts": {str(k): v for k, v in PARTS.items()},
        "targets": eng.targets_payload(), "freqs_mhz": eng.f.tolist(), "bare": eng.bare.tolist(),
        "examples": examples, "model": eng.info()})
    print(f"\n  Design assistant ready: http://localhost:{PORT}\n", flush=True)
    yield
    eng.close()


app = FastAPI(title="PDN design assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_stale_page(request, call_next):
    """The browser re-checks the page files on every load, so an edited page is never served stale."""
    resp = await call_next(request)
    if request.url.path.startswith("/static/") and "/vendor/" not in request.url.path:
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def _call(fn, *args):
    try:
        return fn(*args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"ok": "engine" in STATE, "time": time.time()}


@app.get("/api/setup")
def setup():
    return STATE["setup"]


@app.post("/api/check")
def check(req: CheckReq):
    return _call(STATE["engine"].check, req.codes, req.target)


@app.post("/api/map")
def map_(req: MapReq):
    return _call(STATE["engine"].map, req.codes, req.index)


@app.post("/api/fixes")
def fixes(req: FixReq):
    return _call(STATE["engine"].fixes, req.codes, req.target, req.mode, max(1, min(3, req.max_radius)))


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
