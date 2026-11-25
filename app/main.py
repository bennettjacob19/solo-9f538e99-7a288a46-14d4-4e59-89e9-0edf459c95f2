"""FastAPI 入口：本地单卡夜班训练 + 定时评测演练。

端点：
* ``POST /plans``          提交排程请求；有解才创建演练会话，无解不启动
* ``GET  /sessions``       列出会话
* ``GET  /sessions/{id}``  查询模拟时钟与已执行时段
* ``POST /sessions/{id}/advance``  推进模拟时钟，重放训练/存档/评测
* ``GET  /sessions/{id}/results``  最终权重与不间断训练一致性结论
* ``GET  /docs``           Swagger UI（自动生成）
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from . import ml
from .schemas import PlanRequest
from .scheduler import EvaluationSpec, solve
from .simulator import CheckpointWriteFailed, Session

DATA_DIR = os.environ.get("APP_DATA_DIR", os.path.join(os.getcwd(), "sessions_data"))
os.makedirs(DATA_DIR, exist_ok=True)

SESSIONS: dict[str, Session] = {}
# 仅保护 SESSIONS 字典本身的注册/枚举，临界区绝不包含训练或写盘。
REGISTRY_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # 内存会话随进程结束；检查点文件保留在 DATA_DIR 供核查。


app = FastAPI(
    title="单卡夜班训练与定时评测演练",
    version="1.0.0",
    description=(
        "提交至多 24 批训练耗时、至多 4 场互不重叠评测及检查点/重跑预算，"
        "求解器在满足全部独占与续训约束时受理并启动确定性模拟；"
        "推进模拟时钟可查实跑时段、重跑批数与推理结果，最终权重与"
        "不间断训练逐元素一致。"
    ),
    lifespan=lifespan,
)


@app.post("/plans", status_code=201)
def create_plan(req: PlanRequest) -> dict:
    specs = [
        EvaluationSpec(
            eval_id=e.eval_id,
            start=e.start,
            end=e.end,
            min_checkpoint_batches=e.min_checkpoint_batches,
        )
        for e in req.evaluations
    ]
    result = solve(
        durations=list(req.durations),
        evaluations=specs,
        max_checkpoints=req.max_checkpoints,
        max_rerun_batches=req.max_rerun_batches,
    )
    if not result.feasible:
        # 无解不启动：不创建任何会话/目录，返回 422 说明原因。
        raise HTTPException(
            status_code=422,
            detail={
                "accepted": False,
                "message": "无满足限制的排法，演练未启动",
                "reasons": result.reasons,
            },
        )

    session_id = uuid.uuid4().hex[:12]
    session = Session(
        session_id=session_id,
        plan=result,
        data_dir=DATA_DIR,
        num_batches=len(req.durations),
    )
    with REGISTRY_LOCK:
        SESSIONS[session_id] = session
    return {
        "accepted": True,
        "session_id": session_id,
        "checkpoints_used": result.checkpoints_used,
        "reruns_total": result.reruns_total,
        "makespan": result.makespan,
        "segments": [s.as_dict() for s in result.segments],
    }


@app.get("/sessions")
async def list_sessions() -> dict:
    # 直接读各会话已发布的不可变快照，不碰会话锁，写盘卡住也立即返回。
    with REGISTRY_LOCK:
        items = sorted(SESSIONS.items())
    sessions = []
    for sid, session in items:
        snap = session.status_snapshot()
        item = {k: v for k, v in snap.items() if k != "session_id"}
        sessions.append({"session_id": sid, **item})
    return {"sessions": sessions}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = _require_session(session_id)
    return session.status_snapshot()


@app.post("/sessions/{session_id}/advance")
async def advance_clock(session_id: str, seconds: int) -> dict:
    session = _require_session(session_id)
    if seconds <= 0:
        raise HTTPException(status_code=400, detail="seconds 必须为正整数")

    # 投递到该会话专属的单线程执行器：同会话的推进严格逐个执行，
    # 后一个一定在前一个实际停下后才起算；等待中的请求只 await future，
    # 不占用 anyio/uvicorn 的公共工作线程，因此不会饿死查询和其他会话。
    future = session.submit_advance(seconds)
    try:
        return await asyncio.wrap_future(future)
    except CheckpointWriteFailed as exc:
        detail = exc.error_info()
        # 与正常 /advance 同构的本次已完成部分（clock/advanced_events）。
        if exc.partial_response:
            detail.update(exc.partial_response)
        # 再附完整状态快照，方便调用方核对后原样重试。
        detail["progress"] = session.status_snapshot()
        raise HTTPException(status_code=500, detail=detail) from exc


@app.get("/sessions/{session_id}/results")
async def get_results(session_id: str) -> dict:
    session = _require_session(session_id)
    # 复制一份再补 session_id，避免改写已发布的共享快照。
    out = dict(session.results_snapshot())
    out["session_id"] = session_id
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "numpy": np.__version__}


def _require_session(session_id: str) -> Session:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="会话不存在或未受理")
    return SESSIONS[session_id]
