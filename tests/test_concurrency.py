"""并发 /advance、存档失败接续、写盘阻塞隔离的回归测试。

覆盖题目要求：

* 同一会话并发推进必须串行：后一个从前一个实际停下的 clock 起算，
  每个排程时段恰好完成一次，不重复写检查点、不漏评测；
* 存档写失败返回 500 且保留本次已完成时段；之后按排程原编号补写，
  评测读原定文件，最终权重仍与不间断训练一致；
* 一个会话写盘卡住时，即使等待的推进请求多于 anyio 默认 40 个工作线程，
  GET 会话/结果仍立即返回已完成进度，其他会话照常推进。

并发用例直接起真实 uvicorn（后台线程 + 随机端口），经真实 HTTP 发压，
避免 TestClient 每请求独立事件循环带来的语义偏差。
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
import uvicorn

from app import ml
from app.main import SESSIONS, app
from app.scheduler import solve
from app.simulator import Session

PLAN_WITH_EVAL = {
    "durations": [2] * 5,
    "evaluations": [
        {"eval_id": 1, "start": 10, "end": 12, "min_checkpoint_batches": 1}
    ],
    "max_checkpoints": 2,
    "max_rerun_batches": 10,
}

TWO_EVAL_PLAN = {
    "durations": [3] * 8,
    "evaluations": [
        {"eval_id": 1, "start": 12, "end": 14, "min_checkpoint_batches": 2},
        {"eval_id": 2, "start": 30, "end": 32, "min_checkpoint_batches": 3},
    ],
    "max_checkpoints": 1,
    "max_rerun_batches": 20,
}


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def http_call(method: str, url: str, payload=None, timeout: float = 30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"_raw": raw}


@pytest.fixture
def server(tmp_path, monkeypatch):
    """真实 uvicorn 进程内线程；DATA_DIR 与 SESSIONS 每用例隔离。"""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    monkeypatch.setattr("app.main.DATA_DIR", data_dir)
    SESSIONS.clear()

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="error")
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, name="uvicorn-test", daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            code, body = http_call("GET", base + "/health", timeout=1)
            if code == 200:
                break
        except Exception:
            time.sleep(0.05)
    else:
        raise RuntimeError("测试 uvicorn 未能在 20s 内就绪")

    yield base, data_dir

    uv.should_exit = True
    thread.join(timeout=10)
    # 收掉本用例各会话的单线程执行器，避免空闲线程跨用例堆积。
    for session in SESSIONS.values():
        session.shutdown_executor()
    SESSIONS.clear()


def _create(base: str, plan: dict) -> str:
    code, body = http_call("POST", base + "/plans", plan)
    assert code == 201, body
    return body["session_id"]


def _advance(base: str, sid: str, seconds: int, timeout: float = 60):
    return http_call(
        "POST", f"{base}/sessions/{sid}/advance?seconds={seconds}",
        timeout=timeout,
    )


def _planned_keys(plan_body_segments):
    return [(s["kind"], s["start"], s["end"]) for s in plan_body_segments]


# ---------------------------------------------------------------------------
# 1. 原始 Session 级别的并发安全（不经过 HTTP）
# ---------------------------------------------------------------------------
def test_raw_session_concurrent_advance_is_serial(tmp_path):
    res = solve([4, 4], [], 0, 0)
    session = Session("raw", res, str(tmp_path), 2)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        return session.advance(2)

    with ThreadPoolExecutor(8) as pool:
        outs = list(pool.map(lambda _: worker(), range(8)))

    # 8 个 seconds=2 串行累计：0->2(空) ->4(批0) ->6(空) ->8(批1完成)，
    # 之后全部幂等 finished。两个训练时段各只执行一次。
    assert session.finished is True
    assert session.clock == 8
    assert [e.segment.batch_index for e in session.events] == [0, 1]
    # 所有响应都是合法 200 形态且没有任何重复事件。
    total_events = sum(len(o["advanced_events"]) for o in outs)
    assert total_events == 2


# ---------------------------------------------------------------------------
# 2. HTTP 并发大推进：每个时段恰好一次，检查点不重复，评测不漏
# ---------------------------------------------------------------------------
def test_http_concurrent_big_advances_run_each_segment_once(server):
    base, data_dir = server
    code, plan_body = http_call("POST", base + "/plans", TWO_EVAL_PLAN)
    assert code == 201, plan_body
    sid = plan_body["session_id"]
    n_segments = len(plan_body["segments"])

    n_callers = 10
    barrier = threading.Barrier(n_callers)

    def worker():
        barrier.wait()
        return _advance(base, sid, 1000)

    with ThreadPoolExecutor(n_callers) as pool:
        responses = list(pool.map(lambda _: worker(), range(n_callers)))

    assert all(code == 200 for code, _ in responses), responses

    code, status = http_call("GET", f"{base}/sessions/{sid}")
    assert code == 200
    actual_keys = [
        (e["kind"], e["start"], e["end"])
        for e in status["executed_segments"]
    ]
    # 执行顺序和数量与排程完全一致 => 无重复、无跳过（评测不会漏跑）。
    assert actual_keys == _planned_keys(plan_body["segments"])
    assert len(status["executed_segments"]) == n_segments
    assert sum(1 for e in status["executed_segments"]
               if e["kind"] == "evaluation") == 2
    assert status["reruns_executed"] == plan_body["reruns_total"]
    assert status["checkpoints_written"] == plan_body["checkpoints_used"] == 1

    # 所有并发响应里出现的事件总和也恰好是排程时段一次。
    response_keys = [
        (e["kind"], e["start"], e["end"])
        for _, body in responses
        for e in body["advanced_events"]
    ]
    assert sorted(response_keys) == sorted(_planned_keys(plan_body["segments"]))
    assert any(body["finished"] for _, body in responses)

    # 磁盘上只有原定的 checkpoint_001.npz，没有并发撞出来的 002/重复件。
    files = sorted(
        f for f in os.listdir(os.path.join(data_dir, sid))
        if f.endswith(".npz")
    )
    assert files == ["checkpoint_001.npz"]

    code, results = http_call("GET", f"{base}/sessions/{sid}/results")
    assert code == 200
    assert results["finished"] is True
    assert results["consistent_with_uninterrupted"] is True
    assert results["max_abs_diff"] == 0.0


# ---------------------------------------------------------------------------
# 3. HTTP 并发小推进：后一个从前一个实际 clock 累计起算
# ---------------------------------------------------------------------------
def test_http_concurrent_small_advances_accumulate(server):
    base, _ = server
    code, body = http_call("POST", base + "/plans",
                           {"durations": [4, 4], "evaluations": [],
                            "max_checkpoints": 0, "max_rerun_batches": 0})
    sid = body["session_id"]

    n_callers = 8
    barrier = threading.Barrier(n_callers)

    def worker():
        barrier.wait()
        return _advance(base, sid, 2)

    with ThreadPoolExecutor(n_callers) as pool:
        responses = list(pool.map(lambda _: worker(), range(n_callers)))

    assert all(code == 200 for code, _ in responses), responses
    code, status = http_call("GET", f"{base}/sessions/{sid}")
    assert status["finished"] is True
    assert status["clock"] == 8
    trains = [e for e in status["executed_segments"] if e["kind"] == "train"]
    assert [e["batch_index"] for e in trains] == [0, 1]
    total_events = sum(len(b["advanced_events"]) for _, b in responses)
    assert total_events == 2


# ---------------------------------------------------------------------------
# 3b. 复刻 /docs 双击：两次相同的 10 秒推进，第一次卡在存档，
#     第二次排队；两者必须切分同一时间线，不重复检查点、不漏评测。
# ---------------------------------------------------------------------------
def test_double_advance_while_archiving_partitions_timeline(
    server, monkeypatch
):
    base, _ = server
    sid = _create(base, PLAN_WITH_EVAL)
    original_save = ml.save_checkpoint

    def slow_save(directory, checkpoint_id, batches_trained, weights):
        time.sleep(0.4)  # “前一次还在存档”
        return original_save(directory, checkpoint_id, batches_trained, weights)

    monkeypatch.setattr(ml, "save_checkpoint", slow_save)

    barrier = threading.Barrier(2)

    def twice():
        barrier.wait()
        return _advance(base, sid, 10, timeout=30)

    with ThreadPoolExecutor(2) as pool:
        r1, r2 = list(pool.map(lambda _: twice(), range(2)))

    assert r1[0] == r2[0] == 200, (r1, r2)
    e1 = r1[1]["advanced_events"]
    e2 = r2[1]["advanced_events"]
    # 一个调用拿到 4 训练 + 检查点（停在 clock=10），另一个从 10 起算
    # 拿到评测 + 最后训练并完成；顺序不保证，故事件数集合为 {5, 2}。
    assert sorted([len(e1), len(e2)]) == [2, 5]
    assert r1[1]["finished"] != r2[1]["finished"]
    union = e1 + e2
    assert sum(1 for e in union if e["kind"] == "checkpoint") == 1
    assert sum(1 for e in union if e["kind"] == "evaluation") == 1

    code, status = http_call("GET", f"{base}/sessions/{sid}")
    keys = [(e["kind"], e["start"], e["end"])
            for e in status["executed_segments"]]
    # 直接与内存中的排程比对（避免再建一个空计划）。
    planned = [(s.kind, s.start, s.end)
               for s in SESSIONS[sid].plan.segments]
    assert keys == planned
    code, results = http_call("GET", f"{base}/sessions/{sid}/results")
    assert results["consistent_with_uninterrupted"] is True


# ---------------------------------------------------------------------------
# 4. 存档写失败 -> 500 + 保留进度 -> 原编号补写 -> 评测/一致性正常
# ---------------------------------------------------------------------------
def test_checkpoint_write_failure_preserves_and_retries_original_number(
    server, monkeypatch
):
    base, data_dir = server
    sid = _create(base, PLAN_WITH_EVAL)

    original_save = ml.save_checkpoint
    state = {"attempts": 0}

    def flaky_save(directory, checkpoint_id, batches_trained, weights):
        state["attempts"] += 1
        if state["attempts"] <= 2:
            raise OSError("ENOSPC: injected disk failure")
        return original_save(directory, checkpoint_id, batches_trained, weights)

    monkeypatch.setattr(ml, "save_checkpoint", flaky_save)

    # 第一次大推进：4 个训练时段完成后，checkpoint 写失败 -> 500。
    code, body = _advance(base, sid, 100)
    assert code == 500, body
    detail = body["detail"]
    assert detail["error"] == "checkpoint_write_failed"
    assert detail["checkpoint_no"] == 1
    progress = detail["progress"]
    assert progress["clock"] == 8
    assert [e["kind"] for e in progress["executed_segments"]] == ["train"] * 4
    assert progress["last_error"]["checkpoint_no"] == 1

    # GET 会话查询同样返回保留下来的已完成进度。
    code, status = http_call("GET", f"{base}/sessions/{sid}")
    assert code == 200
    assert status["clock"] == 8 and status["finished"] is False
    assert len(status["executed_segments"]) == 4
    assert status["last_error"]["checkpoint_no"] == 1

    # 失败没有留下原定最终文件（原子写：临时文件已清理）。
    session_dir = os.path.join(data_dir, sid)
    assert not os.path.exists(os.path.join(session_dir, "checkpoint_001.npz"))
    assert not [f for f in os.listdir(session_dir) if f.startswith(".checkpoint")]

    # 小步推进尚不能跑完 8..10 的检查点时段。
    code, body = _advance(base, sid, 1)
    assert code == 200 and body["clock"] == 9
    assert body["advanced_events"] == []

    # 第二次到达检查点末尾：仍然失败，错误编号不变、训练时段不重复。
    code, body = _advance(base, sid, 1)
    assert code == 500
    assert body["detail"]["checkpoint_no"] == 1
    assert len(body["detail"]["progress"]["executed_segments"]) == 4

    # 再推进 1 秒：第三次尝试成功，按原编号 1 补写。
    code, body = _advance(base, sid, 1)
    assert code == 200
    assert [e["kind"] for e in body["advanced_events"]] == ["checkpoint"]
    assert body["advanced_events"][0]["checkpoint_no"] == 1

    npz_files = sorted(f for f in os.listdir(session_dir) if f.endswith(".npz"))
    assert npz_files == ["checkpoint_001.npz"]
    weights, batches = ml.load_checkpoint(
        os.path.join(session_dir, "checkpoint_001.npz")
    )
    assert batches == 4

    # 继续推进到结束：评测读取原定 checkpoint_001 做推理并回滚续训。
    for _ in range(20):
        code, body = _advance(base, sid, 2)
        assert code == 200, body
        if body["finished"]:
            break
    else:
        pytest.fail("失败补写后未能推进到 finished")

    code, status = http_call("GET", f"{base}/sessions/{sid}")
    kinds = [e["kind"] for e in status["executed_segments"]]
    assert kinds.count("checkpoint") == 1
    assert kinds.count("evaluation") == 1
    eval_event = next(e for e in status["executed_segments"]
                      if e["kind"] == "evaluation")
    assert eval_event["inference"]["used_checkpoint_no"] == 1

    code, results = http_call("GET", f"{base}/sessions/{sid}/results")
    assert results["consistent_with_uninterrupted"] is True
    assert results["max_abs_diff"] == 0.0


# ---------------------------------------------------------------------------
# 4b. 第二个检查点写失败：cp1 保留，cp2 按原编号补写，评测 2 读 cp2
# ---------------------------------------------------------------------------
def test_second_checkpoint_failure_retries_original_number(server, monkeypatch):
    base, data_dir = server
    plan = {
        "durations": [2] * 8,
        "evaluations": [
            {"eval_id": 1, "start": 10, "end": 12,
             "min_checkpoint_batches": 1},
            {"eval_id": 2, "start": 16, "end": 18,
             "min_checkpoint_batches": 1},
        ],
        "max_checkpoints": 2,
        "max_rerun_batches": 0,
    }
    sid = _create(base, plan)
    session_dir = os.path.join(data_dir, sid)
    original_save = ml.save_checkpoint

    def fail_second(directory, checkpoint_id, batches_trained, weights):
        if checkpoint_id == 2:
            raise OSError("EROFS: injected failure on checkpoint 2")
        return original_save(directory, checkpoint_id, batches_trained, weights)

    monkeypatch.setattr(ml, "save_checkpoint", fail_second)

    code, body = _advance(base, sid, 100)
    assert code == 500
    detail = body["detail"]
    assert detail["checkpoint_no"] == 2
    kinds = [e["kind"] for e in detail["progress"]["executed_segments"]]
    # cp1 与 eval1 已完成并保留，失败的 cp2 不在已完成列表里。
    assert kinds.count("checkpoint") == 1
    assert kinds.count("evaluation") == 1
    assert os.path.exists(os.path.join(session_dir, "checkpoint_001.npz"))
    assert not os.path.exists(os.path.join(session_dir, "checkpoint_002.npz"))

    # 恢复写盘后原样重试：cp2 以编号 2 补写，全程只多出这一个文件。
    monkeypatch.setattr(ml, "save_checkpoint", original_save)
    for _ in range(20):
        code, body = _advance(base, sid, 2)
        assert code == 200, body
        if body["finished"]:
            break
    else:
        pytest.fail("cp2 补写后未能完成")

    files = sorted(f for f in os.listdir(session_dir) if f.endswith(".npz"))
    assert files == ["checkpoint_001.npz", "checkpoint_002.npz"]

    code, status = http_call("GET", f"{base}/sessions/{sid}")
    evals = [e for e in status["executed_segments"] if e["kind"] == "evaluation"]
    assert [e["inference"]["used_checkpoint_no"] for e in evals] == [1, 2]
    code, results = http_call("GET", f"{base}/sessions/{sid}/results")
    assert results["consistent_with_uninterrupted"] is True


# ---------------------------------------------------------------------------
# 5. 一个会话写盘卡住：查询秒回、其他会话照常、排队请求数 > 工作线程数
# ---------------------------------------------------------------------------
def test_stuck_write_isolates_queries_and_other_sessions(
    server, monkeypatch
):
    base, _ = server
    sid_a = _create(base, PLAN_WITH_EVAL)
    # B 也带检查点+评测，证明其他会话连存档都能正常写。
    sid_b = _create(base, PLAN_WITH_EVAL)

    entered = threading.Event()
    release = threading.Event()
    save_calls: list[int] = []
    original_save = ml.save_checkpoint

    def stuck_only_a(directory, checkpoint_id, batches_trained, weights):
        if sid_a in directory:
            save_calls.append(checkpoint_id)
            entered.set()
            # 测试主线程会在断言结束后 set release；20s 兜底防挂死。
            assert release.wait(timeout=20), "等待 release 超时"
        return original_save(directory, checkpoint_id, batches_trained, weights)

    monkeypatch.setattr(ml, "save_checkpoint", stuck_only_a)

    # 明显多于 anyio 默认 40 个公共工作线程。
    n_waiters = 48
    barrier = threading.Barrier(n_waiters)
    pool = ThreadPoolExecutor(max_workers=n_waiters)
    futures = []
    try:
        def waiter():
            barrier.wait()
            return _advance(base, sid_a, 1000, timeout=60)

        futures = [pool.submit(waiter) for _ in range(n_waiters)]

        assert entered.wait(timeout=10), "卡住的存档没有开始"
        # 给执行线程一点时间发布检查点之前的进度快照。
        deadline = time.time() + 5
        progress = None
        while time.time() < deadline:
            code, progress = http_call("GET", f"{base}/sessions/{sid_a}",
                                       timeout=2)
            if code == 200 and progress["executed_segments"]:
                break
            time.sleep(0.05)
        assert progress is not None
        assert progress["finished"] is False
        assert progress["executed_segments"], "应已发布检查点之前的完成时段"
        assert all(e["kind"] != "checkpoint"
                   for e in progress["executed_segments"])

        # GET 会话必须立即返回，不能被卡住的写盘/会话锁挡住。
        t0 = time.monotonic()
        code, status = http_call("GET", f"{base}/sessions/{sid_a}", timeout=5)
        get_dt = time.monotonic() - t0
        assert code == 200 and get_dt < 5.0

        # GET results 同样立即返回已完成（未 finish）进度。
        t0 = time.monotonic()
        code, results = http_call("GET", f"{base}/sessions/{sid_a}/results",
                                  timeout=5)
        res_dt = time.monotonic() - t0
        assert code == 200 and results["finished"] is False
        assert res_dt < 5.0

        # GET /sessions 列表也不能被拖死。
        code, listing = http_call("GET", base + "/sessions", timeout=5)
        assert code == 200
        assert any(s["session_id"] == sid_a for s in listing["sessions"])

        # 其他会话照常一次性推进到完成（含自己的检查点写入，不被 A 阻塞）。
        t0 = time.monotonic()
        code, body_b = _advance(base, sid_b, 1000, timeout=15)
        b_dt = time.monotonic() - t0
        assert code == 200, body_b
        assert body_b["finished"] is True
        assert b_dt < 15.0
        code, res_b = http_call("GET", f"{base}/sessions/{sid_b}/results",
                                timeout=5)
        assert res_b["consistent_with_uninterrupted"] is True
    finally:
        # 无论上面断言是否失败都放行卡住的写盘，并收完所有请求，避免泄漏。
        release.set()
        responses = []
        for fut in futures:
            try:
                responses.append(fut.result(timeout=30))
            except Exception as exc:  # noqa: BLE001
                responses.append(("EXC", repr(exc)))
        pool.shutdown(wait=True)

    assert responses, "应当收集到并发推进响应"
    bad = [r for r in responses if not (isinstance(r, tuple) and r[0] == 200)]
    assert not bad, bad[:3]
    # 48 个排队推进只触发了一次存档写，其余在 finished 后幂等返回。
    assert save_calls == [1]

    code, status_a = http_call("GET", f"{base}/sessions/{sid_a}")
    assert status_a["finished"] is True
    kinds = [e["kind"] for e in status_a["executed_segments"]]
    assert kinds.count("checkpoint") == 1
    assert kinds.count("evaluation") == 1
    assert len(status_a["executed_segments"]) == len(
        SESSIONS[sid_a].plan.segments
    )
    code, res_a = http_call("GET", f"{base}/sessions/{sid_a}/results")
    assert res_a["consistent_with_uninterrupted"] is True
