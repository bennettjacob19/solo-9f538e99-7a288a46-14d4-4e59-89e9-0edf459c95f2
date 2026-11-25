"""模拟时钟驱动的演练会话。

排程由 :mod:`app.scheduler` 预先求得并保证可行；本模块负责按模拟时钟
重放排程，并在对应时段真正执行 NumPy 训练、检查点落盘与推理：

* 训练时段调用 :func:`app.ml.train_one_batch`，重跑批次使用相同数据与
  相同起点权重，更新结果与首跑逐元素一致；
* 检查点时段把当前权重写入本地 .npz 文件（模拟独占卡 2 秒）；
* 评测时段加载最近检查点权重，在固定评测集上推理，随后把训练权重
  回滚到该检查点（从同一检查点续训）。
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import ml
from .scheduler import PlanSegment, SolveResult


class CheckpointWriteFailed(RuntimeError):
    """检查点时段写盘失败。

    此前本次 ``advance`` 已跑完的时段保留；失败的检查点时段仍停在原地，
    之后再次推进覆盖该时段时，用排程原定的检查点编号补写。
    """

    def __init__(
        self,
        *,
        session_id: str,
        segment: PlanSegment,
        intended_path: str,
        cause: BaseException,
    ) -> None:
        self.session_id = session_id
        self.segment = segment
        self.checkpoint_no = segment.checkpoint_no
        self.intended_path = intended_path
        self.cause = cause
        self.partial_response: Optional[dict] = None
        super().__init__(
            f"会话 {session_id} 检查点 {segment.checkpoint_no:03d} "
            f"写盘失败（{type(cause).__name__}: {cause}）；"
            f"已完成时段保留，可再次 /advance 按原编号补写"
        )

    def error_info(self) -> dict:
        return {
            "error": "checkpoint_write_failed",
            "message": str(self),
            "session_id": self.session_id,
            "checkpoint_no": self.checkpoint_no,
            "checkpoint_path": self.intended_path,
            "cause": f"{type(self.cause).__name__}: {self.cause}",
        }


@dataclass
class ExecutedEvent:
    segment: PlanSegment
    inference: Optional[dict] = None
    checkpoint_path: Optional[str] = None

    def as_dict(self) -> dict:
        out = self.segment.as_dict()
        if self.inference is not None:
            out["inference"] = self.inference
        if self.checkpoint_path is not None:
            out["checkpoint_path"] = self.checkpoint_path
        return out


@dataclass
class Session:
    session_id: str
    plan: SolveResult
    data_dir: str
    num_batches: int

    clock: int = 0
    cursor: int = 0
    current_weights: np.ndarray = field(default_factory=ml.initial_weights)
    # 已首跑完成的最大批次数（用于判定某段是否为重跑）。
    peak_batches: int = 0
    # 当前已完成（含重跑）到的批次序号。
    trained_cursor: int = 0
    reruns_executed: int = 0
    checkpoints_written: int = 0
    events: list[ExecutedEvent] = field(default_factory=list)
    # checkpoint_no -> Checkpoint 对象。
    checkpoint_store: dict = field(default_factory=dict)
    finished: bool = False

    def __post_init__(self) -> None:
        self.ckpt_dir = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.reference_weights = ml.reference_uninterrupted(self.num_batches)
        # 同一会话的推进在“会话锁 + 单线程执行器”上串行：
        # * 直接多线程调用 advance() 时由 _advance_lock 保证安全；
        # * HTTP 层把请求投递到 _executor（max_workers=1），排队请求只是
        #   队列里的 future，不占用 FastAPI/anyio 的公共工作线程。
        self._advance_lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_init_lock = threading.Lock()
        # 最近一次存档失败的结构化信息（成功补写后清除）。
        self.last_error: Optional[dict] = None
        # 不可变进度快照：写线程每完成一个时段整体替换一次引用，
        # GET 查询直接读快照，绝不阻塞在写盘/会话锁上。
        self._status_snapshot: dict = self.status()
        self._results_snapshot: dict = self.results()

    # ------------------------------------------------------------------
    # 串行执行与进度快照（供 HTTP 层使用）
    # ------------------------------------------------------------------
    def submit_advance(self, seconds: int):
        """把一次推进投递到本会话专属的单线程执行器，返回 concurrent Future。

        同一会话的所有推进严格按投递顺序逐个执行；调用方可以在事件循环里
        ``await asyncio.wrap_future(...)``，等待期间不占公共工作线程。
        """
        if self._executor is None:
            with self._executor_init_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix=f"advance-{self.session_id[:8]}",
                    )
        return self._executor.submit(self.advance, seconds)

    def shutdown_executor(self) -> None:
        """测试/收尾用：关闭会话专属执行器（不等待卡住任务之外的队列）。"""
        with self._executor_init_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

    def status_snapshot(self) -> dict:
        return self._status_snapshot

    def results_snapshot(self) -> dict:
        return self._results_snapshot

    def _publish_snapshots(self) -> None:
        # 每次都新建 dict/list 图，发布后不再原地修改，读侧无需加锁。
        self._status_snapshot = self.status()
        self._results_snapshot = self.results()

    # ------------------------------------------------------------------
    # 时钟推进
    # ------------------------------------------------------------------
    def advance(self, seconds: int) -> dict:
        if seconds <= 0:
            raise ValueError("推进时长必须为正整数秒")
        # 同一会话的并发直接调用也在此串行：后一个调用进入时读到的
        # self.clock / self.cursor 一定是前一个调用实际停下后的状态。
        with self._advance_lock:
            return self._advance_locked(seconds)

    def _advance_locked(self, seconds: int) -> dict:
        if self.finished:
            out = {
                "clock": self.clock,
                "advanced_events": [],
                "finished": True,
            }
            self._publish_snapshots()
            return out

        # target 必须在锁内、从前一次调用实际提交的 clock 起算，
        # 两个连发请求因此不会再基于同一 clock 重复执行同一批时段。
        target = self.clock + seconds
        produced: list[ExecutedEvent] = []

        try:
            while self.cursor < len(self.plan.segments):
                seg = self.plan.segments[self.cursor]
                if seg.start >= target or seg.end > target:
                    # 下一时段尚未开始，或时段不可拆、当前跑不完：
                    # 不产生任何副作用，时钟先推进到请求时刻。
                    break
                # _execute 成功后才提交 cursor/事件；检查点写失败时
                # cursor 仍停在该时段，后续推进可按原编号补写。
                event = self._execute(seg)
                produced.append(event)
                self.events.append(event)
                self.cursor += 1
                self.clock = seg.end
                # 每跑完一个时段立即发布进度：即使下一次写盘卡住，
                # GET 也能读到截至目前已完成的全部时段。
                self._publish_snapshots()

            if self.cursor >= len(self.plan.segments):
                self.finished = True
                self.clock = self.plan.makespan
            else:
                # 未完成：时钟走到请求时刻，但不超过排程总时长。
                self.clock = min(target, self.plan.makespan)
        except CheckpointWriteFailed as exc:
            # 保留本次此前已跑完并提交的时段；记录错误并发布快照，
            # 然后把错误抛给 HTTP 层返回 500。失败的检查点时段未提交。
            self.last_error = exc.error_info()
            # 与正常 /advance 响应同构的“部分结果”，方便调用方直接核对。
            exc.partial_response = {
                "clock": self.clock,
                "advanced_events": [e.as_dict() for e in produced],
                "finished": False,
            }
            self._publish_snapshots()
            raise

        self._publish_snapshots()
        return {
            "clock": self.clock,
            "advanced_events": [e.as_dict() for e in produced],
            "finished": self.finished,
        }

    # ------------------------------------------------------------------
    # 各时段的实际副作用
    # ------------------------------------------------------------------
    def _execute(self, seg: PlanSegment) -> ExecutedEvent:
        if seg.kind == "train":
            assert seg.batch_index == self.trained_cursor, (
                f"排程批次序列断裂：期望 {self.trained_cursor}，"
                f"实际 {seg.batch_index}"
            )
            self.current_weights = ml.train_one_batch(
                self.current_weights, seg.batch_index
            )
            self.trained_cursor += 1
            if seg.rerun:
                self.reruns_executed += 1
            else:
                self.peak_batches = max(self.peak_batches, self.trained_cursor)
            return ExecutedEvent(seg)

        if seg.kind == "checkpoint":
            # 一律使用排程原定的检查点编号（1 基，对应 checkpoint_NNN.npz），
            # 而不是“写前自增的计数器”：写失败后重试仍是同一编号、同一路径，
            # 评测段里引用的 seg.checkpoint_no 必然能读到补写的文件。
            cp_no = seg.checkpoint_no
            intended_path = ml.checkpoint_path(self.ckpt_dir, cp_no)
            try:
                ckpt = ml.save_checkpoint(
                    self.ckpt_dir,
                    checkpoint_id=cp_no,
                    batches_trained=self.trained_cursor,
                    weights=self.current_weights,
                )
            except Exception as exc:  # OSError/ENOSPC/注入故障等
                raise CheckpointWriteFailed(
                    session_id=self.session_id,
                    segment=seg,
                    intended_path=intended_path,
                    cause=exc,
                ) from exc
            self.checkpoint_store[cp_no] = ckpt
            # 编号由排程固定为 1..K，故成功落盘的存储键数即已写数量；
            # 失败的尝试不会预先自增，杜绝 checkpoint_002 之类的幽灵编号。
            self.checkpoints_written = len(self.checkpoint_store)
            self.last_error = None
            return ExecutedEvent(seg, checkpoint_path=ckpt.path)

        if seg.kind == "evaluation":
            assert seg.checkpoint_no in self.checkpoint_store
            ckpt = self.checkpoint_store[seg.checkpoint_no]
            # 用最近写完的检查点“实际推理”：从磁盘加载后再推理。
            loaded_weights, loaded_batches = ml.load_checkpoint(ckpt.path)
            assert loaded_batches == ckpt.batches_trained
            inference = ml.run_inference(loaded_weights)
            inference["used_checkpoint_no"] = seg.checkpoint_no
            inference["used_checkpoint_batches"] = ckpt.batches_trained
            event = ExecutedEvent(seg, inference=inference)
            # 评测结束后从同一检查点续训：权重与进度回滚。
            rolled_back = self.trained_cursor - ckpt.batches_trained
            assert rolled_back >= 0
            self.current_weights = loaded_weights.copy()
            self.trained_cursor = ckpt.batches_trained
            return event

        if seg.kind == "idle":
            return ExecutedEvent(seg)

        raise ValueError(f"未知时段类型: {seg.kind}")

    # ------------------------------------------------------------------
    # 状态与结果
    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "clock": self.clock,
            "finished": self.finished,
            "trained_cursor": self.trained_cursor,
            "peak_batches_trained": self.peak_batches,
            "reruns_executed": self.reruns_executed,
            "checkpoints_written": self.checkpoints_written,
            "limits": {
                "checkpoints_planned": self.plan.checkpoints_used,
                "reruns_planned": self.plan.reruns_total,
                "makespan": self.plan.makespan,
            },
            "executed_segments": [e.as_dict() for e in self.events],
            "last_error": self.last_error,
        }

    def results(self) -> dict:
        out = {
            "finished": self.finished,
            "clock": self.clock,
            "reruns_executed": self.reruns_executed,
            "checkpoints_written": self.checkpoints_written,
        }
        if self.finished:
            final_w = self.current_weights
            ref_w = self.reference_weights
            out["final_weights"] = [float(v) for v in final_w]
            out["reference_uninterrupted_weights"] = [
                float(v) for v in ref_w
            ]
            out["max_abs_diff"] = float(np.max(np.abs(final_w - ref_w)))
            out["consistent_with_uninterrupted"] = bool(
                np.array_equal(final_w, ref_w)
            )
        else:
            out["final_weights"] = None
            out["consistent_with_uninterrupted"] = None
        return out
