from __future__ import annotations

import multiprocessing as mp
import sys
import os
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from collections.abc import Callable, Sequence
from typing import TypeVar, Any

T = TypeVar("T")
R = TypeVar("R")

# Live progress of long jobs: workers write a short status for the job they run (e.g., the selection round) and the
# main process shows it next to the job's line. No-ops outside a verbose parallel run.
_PROGRESS = None
_CURRENT = None


def report(status: str) -> None:
    if _PROGRESS is not None and _CURRENT is not None:
        try:
            _PROGRESS[_CURRENT] = status
        except Exception:
            pass


def _tracked(fn, index: int, job):
    global _CURRENT
    _CURRENT = index
    try:
        return fn(job)
    finally:
        _CURRENT = None


def _threads_per_worker(max_workers: int, requested: int | None = None) -> int:
    if requested is not None:
        return max(1, int(requested))
    return max(1, (os.cpu_count() or max_workers) // max(1, max_workers))


def configure_threads(n_threads: int) -> None:
    value = str(max(1, int(n_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value
    import torch

    torch.set_num_threads(int(value))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _init_worker(n_threads: int, progress=None) -> None:
    global _PROGRESS
    _PROGRESS = progress
    configure_threads(n_threads)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy("file_system")


def _enable_ansi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _job_label(job: Any) -> str:
    if not isinstance(job, dict):
        return str(job)
    parts = [str(job.get("arm", "(unknown)"))]
    if job.get("select"):
        parts.append("per-item")
    if "prm" in job:
        parts.append("PRM" if job["prm"] else "no PRM")
    c, d = job.get("c"), job.get("d")
    if c is not None and not hasattr(c, "shape"):
        if d is not None and not hasattr(d, "shape"):
            parts.append(f"({c},{d})")
    if job.get("n_basis") is not None:
        parts.append(f"basis={job['n_basis']}")
    if job.get("study"):
        parts.append(f"[{job['study']} {job.get('generator', '')} rep={job.get('seed')}]")
    elif job.get("prediction"):
        parts.append(f"- pred (seed={job.get('seed')})")
    else:
        parts.append(f"- full")
    
    return " ".join(parts)


def _live_stream(verbose: bool):
    """Where the live (redrawn) job display goes: the terminal itself. With stdout piped (e.g., into tee for a log),
    the display is written to the controlling terminal (/dev/tty) so the log stays free of redraws; None when there
    is no terminal (background runs), which falls back to one line per finished job on stdout."""
    if not verbose:
        return None
    if sys.stdout.isatty():
        return sys.stdout
    try:
        return open("/dev/tty", "w", buffering=1)
    except OSError:
        return None


def _summary(total: int, done: int, running: int, queued: int, failed: int) -> str:
    return f"[Jobs] {done}/{total} done | {running} running | {queued} queued | {failed} failed"


def _draw(out, total: int, done: int, failed: int, slots: list[str | None], *, first: bool,
          status: list[str | None] | None = None) -> None:
    running = sum(slot is not None for slot in slots)
    queued = total - done - failed - running
    lines = [_summary(total, done, running, queued, failed)]
    for index, label in enumerate(slots):
        note = status[index] if status is not None and label is not None else None
        lines.append(f"  [{index}] {label or 'idle '}" + (f" | {note}" if note else ""))
    if not first:
        out.write(f"\x1b[{len(lines)}A")
    for line in lines:
        out.write(f"\x1b[2K{line}\n")
    out.flush()


def _unshare(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, torch.nn.Module):
        for module in value.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is not None:
                    parameter.data = parameter.data.clone()
            for name, buffer in list(module._buffers.items()):
                if buffer is not None:
                    module._buffers[name] = buffer.clone()
        return value
    if isinstance(value, dict):
        return {k: _unshare(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unshare(v) for v in value)
    return value


def map_parallel(
    fn: Callable[[T], R],
    jobs: Sequence[T],
    *,
    max_workers: int,
    threads_per_worker: int | None = None,
    verbose: bool = False,
) -> list[R]:
    if not jobs:
        return []

    max_workers = min(max_workers, os.cpu_count() or 1)
    
    workers = max(1, min(int(max_workers), len(jobs)))
    n_threads = _threads_per_worker(workers, threads_per_worker)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy("file_system")
    context = mp.get_context("spawn")
    out = _live_stream(verbose)
    live = out is not None
    manager = context.Manager() if live else None
    progress = manager.dict() if manager is not None else None

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(n_threads, progress),
    ) as pool:
        if not verbose:
            return [_unshare(result) for result in pool.map(fn, jobs)]

        if live:
            _enable_ansi()

        results: list[R | None] = [None] * len(jobs)
        slots: list[str | None] = [None] * workers
        owners: list[int | None] = [None] * workers
        inflight: dict[Any, tuple[int, int]] = {}

        def status() -> list[str | None]:
            if progress is None:
                return [None] * workers
            snapshot = dict(progress)
            return [snapshot.get(owner) if owner is not None else None for owner in owners]
        queued = list(enumerate(jobs))
        done, failed = 0, 0
        first = True

        def fill() -> None:
            for slot, label in enumerate(slots):
                if label is not None or not queued:
                    continue
                index, job = queued.pop(0)
                future = pool.submit(_tracked, fn, index, job)
                slots[slot] = _job_label(job)
                owners[slot] = index
                inflight[future] = (index, slot)

        fill()
        if live:
            _draw(out, len(jobs), done, failed, slots, first=True, status=status())
            first = False
        else:
            print(_summary(len(jobs), done, sum(slot is not None for slot in slots), len(queued), failed), flush=True)

        try:
            while inflight:
                finished, _ = wait(inflight, timeout=2.0 if live else None, return_when=FIRST_COMPLETED)
                if not finished:
                    _draw(out, len(jobs), done, failed, slots, first=first, status=status())
                    continue
                for future in finished:
                    index, slot = inflight.pop(future)
                    slots[slot] = None
                    owners[slot] = None
                    if progress is not None:
                        progress.pop(index, None)
                    try:
                        results[index] = _unshare(future.result())
                        done += 1
                    except Exception:
                        failed += 1
                        if live:
                            _draw(out, len(jobs), done, failed, slots, first=first, status=status())
                            out.write("\n")
                            out.flush()
                        raise
                fill()
                if live:
                    _draw(out, len(jobs), done, failed, slots, first=first, status=status())
                    first = False
                else:
                    running = sum(slot is not None for slot in slots)
                    print(_summary(len(jobs), done, running, len(queued), failed), flush=True)
        except BaseException:
            if live and not first:
                out.write("\n")
                out.flush()
            raise
        finally:
            if manager is not None:
                manager.shutdown()
            if live and out is not sys.stdout:
                print(_summary(len(jobs), done, 0, len(queued), failed), flush=True)   # final state into the log
                out.close()

    return results