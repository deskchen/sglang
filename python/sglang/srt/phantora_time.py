# SPDX-License-Identifier: Apache-2.0
"""Phantora cross-process simulated-time propagation (SGLang port).

SGLang runs as several OS processes: the tokenizer manager (frontend, in the
server/Engine process), one scheduler process per TP/PP rank (which owns the
model runner), and a detokenizer process. Under Phantora's simulator each
process keeps its *own* virtual clock, so the frontend (which only waits in
real time) and the scheduler (which advances simulated time through modeled
GPU work) drift apart.

We reconcile them causally (Lamport-clock style): every message across a
process boundary carries the sender's simulated time (``stamp()``), and the
receiver advances its own clock forward to it (``adopt()`` == a max).
Stamping both directions makes the result the critical path -- non-overlapped
work sums, overlapped work is hidden.

Carrier edges in SGLang (all patched to call into this module):
  - every zmq hop, via the ``sock_send``/``sock_recv`` helper quartet in
    ``managers/io_struct.py`` (tokenizer->scheduler, scheduler->detokenizer,
    detokenizer->tokenizer, scheduler->tokenizer direct, RPC);
  - the TP/PP object broadcasts over the gloo CPU group
    (``utils/common.py`` ``broadcast_pyobj``/``point_to_point_pyobj``) --
    the NCCL forward-pass collectives are already synchronized by the
    simulator itself;
  - the scheduler-ready handshake over multiprocessing.Pipe
    (``managers/scheduler.py`` ``get_init_info`` ->
    ``entrypoints/engine.py`` ``_wait_for_scheduler_ready``).

Entirely a no-op unless running under the Phantora simulator (detected via
the ``PHANTORA_SOCKET_PREFIX`` env the simulator sets), so normal SGLang is
unaffected.
"""

import ctypes
import os

_enabled = False
_stamp = None
_adopt = None


def _init() -> None:
    global _enabled, _stamp, _adopt
    if os.environ.get("PHANTORA_SOCKET_PREFIX") is None:
        return  # not running under the Phantora simulator
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.get_time_double.restype = ctypes.c_double
        lib.phantora_adopt_time_double.argtypes = [ctypes.c_double]
        lib.phantora_adopt_time_double.restype = None
        _stamp = lib.get_time_double
        _adopt = lib.phantora_adopt_time_double
        _enabled = True
    except (OSError, AttributeError):
        pass  # stub not loadable / missing symbol -> stay disabled


_init()


def enabled() -> bool:
    return _enabled


def stamp() -> float:
    """This process's current simulated time in seconds (0.0 if disabled)."""
    return _stamp() if _enabled else 0.0


def adopt(t: float) -> None:
    """Advance this process's virtual clock FORWARD to ``t`` seconds (a max).

    No-op if disabled or ``t`` is unset/0.0. Never moves the clock backward.
    """
    if _enabled and t and t > 0.0:
        _adopt(t)


# ---------------------------------------------------------------------------
# Open-loop arrival schedule.
#
# Virtual time only advances through WORK (host accrual, GPU completions), so
# the EXOGENOUS inter-arrival spacing of an open-loop client (think time, a
# trace's arrival cadence) is invisible to the virtual clock — without help,
# requests an hour apart look back-to-back. Open-loop clients pace submissions
# with REAL sleeps (time.sleep; asyncio.sleep is sim-clocked under the
# bootstrap and would deadlock while idle), and that real cadence maps 1:1
# into simulated time.
#
# We reconstruct it as a SCHEDULE, not as a sum of per-idle gaps. Request i
# arrives at sim slot
#     origin_sim + (real_now - origin_real)
# and the clock is advanced to that slot as a FORWARD-ONLY max. So when
# processing already carried the clock past the slot (the engine kept up, or a
# request queued behind others), nothing is pushed — processing OVERLAPS the
# arrival spacing, and each cycle charges max(processing, spacing): the
# correct open-loop queueing model. (Anchoring at busy->idle transitions and
# adopting the wall gap since drain instead double-charges by
# processing - sim_execution_wall per cycle — measured +56.8% on ShareGPT
# open-loop c8 with the vLLM engine before the schedule form replaced it.)
#
# Gated behind arm_open_loop(): a closed-loop client's pacing is ENDOGENOUS
# (driven by simulated completions), and real elapsed time then reflects only
# the simulator's own execution speed — never an arrival gap — so it must not
# be adopted. Off open-loop, adopt_idle_gap() is a no-op. A custom open-loop
# client must call arm_open_loop() once before submitting.

_sched_origin: "tuple[float, float] | None" = None
_open_loop: bool = False


def _real_now() -> float:
    import time

    return getattr(time, "_real_time", time.time)()


def arm_open_loop() -> None:
    """Enable open-loop arrival-schedule adoption and set the schedule origin
    to NOW. Call once, right before submitting open-loop (real-paced) traffic;
    a no-op under closed-loop leaves the clock purely work-driven."""
    global _open_loop, _sched_origin
    if _enabled:
        _open_loop = True
        _sched_origin = (_real_now(), _stamp())


def rearm_idle() -> None:
    """Reset the schedule origin to NOW. Called when the engine becomes ready
    (after _wait_for_scheduler_ready): real init time is startup work, not an
    arrival gap, so the origin must not predate it (without the reset the
    first request would adopt the whole init window)."""
    global _sched_origin
    if _enabled and _open_loop:
        _sched_origin = (_real_now(), _stamp())


def adopt_idle_gap(arrival_real: float | None = None) -> None:
    """At an open-loop arrival: advance the virtual clock to this request's
    scheduled sim slot (origin_sim + real time since origin), forward-only.

    ``arrival_real`` is the request's true arrival instant on the raw wall
    clock (``time._real_time``). A paced client MUST pass its intended arrival
    (submit_start + i*gap): the exogenous cadence is what maps into sim time,
    whereas ``_real_now()`` at the call site is delayed by the simulator's own
    wall-execution speed and by client-side concurrency back-pressure — reading
    it directly re-inflates the very gap we are trying to model. No-op unless
    armed for open-loop (arm_open_loop)."""
    global _sched_origin
    if not (_enabled and _open_loop) or _sched_origin is None:
        return
    real0, sim0 = _sched_origin
    now = arrival_real if arrival_real is not None else _real_now()
    slot = sim0 + (now - real0)
    cur = _stamp()
    if slot > cur:
        if os.environ.get("PHANTORA_IDLE_TRACE") == "1":
            print(
                f"[idle-gap] slot={slot:.3f} cur={cur:.3f} " f"adv={slot - cur:.3f}s",
                flush=True,
            )
        _adopt(slot)


def replace_sampled_token_ids(next_token_ids, forward_batch, *, overlap_enabled):
    """Use an explicit workload token trace only under Phantora simulation."""
    if not _enabled:
        return next_token_ids

    info = forward_batch.sampling_info
    params = info.custom_params
    traced = [
        isinstance(param, dict) and "phantora_token_trace" in param
        for param in (params or [])
    ]
    if not any(traced):
        return next_token_ids
    if (
        len(traced) != next_token_ids.numel()
        or not all(traced)
        or not forward_batch.spec_algorithm.is_none()
        or forward_batch.return_logprob
        or any(info.return_sampling_masks or [])
        or info.has_custom_logit_processor
    ):
        raise RuntimeError("invalid Phantora token-trace batch")

    resolved = []
    for param in params:
        req = param.get("__req__")
        trace = param["phantora_token_trace"]
        sampling = getattr(req, "sampling_params", None)
        position = getattr(req, "_phantora_token_trace_position", 0)
        if (
            req is None
            or sampling is None
            or not isinstance(trace, list)
            or not trace
            or not all(type(token_id) is int for token_id in trace)
            or type(req.vocab_size) is not int
            or req.vocab_size <= 0
            or any(token_id < 0 or token_id >= req.vocab_size for token_id in trace)
            or type(position) is not int
            or position < 0
            or position > len(trace)
            or sampling.max_new_tokens != len(trace)
            or not sampling.ignore_eos
            or sampling.stop_token_ids
            or sampling.stop_strs
            or sampling.stop_regex_strs
            or req.grammar is not None
            or req.inflight_middle_chunks > 0
        ):
            raise RuntimeError("invalid or exhausted Phantora token trace")
        terminal_drain = position == len(trace)
        if terminal_drain and (
            not overlap_enabled
            or not forward_batch.forward_mode.is_decode()
            or req.finished()
            or req.is_retracted
            or req.to_finish is not None
            or len(req.output_ids) != len(trace) - 1
            or param.get("__phantora_token_trace_terminal_drain", False)
        ):
            raise RuntimeError("invalid or exhausted Phantora token trace")
        resolved.append(
            (
                param,
                req,
                position,
                trace[-1] if terminal_drain else trace[position],
                terminal_drain,
            )
        )

    for param, req, position, _, terminal_drain in resolved:
        if terminal_drain:
            # Request-local custom params own this flag, so a recycled request-pool
            # slot cannot inherit terminal-drain state from its previous request.
            param["__phantora_token_trace_terminal_drain"] = True
        else:
            req._phantora_token_trace_position = position + 1
        req.skip_radix_cache_insert = True
    return next_token_ids.new_tensor([token_id for _, _, _, token_id, _ in resolved])
