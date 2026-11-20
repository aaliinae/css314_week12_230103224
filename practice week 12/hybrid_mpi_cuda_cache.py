import os
import sys
import json
import math
import hashlib
import subprocess
import time
import re
import platform
from pathlib import Path
import numpy as np
from mpi4py import MPI
from numba import cuda, float64, njit, uint64

# ---------------------------------------------------------------------
# Constants from the laboratory specification
# ---------------------------------------------------------------------
TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 300
LAMBDA_DAMP = 0.15
DX = 1.0 / 513.0
DY = 1.0 / 513.0
DT = 1.0e-6
SHM_ITERS = 40_000_000
SHM_SLOTS = 64


# ---------------------------------------------------------------------
# TASK 1: identity, MPI Cartesian topology, GPU selection
# ---------------------------------------------------------------------
def get_gpu_uuid():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True
    )
    return result.strip().splitlines()[0].strip()


def detect_cpu_line_size():
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return 64


def cpu_signature():
    model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                model = line.split(":", 1)[-1].strip()
                break
    except OSError:
        model = platform.processor() or "unknown"
    return f"{platform.machine()}::{os.cpu_count()}::{model}"


def derive_student_seed(github_user, cpu_sig, gpu_uuid):
    digest = hashlib.sha256(
        f"{github_user.strip()}::{cpu_sig}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(
            f"Expected exactly 4 MPI ranks (2x2 grid), got {size}"
        )

    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    cpu_sig = cpu_signature()
    cache_line_bytes = detect_cpu_line_size()
    instance_seed = derive_student_seed(github_user, cpu_sig, gpu_uuid)

    cart = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False
    )
    coords = cart.Get_coords(cart.Get_rank())
    nbr_up, nbr_down = cart.Shift(0, 1)
    nbr_left, nbr_right = cart.Shift(1, 1)

    # MPI-3 node-local shared window. Reserve 63 extra bytes so the uint64
    # telemetry area can be explicitly aligned to a 64-byte CPU cache line.
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    allocation_bytes = SHM_SLOTS * np.dtype(np.uint64).itemsize + 63 if node_rank == 0 else 0
    shm_win = MPI.Win.Allocate_shared(allocation_bytes, disp_unit=1, comm=node_comm)
    shared_buf, _ = shm_win.Shared_query(0)
    raw_shared = np.ndarray((SHM_SLOTS * 8 + 63,), dtype=np.uint8, buffer=shared_buf)
    align_offset = (-raw_shared.ctypes.data) % 64
    shm_view = raw_shared[align_offset:align_offset + SHM_SLOTS * 8].view(np.uint64)
    if shm_view.ctypes.data % 64 != 0:
        raise RuntimeError("Shared telemetry buffer is not 64-byte aligned")

    return {
        "comm": comm,
        "cart": cart,
        "rank": rank,
        "size": size,
        "coords": coords,
        "nbr_up": nbr_up,
        "nbr_down": nbr_down,
        "nbr_left": nbr_left,
        "nbr_right": nbr_right,
        "gpu_uuid": gpu_uuid,
        "instance_seed": instance_seed,
        "cpu_signature": cpu_sig,
        "cache_line_bytes": cache_line_bytes,
        "node_comm": node_comm,
        "node_rank": node_rank,
        "shm_win": shm_win,
        "shm_view": shm_view,
        "shm_alignment_offset": int(align_offset),
    }


# ---------------------------------------------------------------------
# TASK 2: rank-local initialization; no global 512x512 field
# ---------------------------------------------------------------------
def initialize_subdomains(ctx):
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2
    row_rank, col_rank = ctx["coords"]

    # Local indices 0..257 correspond to global indices with one ghost cell.
    global_rows = row_rank * ny + (np.arange(ny + 2) - 1)
    global_cols = col_rank * nx + (np.arange(nx + 2) - 1)

    y = ((global_rows + 1.0) * DY)[:, None]
    x = ((global_cols + 1.0) * DX)[None, :]

    kappa = (
        0.16
        + 0.04 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        + 0.04 * np.sin(2.0 * np.pi * (x + y))
    )
    kappa = np.clip(kappa, 0.08, 0.24).astype(np.float64)

    rng = np.random.default_rng(ctx["instance_seed"])
    centers = rng.uniform(0.15, 0.85, size=(3, 2))
    amplitudes = rng.uniform(0.6, 1.0, size=3)

    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    for p in range(3):
        cx, cy = centers[p]
        amp = amplitudes[p]
        u += amp * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.055 ** 2)
        )

    sx, sy = centers[0]
    source = 0.02 * np.exp(
        -((x - sx) ** 2 + (y - sy) ** 2) / (2.0 * 0.035 ** 2)
    )
    source = np.broadcast_to(source, (ny + 2, nx + 2)).copy()
    u = np.asarray(u, dtype=np.float64)

    # Physical boundary values are fixed at zero.
    if ctx["nbr_up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx["nbr_down"] == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
    if ctx["nbr_left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx["nbr_right"] == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0

    return u, kappa, source, ny, nx, centers, amplitudes


# ---------------------------------------------------------------------
# TASK 3: GPU halo pack/unpack kernels
# ---------------------------------------------------------------------
@cuda.jit
def pack_halos_kernel(u, send_top, send_bottom, send_left, send_right,
                      ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        send_top[idx] = u[1, idx + 1]
        send_bottom[idx] = u[ny, idx + 1]
    if idx < ny:
        send_left[idx] = u[idx + 1, 1]
        send_right[idx] = u[idx + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, recv_top, recv_bottom, recv_left, recv_right,
                        ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        u[0, idx + 1] = recv_top[idx]
        u[ny + 1, idx + 1] = recv_bottom[idx]
    if idx < ny:
        u[idx + 1, 0] = recv_left[idx]
        u[idx + 1, nx + 1] = recv_right[idx]


class HaloBuffers:
    """Pinned host staging arrays plus contiguous device halo vectors."""
    def __init__(self, ny, nx):
        self.d_send_top = cuda.device_array(nx, dtype=np.float64)
        self.d_send_bottom = cuda.device_array(nx, dtype=np.float64)
        self.d_send_left = cuda.device_array(ny, dtype=np.float64)
        self.d_send_right = cuda.device_array(ny, dtype=np.float64)

        self.d_recv_top = cuda.device_array(nx, dtype=np.float64)
        self.d_recv_bottom = cuda.device_array(nx, dtype=np.float64)
        self.d_recv_left = cuda.device_array(ny, dtype=np.float64)
        self.d_recv_right = cuda.device_array(ny, dtype=np.float64)

        self.h_send_top = cuda.pinned_array(nx, dtype=np.float64)
        self.h_send_bottom = cuda.pinned_array(nx, dtype=np.float64)
        self.h_send_left = cuda.pinned_array(ny, dtype=np.float64)
        self.h_send_right = cuda.pinned_array(ny, dtype=np.float64)

        self.h_recv_top = cuda.pinned_array(nx, dtype=np.float64)
        self.h_recv_bottom = cuda.pinned_array(nx, dtype=np.float64)
        self.h_recv_left = cuda.pinned_array(ny, dtype=np.float64)
        self.h_recv_right = cuda.pinned_array(ny, dtype=np.float64)

        self.stream_boundary = cuda.stream()
        self.stream_interior = cuda.stream()


def _post_halo_mpi(ctx, buffers, ny, nx):
    """Post tag-matched receives and sends after D2H staging is complete."""
    cart = ctx["cart"]
    reqs = []

    # Tags identify the direction in which the sender transmits.
    # top=10, bottom=11, left=12, right=13.
    if ctx["nbr_up"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(buffers.h_recv_top, source=ctx["nbr_up"], tag=11))
    else:
        buffers.h_recv_top.fill(0.0)

    if ctx["nbr_down"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(buffers.h_recv_bottom, source=ctx["nbr_down"], tag=10))
    else:
        buffers.h_recv_bottom.fill(0.0)

    if ctx["nbr_left"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(buffers.h_recv_left, source=ctx["nbr_left"], tag=13))
    else:
        buffers.h_recv_left.fill(0.0)

    if ctx["nbr_right"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(buffers.h_recv_right, source=ctx["nbr_right"], tag=12))
    else:
        buffers.h_recv_right.fill(0.0)

    if ctx["nbr_up"] != MPI.PROC_NULL:
        reqs.append(cart.Isend(buffers.h_send_top, dest=ctx["nbr_up"], tag=10))
    if ctx["nbr_down"] != MPI.PROC_NULL:
        reqs.append(cart.Isend(buffers.h_send_bottom, dest=ctx["nbr_down"], tag=11))
    if ctx["nbr_left"] != MPI.PROC_NULL:
        reqs.append(cart.Isend(buffers.h_send_left, dest=ctx["nbr_left"], tag=12))
    if ctx["nbr_right"] != MPI.PROC_NULL:
        reqs.append(cart.Isend(buffers.h_send_right, dest=ctx["nbr_right"], tag=13))

    return reqs


def exchange_halos_async(u_d, ctx, buffers, ny, nx):
    """Pack on GPU, stage to pinned host memory, exchange using nonblocking MPI."""
    threads = 256
    blocks = (max(ny, nx) + threads - 1) // threads
    s = buffers.stream_boundary

    pack_halos_kernel[blocks, threads, s](
        u_d,
        buffers.d_send_top, buffers.d_send_bottom,
        buffers.d_send_left, buffers.d_send_right,
        ny, nx
    )

    buffers.d_send_top.copy_to_host(buffers.h_send_top, stream=s)
    buffers.d_send_bottom.copy_to_host(buffers.h_send_bottom, stream=s)
    buffers.d_send_left.copy_to_host(buffers.h_send_left, stream=s)
    buffers.d_send_right.copy_to_host(buffers.h_send_right, stream=s)
    s.synchronize()

    requests = _post_halo_mpi(ctx, buffers, ny, nx)
    return requests


# ---------------------------------------------------------------------
# TASK 4: 16x16 shared-memory tiled stencil with 18x18 apron
# ---------------------------------------------------------------------
@cuda.jit
def tiled_stencil_kernel(u, kappa, source, out, ny, nx, dt, dx, dy,
                         r_start, r_end, c_start, c_end):
    sm_u = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    sm_k = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    base_r = cuda.blockIdx.y * TILE_DIM + 1
    base_c = cuda.blockIdx.x * TILE_DIM + 1

    # 256 threads cooperatively load all 324 tile locations.
    linear = ty * TILE_DIM + tx
    for q in range(2):
        sidx = linear + q * TILE_DIM * TILE_DIM
        if sidx < SHARED_DIM * SHARED_DIM:
            sy = sidx // SHARED_DIM
            sx = sidx % SHARED_DIM
            gi = base_r + sy - 1
            gj = base_c + sx - 1

            if gi < 0:
                gi = 0
            if gi > ny + 1:
                gi = ny + 1
            if gj < 0:
                gj = 0
            if gj > nx + 1:
                gj = nx + 1

            sm_u[sy, sx] = u[gi, gj]
            sm_k[sy, sx] = kappa[gi, gj]

    cuda.syncthreads()

    i = base_r + ty
    j = base_c + tx

    if (i >= r_start and i < r_end and
            j >= c_start and j < c_end and
            i >= 1 and i <= ny and j >= 1 and j <= nx):

        center = sm_u[ty + 1, tx + 1]
        kc = sm_k[ty + 1, tx + 1]

        kup = 0.5 * (kc + sm_k[ty, tx + 1])
        kdown = 0.5 * (kc + sm_k[ty + 2, tx + 1])
        kleft = 0.5 * (kc + sm_k[ty + 1, tx])
        kright = 0.5 * (kc + sm_k[ty + 1, tx + 2])

        diff_y = (
            kdown * (sm_u[ty + 2, tx + 1] - center)
            - kup * (center - sm_u[ty, tx + 1])
        ) / (dy * dy)

        diff_x = (
            kright * (sm_u[ty + 1, tx + 2] - center)
            - kleft * (center - sm_u[ty + 1, tx])
        ) / (dx * dx)

        out[i, j] = center + dt * (
            diff_x + diff_y
            - LAMBDA_DAMP * center * center * center
            + source[i, j]
        )


def launch_stencil(u_d, k_d, s_d, out_d, ny, nx, stream,
                   r_start, r_end, c_start, c_end):
    threads = (TILE_DIM, TILE_DIM)
    blocks = ((nx + TILE_DIM - 1) // TILE_DIM,
              (ny + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[blocks, threads, stream](
        u_d, k_d, s_d, out_d, ny, nx, DT, DX, DY,
        r_start, r_end, c_start, c_end
    )


def step_overlapped(u_d, k_d, s_d, out_d, ctx, buffers, ny, nx):
    # Stage A: pack borders and copy to pinned host buffers.
    requests = exchange_halos_async(u_d, ctx, buffers, ny, nx)

    # Stage B: compute deep interior while MPI transfers are in flight.
    launch_stencil(
        u_d, k_d, s_d, out_d, ny, nx, buffers.stream_interior,
        2, ny, 2, nx
    )

    # Stage C: wait for MPI, copy received halos back to GPU and unpack.
    if requests:
        MPI.Request.Waitall(requests)

    sb = buffers.stream_boundary
    buffers.d_recv_top.copy_to_device(buffers.h_recv_top, stream=sb)
    buffers.d_recv_bottom.copy_to_device(buffers.h_recv_bottom, stream=sb)
    buffers.d_recv_left.copy_to_device(buffers.h_recv_left, stream=sb)
    buffers.d_recv_right.copy_to_device(buffers.h_recv_right, stream=sb)

    unpack_blocks = (max(ny, nx) + 255) // 256
    unpack_halos_kernel[unpack_blocks, 256, sb](
        u_d,
        buffers.d_recv_top, buffers.d_recv_bottom,
        buffers.d_recv_left, buffers.d_recv_right,
        ny, nx
    )

    # Boundary ring: top, bottom, left, right. Corners may be launched twice;
    # each launch writes the same deterministic value at those corner cells.
    launch_stencil(u_d, k_d, s_d, out_d, ny, nx, sb, 1, 2, 1, nx + 1)
    launch_stencil(u_d, k_d, s_d, out_d, ny, nx, sb, ny, ny + 1, 1, nx + 1)
    launch_stencil(u_d, k_d, s_d, out_d, ny, nx, sb, 2, ny, 1, 2)
    launch_stencil(u_d, k_d, s_d, out_d, ny, nx, sb, 2, ny, nx, nx + 1)

    # Ensure both streams have finished before swapping buffers.
    buffers.stream_interior.synchronize()
    buffers.stream_boundary.synchronize()


def step_serialized(u_d, k_d, s_d, out_d, ctx, buffers, ny, nx):
    requests = exchange_halos_async(u_d, ctx, buffers, ny, nx)
    if requests:
        MPI.Request.Waitall(requests)

    sb = buffers.stream_boundary
    buffers.d_recv_top.copy_to_device(buffers.h_recv_top, stream=sb)
    buffers.d_recv_bottom.copy_to_device(buffers.h_recv_bottom, stream=sb)
    buffers.d_recv_left.copy_to_device(buffers.h_recv_left, stream=sb)
    buffers.d_recv_right.copy_to_device(buffers.h_recv_right, stream=sb)

    unpack_halos_kernel[1, 256, sb](
        u_d,
        buffers.d_recv_top, buffers.d_recv_bottom,
        buffers.d_recv_left, buffers.d_recv_right,
        ny, nx
    )

    launch_stencil(u_d, k_d, s_d, out_d, ny, nx, sb, 1, ny + 1, 1, nx + 1)
    sb.synchronize()


# ---------------------------------------------------------------------
# TASK 4: MPI-3 shared-window false-sharing benchmark and LLVM audit
# ---------------------------------------------------------------------
@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iterations):
    # Deliberate in-place read-modify-write on shared memory. Keeping the
    # array access in the loop prevents replacing it with a local accumulator.
    for _ in range(iterations):
        shm_view[slot] = shm_view[slot] + np.uint64(1)


def benchmark_shm_strides(ctx):
    node_comm = ctx["node_comm"]
    rank = ctx["node_rank"]
    shm_view = ctx["shm_view"]
    results = {}

    for stride in (1, 8, 16):
        if rank == 0:
            shm_view[:] = 0
        ctx["shm_win"].Sync()
        node_comm.Barrier()

        slot = rank * stride
        start = time.perf_counter()
        host_shm_accumulate(shm_view, slot, SHM_ITERS)
        ctx["shm_win"].Sync()
        node_comm.Barrier()
        elapsed = time.perf_counter() - start
        max_elapsed = node_comm.allreduce(elapsed, op=MPI.MAX)

        if rank == 0:
            results[str(stride)] = {
                "seconds": float(max_elapsed),
                "milliseconds": float(max_elapsed * 1000.0),
                "iterations_per_rank": SHM_ITERS,
                "values": [int(shm_view[r * stride]) for r in range(4)],
            }

    llvm_ir = host_shm_accumulate.inspect_llvm(host_shm_accumulate.signatures[0])
    llvm_counts = {
        "load_instructions": len(re.findall(r"\bload\b", llvm_ir)),
        "store_instructions": len(re.findall(r"\bstore\b", llvm_ir)),
        "has_load_and_store": bool(re.search(r"\bload\b", llvm_ir) and re.search(r"\bstore\b", llvm_ir)),
    }
    if rank == 0:
        results["false_sharing_slowdown_stride1_vs_stride8"] = (
            results["1"]["milliseconds"] / results["8"]["milliseconds"]
            if results["8"]["milliseconds"] else None
        )
        results["padding_128B_vs_64B_ratio_stride16_over_stride8"] = (
            results["16"]["milliseconds"] / results["8"]["milliseconds"]
            if results["8"]["milliseconds"] else None
        )
    results["llvm_audit"] = llvm_counts
    return results, llvm_ir


# ---------------------------------------------------------------------
# TASK 5: reductions, parity, PTX audit, benchmark, manifest
# ---------------------------------------------------------------------
def global_diagnostics(u_d, ctx, ny, nx):
    local = u_d.copy_to_host()
    interior = local[1:ny + 1, 1:nx + 1]
    local_energy = float(np.sum(interior) * DX * DY)
    local_l2_sq = float(np.sum(interior * interior) * DX * DY)

    energy = ctx["cart"].allreduce(local_energy, op=MPI.SUM)
    l2 = math.sqrt(ctx["cart"].allreduce(local_l2_sq, op=MPI.SUM))
    return energy, l2


def ptx_audit(kernel, u_d, k_d, s_d, out_d, ny, nx):
    # Compile the exact kernel specialization used by this run.
    launch_stencil(
        u_d, k_d, s_d, out_d, ny, nx,
        cuda.stream(), 1, ny + 1, 1, nx + 1
    )
    cuda.synchronize()

    asm = kernel.inspect_asm()
    combined = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)

    counts = {
        "st_shared_f64": len(re.findall(r"\bst\.shared(?:\.f64)?\b", combined)),
        "ld_shared_f64": len(re.findall(r"\bld\.shared(?:\.f64)?\b", combined)),
        "bar_sync": len(re.findall(r"\bbar\.sync\b", combined)),
    }
    # More precise typed instruction counts where present.
    counts["st_shared_f64_exact"] = len(
        re.findall(r"\bst\.shared\.f64\b", combined)
    )
    counts["ld_shared_f64_exact"] = len(
        re.findall(r"\bld\.shared\.f64\b", combined)
    )
    return counts, combined


def run_mode(mode, initial_u, kappa, source, ctx, ny, nx):
    u_d = cuda.to_device(initial_u)
    out_d = cuda.device_array_like(u_d)
    k_d = cuda.to_device(kappa)
    s_d = cuda.to_device(source)
    buffers = HaloBuffers(ny, nx)

    trace = []
    start = time.perf_counter()

    e0, l20 = global_diagnostics(u_d, ctx, ny, nx)
    trace.append({"step": 0, "energy": e0, "l2": l20})

    step_fn = step_serialized if mode == "serialized" else step_overlapped

    for step in range(1, STEPS + 1):
        step_fn(u_d, k_d, s_d, out_d, ctx, buffers, ny, nx)
        u_d, out_d = out_d, u_d

        if step % 100 == 0:
            energy, l2 = global_diagnostics(u_d, ctx, ny, nx)
            trace.append({"step": step, "energy": energy, "l2": l2})

    cuda.synchronize()
    elapsed = time.perf_counter() - start
    final_state = u_d.copy_to_host()
    return elapsed, final_state, trace, (u_d, out_d, k_d, s_d, buffers)


def main(github_user):
    ctx = init_distributed_context(github_user)
    u0, kappa, source, ny, nx, centers, amplitudes = initialize_subdomains(ctx)

    print(
        f"[rank {ctx['rank']}] coords={ctx['coords']} "
        f"neighbors=({ctx['nbr_up']},{ctx['nbr_down']},"
        f"{ctx['nbr_left']},{ctx['nbr_right']})",
        flush=True
    )
    print(
        f"[rank {ctx['rank']}] local shape={u0.shape}, "
        f"seed={ctx['instance_seed']}",
        flush=True
    )

    # Run the CPU false-sharing sweep once, before GPU timing runs.
    shm_results, llvm_text = benchmark_shm_strides(ctx)
    if ctx["rank"] == 0:
        with open("llvm_audit.txt", "w", encoding="utf-8") as f:
            f.write(llvm_text)
        print("\n========== SHARED-MEMORY STRIDE SWEEP ==========")
        print(json.dumps(shm_results, indent=2))

    # Ensure every rank begins each benchmark from the same local initial state.
    ctx["cart"].Barrier()
    serialized_ms, serialized_final, serialized_trace, serialized_objs = run_mode(
        "serialized", u0, kappa, source, ctx, ny, nx
    )

    ctx["cart"].Barrier()
    overlapped_ms, overlapped_final, overlapped_trace, overlapped_objs = run_mode(
        "overlapped", u0, kappa, source, ctx, ny, nx
    )

    local_diff = float(np.max(np.abs(serialized_final - overlapped_final)))
    parity_max_diff = ctx["cart"].allreduce(local_diff, op=MPI.MAX)

    # Audit the compiled overlapped kernel specialization.
    u_d, out_d, k_d, s_d, _ = overlapped_objs
    ptx_counts, ptx_text = ptx_audit(
        tiled_stencil_kernel, u_d, k_d, s_d, out_d, ny, nx
    )

    max_kappa_local = float(np.max(kappa))
    max_kappa = ctx["cart"].allreduce(max_kappa_local, op=MPI.MAX)
    dt_max = DX * DX / (4.0 * max_kappa)

    # All ranks hold identical global reductions in their trace.
    if ctx["rank"] == 0:
        manifest = {
            "github_username": github_user,
            "gpu_uuid": ctx["gpu_uuid"],
            "cpu_signature": ctx["cpu_signature"],
            "cpu_cache_line_bytes": ctx["cache_line_bytes"],
            "shared_window_alignment_bytes": 64,
            "shared_window_slots": SHM_SLOTS,
            "gpu": "NVIDIA Tesla T4",
            "compute_capability": "7.5",
            "mpi_ranks": ctx["size"],
            "cartesian_grid": [2, 2],
            "local_shape_with_halo": [ny + 2, nx + 2],
            "steps": STEPS,
            "dt": DT,
            "dx": DX,
            "kappa_max": max_kappa,
            "dt_stability_limit": dt_max,
            "shm_stride_benchmark": shm_results,
            "llvm_audit_passed": shm_results["llvm_audit"]["has_load_and_store"],
            "serialized_ms_rank0": serialized_ms * 1000.0,
            "overlapped_ms_rank0": overlapped_ms * 1000.0,
            "parity_max_diff": parity_max_diff,
            "ptx_audit": ptx_counts,
            "ptx_audit_passed": (
                ptx_counts["st_shared_f64"] > 0
                and ptx_counts["ld_shared_f64"] > 0
                and ptx_counts["bar_sync"] > 0
            ),
            "serialized_energy_trace": serialized_trace,
            "overlapped_energy_trace": overlapped_trace,
            "pulse_centers": centers.tolist(),
            "pulse_amplitudes": amplitudes.tolist(),
        }

        with open("submission_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n========== RUN SUMMARY ==========")
        print(json.dumps(manifest, indent=2))
        print("Saved submission_manifest.json")

        with open("ptx_audit.txt", "w", encoding="utf-8") as f:
            f.write(ptx_text)
        print("Saved ptx_audit.txt")
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Saved hybrid_lab_manifest.json")

    ctx["cart"].Barrier()
    ctx["shm_win"].Free()
    ctx["node_comm"].Free()
    ctx["cart"].Free()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: mpirun -np 4 python hybrid_mpi_cuda_cache.py GITHUB_USERNAME"
        )
    main(sys.argv[1])
