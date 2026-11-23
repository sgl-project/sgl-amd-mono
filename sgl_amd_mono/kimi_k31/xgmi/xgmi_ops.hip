// SPDX-License-Identifier: Apache-2.0
// AMD Iris-style XGMI fused collective & compute kernels for Kimi-K3.1 on gfx950 (8x MI355X).
//
// Operates inside one <= 0.4 GiB uncached HIP IPC symmetric peer heap (214 MiB/GPU):
//   1. xgmi_lamport_allgather       (Decode 1..192 rows: 1/8 latent_input_gather, zero-copy dual split)
//   2. xgmi_deferred_topk_lamport_ar (Decode 1..192 rows: deferred top-k + routed AR + shared RS/AR)
//   3. xgmi_moetail_column_gather   (Decode 1..192 rows: 1/8 up_slice + shared + prefix -> full stream)
//   4. xgmi_fused_rmsnorm_bf16      (Decode 1..192 rows: 1-launch wave64 RMSNorm for latent [M, 3584])
//   5. xgmi_rs_chunk_32cu / push / reduce (Prefill 200..8192 rows: 32-CU RS overlapped with 224-CU o_proj)
//   6. xgmi_fp4_pack_allgather      (Prefill 200..8192 rows: fused E2M1 FP4 / FP8 + scale + topk AllGather)
//   7. xgmi_row_allgather           (Prefill 200..8192 rows: 2112-dim qkv_a AG & chunked 7168 Attn AG)

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#ifndef C10_HIP_KERNEL_LAUNCH_CHECK
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

namespace {

using bf16_t = __bf16;
using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));

constexpr int TP = 8;

constexpr int DECODE_BLOCKS = 128;
constexpr int DECODE_THREADS = 256;

constexpr int RS_BLOCKS = 64;
constexpr int RS_THREADS = 512;
constexpr int RS_RING_SLOTS = 4;

constexpr int AG_BLOCKS = 64;
constexpr int AG_THREADS = 512;

constexpr int64_t align_up_4k(int64_t n) { return (n + 4095ll) & ~4095ll; }

// Single source of truth for the 214 MiB XgmiHeap layout (matches heap.py).
constexpr int64_t DECODE_RS_BYTES = align_up_4k(192ll * 10752ll * 4ll);
constexpr int64_t DECODE_AG_BYTES = align_up_4k(192ll * 10752ll * 4ll);
constexpr int64_t DECODE_HALF_BYTES = align_up_4k(DECODE_RS_BYTES + DECODE_AG_BYTES);
constexpr int64_t DECODE_REGION_OFF = 0;
constexpr int64_t DECODE_REGION_BYTES = 2 * DECODE_HALF_BYTES;

constexpr int64_t RS_SLOT_BYTES = align_up_4k(2048ll * 7168ll * 2ll);
constexpr int64_t RS_REGION_OFF = align_up_4k(DECODE_REGION_OFF + DECODE_REGION_BYTES);
constexpr int64_t RS_REGION_BYTES = RS_RING_SLOTS * RS_SLOT_BYTES;

constexpr int64_t AG_HALF_BYTES = 35ll << 20;
constexpr int64_t AG_REGION_OFF = align_up_4k(RS_REGION_OFF + RS_REGION_BYTES);
constexpr int64_t AG_REGION_BYTES = 2 * AG_HALF_BYTES;

constexpr int64_t RS_FLAGS_OFF = align_up_4k(AG_REGION_OFF + AG_REGION_BYTES);
constexpr int64_t RS_FLAGS_BYTES = align_up_4k(RS_RING_SLOTS * RS_BLOCKS * TP * 16ll);
constexpr int64_t AG_FLAGS_OFF = align_up_4k(RS_FLAGS_OFF + RS_FLAGS_BYTES);
constexpr int64_t AG_FLAGS_BYTES = align_up_4k(2ll * AG_BLOCKS * TP * 16ll);
constexpr int64_t HEAP_MIN_BYTES = AG_FLAGS_OFF + AG_FLAGS_BYTES;

constexpr int EPOCH_DECODE_OFF = 0;
constexpr int EPOCH_RS_OFF = DECODE_BLOCKS;
constexpr int EPOCH_AG_OFF = EPOCH_RS_OFF + RS_BLOCKS;
constexpr int EPOCH_TOTAL_WORDS = EPOCH_AG_OFF + AG_BLOCKS;

static_assert(HEAP_MIN_BYTES <= (214ll << 20), "XgmiHeap exceeds 214 MiB");

struct PeerTable {
    uint8_t *base[TP];
};

constexpr int64_t HEAP_GRANULE = 2ll << 20;
constexpr double MIN_DEADLINE_S = 1e-8;
constexpr double MAX_DEADLINE_S = 86400.0;
constexpr uint32_t MAX_BLOCK_EPOCH = 0x3FFFFFFFu;
constexpr uint32_t FLAG_OP_RS = 1u;
constexpr uint32_t FLAG_OP_AG_FP4 = 2u;
constexpr uint32_t FLAG_OP_AG_ROW = 3u;

constexpr bool next_epoch_valid(uint64_t launches) {
    return launches < static_cast<uint64_t>(MAX_BLOCK_EPOCH);
}

static inline uint64_t checked_deadline_ticks(double deadline_s) {
    TORCH_CHECK(std::isfinite(deadline_s) && deadline_s >= MIN_DEADLINE_S &&
                    deadline_s <= MAX_DEADLINE_S,
                "deadline_s must be finite and in [1e-8, 86400]");
    const uint64_t ticks = static_cast<uint64_t>(deadline_s * 1e8);
    TORCH_CHECK(ticks >= 1u, "deadline_s converts to zero watchdog ticks");
    return ticks;
}

static inline int checked_int32_dim(int64_t val, const char *name) {
    TORCH_CHECK(val >= 0 && val <= std::numeric_limits<int>::max(),
                name, " exceeds int32 range");
    return static_cast<int>(val);
}

constexpr bool decode_ag_dims_valid(
    int64_t M, int64_t slice_cols, int64_t split_cols = 0, bool has_out_b = false) {
    if (M < 1 || slice_cols < 4 || (slice_cols & 3) != 0) return false;
    if (M > DECODE_AG_BYTES || slice_cols > DECODE_AG_BYTES / (TP * 4)) return false;
    if (has_out_b) {
        if (split_cols <= 0 || split_cols >= slice_cols || (split_cols & 3) != 0) return false;
    } else if (split_cols != 0) {
        return false;
    }
    return M * slice_cols * TP * 4 <= DECODE_AG_BYTES;
}

constexpr bool decode_ar_dims_valid(
    int64_t M, int64_t top_k, int64_t D_lat, int64_t D_sh, bool gather_full_shared) {
    if (M < 1 || top_k < 1 || top_k > 65536 || D_lat < TP * 4 || D_sh < TP * 4) return false;
    if ((D_lat % (TP * 4)) != 0 || (D_sh % (TP * 4)) != 0) return false;
    if (M > DECODE_RS_BYTES || D_lat > DECODE_RS_BYTES / 4 || D_sh > DECODE_RS_BYTES / 4) return false;
    if (M > std::numeric_limits<int>::max() / top_k) return false;
    if (M * (D_lat + D_sh) * 4 > DECODE_RS_BYTES) return false;
    if (M * (D_lat + (gather_full_shared ? D_sh : 0ll)) * 4 > DECODE_AG_BYTES) return false;
    return true;
}

constexpr bool fp4_pack_dims_valid(
    int64_t rows, int64_t q_bytes, int64_t s_bytes, int64_t top_k) {
    if (rows < 1 || rows > AG_HALF_BYTES / TP) return false;
    if (q_bytes < 16 || (q_bytes & 15) != 0 || (q_bytes >> 4) >= 65536) return false;
    if (s_bytes < 16 || (s_bytes & 15) != 0 || (s_bytes >> 4) >= 256) return false;
    if (top_k < 4 || (top_k & 3) != 0 || (top_k >> 2) >= 256) return false;
    const int64_t row_bytes = q_bytes + s_bytes + 8ll * top_k;
    if (row_bytes > AG_HALF_BYTES / TP) return false;
    return static_cast<int64_t>(TP) * rows * row_bytes <= AG_HALF_BYTES;
}

constexpr bool row_ag_dims_valid(int64_t rows, int64_t cols) {
    if (rows < 1 || cols < 8 || (cols & 7) != 0) return false;
    if (rows > AG_HALF_BYTES / (TP * 16) || cols > AG_HALF_BYTES / (TP * 2)) return false;
    return rows * TP * cols * 2 <= AG_HALF_BYTES;
}

// Compile-time overflow-free regression checks (signed overflow is ill-formed in constexpr).
static_assert(next_epoch_valid(0));
static_assert(next_epoch_valid(static_cast<uint64_t>(MAX_BLOCK_EPOCH) - 1u));
static_assert(!next_epoch_valid(static_cast<uint64_t>(MAX_BLOCK_EPOCH)));
static_assert(!next_epoch_valid(static_cast<uint64_t>(MAX_BLOCK_EPOCH) + 1u));
static_assert(decode_ag_dims_valid(192, 1344));
static_assert(!decode_ag_dims_valid(1, 1ll << 30));
static_assert(decode_ag_dims_valid(1, 4, 0, false));
static_assert(!decode_ag_dims_valid(1, 4, 1ll << 30, false));
static_assert(decode_ag_dims_valid(192, 560, 112, true));
static_assert(!decode_ag_dims_valid(192, 560, 560, true));
static_assert(decode_ar_dims_valid(192, 8, 3584, 7168, true));
static_assert(!decode_ar_dims_valid(1, 8, 1ll << 30, 1ll << 30, true));
static_assert(!decode_ar_dims_valid(1, 8, 1ll << 30, 1ll << 30, false));
static_assert(!decode_ar_dims_valid(1, 1ll << 30, 3584, 7168, true));
static_assert(fp4_pack_dims_valid(1024, 1792, 112, 8));
static_assert(!fp4_pack_dims_valid(1, 16, 16, 1ll << 30));
static_assert(row_ag_dims_valid(1024, 2112));
static_assert(!row_ag_dims_valid(1, 1ll << 30));

static inline void close_opened_peers(const std::vector<void *> &opened) {
    for (void *p : opened) {
        (void)hipIpcCloseMemHandle(p);
    }
}

static inline void validate_peer_mapping(
    void *ptr,
    c10::Device dev,
    int64_t required_bytes,
    bool is_local) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    TORCH_CHECK(addr != 0 && (addr & 15u) == 0,
                "peer base pointer must be non-null and 16-byte aligned");
    hipPointerAttribute_t attr{};
    TORCH_CHECK(hipPointerGetAttributes(&attr, ptr) == hipSuccess &&
                    attr.type == hipMemoryTypeDevice,
                "peer pointer must reference a valid HIP device mapping");
    if (is_local) {
        TORCH_CHECK(attr.device == dev.index(),
                    "local heap pointer device does not match heap device");
    } else if (attr.device != dev.index()) {
        int can_access = 0;
        TORCH_CHECK(hipDeviceCanAccessPeer(&can_access, dev.index(), attr.device) == hipSuccess &&
                        can_access != 0,
                    "peer device is not P2P-accessible from heap device");
    }
    hipDeviceptr_t alloc_base = nullptr;
    size_t alloc_size = 0;
    TORCH_CHECK(hipMemGetAddressRange(&alloc_base, &alloc_size,
                                      reinterpret_cast<hipDeviceptr_t>(ptr)) == hipSuccess,
                "hipMemGetAddressRange failed for peer pointer");
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(alloc_base);
    TORCH_CHECK(addr >= base_addr &&
                    alloc_size >= static_cast<size_t>(addr - base_addr) &&
                    (alloc_size - static_cast<size_t>(addr - base_addr)) >=
                        static_cast<size_t>(required_bytes),
                "peer mapping is smaller than required heap size");
}

static inline void check_align(const at::Tensor &t, size_t align, const char *name) {
    TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & (align - 1)) == 0,
                name, " pointer must be aligned to ", align, " bytes");
}

static inline void check_epoch_span(const at::Tensor &block_epoch, int64_t epoch_off, int blocks, c10::Device dev) {
    TORCH_CHECK(block_epoch.is_contiguous() && block_epoch.dtype() == at::kInt &&
                    block_epoch.device() == dev && epoch_off >= 0 &&
                    epoch_off + blocks <= block_epoch.numel(),
                "block_epoch span out of bounds");
}

static inline hipStream_t checked_eager_stream(c10::Device dev) {
    const hipStream_t stream =
        at::hip::getCurrentHIPStreamMasqueradingAsCUDA(dev.index()).stream();
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    C10_HIP_CHECK(hipStreamIsCapturing(stream, &status));
    TORCH_CHECK(status == hipStreamCaptureStatusNone,
                "XgmiHeap kernels do not support HIP/CUDA graph capture or replay");
    return stream;
}

struct HeapHandle {
    PeerTable peers;
    int rank;
    int64_t heap_bytes;
    c10::Device device;
    void *local_ptr;
    std::vector<void *> opened_peers;
    at::Tensor block_epoch; // keeps device epoch buffer alive while registered
    int32_t *epoch_ptr;
    bool ready;
    uint64_t decode_launches;
    uint64_t rs_launches;
    uint64_t ag_launches;
};

static std::mutex &heap_mutex() {
    static std::mutex mtx;
    return mtx;
}

static std::unordered_map<int64_t, HeapHandle> &heap_registry() {
    static std::unordered_map<int64_t, HeapHandle> reg;
    return reg;
}

static std::vector<void *> &retired_local_heaps() {
    static std::vector<void *> retired;
    return retired;
}

static std::vector<HeapHandle> &retired_unsynced_heaps() {
    static std::vector<HeapHandle> retired;
    return retired;
}

static std::tuple<int64_t, at::Tensor> create_local_heap_op(
    const at::Tensor &block_epoch,
    int64_t heap_bytes) {
    TORCH_CHECK(heap_bytes >= HEAP_MIN_BYTES && heap_bytes <= (4ll << 30),
                "heap_bytes out of supported range [HEAP_MIN_BYTES, 4 GiB]");
    TORCH_CHECK(block_epoch.is_cuda() && block_epoch.is_contiguous() &&
                    block_epoch.dtype() == at::kInt &&
                    block_epoch.numel() >= EPOCH_TOTAL_WORDS,
                "block_epoch must be contiguous CUDA int32[>= EPOCH_TOTAL_WORDS]");
    const int64_t alloc_bytes = ((heap_bytes + HEAP_GRANULE - 1) / HEAP_GRANULE) * HEAP_GRANULE;
    const c10::Device dev = block_epoch.device();
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(dev);
    void *local_ptr = nullptr;
    C10_HIP_CHECK(hipExtMallocWithFlags(&local_ptr, static_cast<size_t>(alloc_bytes),
                                        hipDeviceMallocUncached));
    try {
        C10_HIP_CHECK(hipMemset(local_ptr, 0, static_cast<size_t>(alloc_bytes)));
        hipIpcMemHandle_t ipc_handle{};
        C10_HIP_CHECK(hipIpcGetMemHandle(&ipc_handle, local_ptr));
        validate_peer_mapping(local_ptr, dev, alloc_bytes, true);
        at::Tensor handle_cpu = at::empty(
            {static_cast<int64_t>(sizeof(hipIpcMemHandle_t))},
            at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        std::memcpy(handle_cpu.data_ptr<uint8_t>(), &ipc_handle, sizeof(hipIpcMemHandle_t));
        std::lock_guard<std::mutex> lock(heap_mutex());
        static int64_t next_id = 1;
        const int64_t id = next_id++;
        heap_registry().emplace(
            id,
            HeapHandle{PeerTable{}, -1, alloc_bytes, dev, local_ptr, {},
                       block_epoch, block_epoch.data_ptr<int32_t>(), false,
                       0u, 0u, 0u});
        return std::make_tuple(id, handle_cpu);
    } catch (...) {
        (void)hipFree(local_ptr);
        throw;
    }
}

static void open_peer_heaps_op(
    int64_t heap_id,
    const at::Tensor &peer_handles_cpu,
    int64_t rank) {
    TORCH_CHECK(rank >= 0 && rank < TP, "rank out of range [0, 8)");
    TORCH_CHECK(peer_handles_cpu.device().is_cpu() &&
                    peer_handles_cpu.dtype() == at::kByte &&
                    peer_handles_cpu.is_contiguous() &&
                    peer_handles_cpu.dim() == 2 &&
                    peer_handles_cpu.size(0) == TP &&
                    peer_handles_cpu.size(1) == static_cast<int64_t>(sizeof(hipIpcMemHandle_t)),
                "peer_handles_cpu must be contiguous CPU uint8[8, sizeof(hipIpcMemHandle_t)]");
    std::lock_guard<std::mutex> lock(heap_mutex());
    auto it = heap_registry().find(heap_id);
    TORCH_CHECK(it != heap_registry().end(), "Unknown or unregistered XgmiHeap id: ", heap_id);
    HeapHandle &h = it->second;
    TORCH_CHECK(!h.ready, "XgmiHeap peer mappings already opened for id: ", heap_id);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(h.device);
    PeerTable pt{};
    std::vector<void *> opened;
    opened.reserve(TP - 1);
    pt.base[rank] = reinterpret_cast<uint8_t *>(h.local_ptr);
    const uint8_t *raw_handles = peer_handles_cpu.data_ptr<uint8_t>();
    try {
        for (int r = 0; r < TP; ++r) {
            if (r == rank) {
                continue;
            }
            hipIpcMemHandle_t handle{};
            std::memcpy(
                &handle,
                raw_handles + static_cast<size_t>(r) * sizeof(hipIpcMemHandle_t),
                sizeof(hipIpcMemHandle_t));
            void *peer_ptr = nullptr;
            C10_HIP_CHECK(hipIpcOpenMemHandle(
                &peer_ptr, handle, hipIpcMemLazyEnablePeerAccess));
            opened.push_back(peer_ptr);
            validate_peer_mapping(peer_ptr, h.device, h.heap_bytes, false);
            uint8_t *peer_u8 = reinterpret_cast<uint8_t *>(peer_ptr);
            for (int j = 0; j < r; ++j) {
                TORCH_CHECK(pt.base[j] != peer_u8, "duplicate peer mapping address");
            }
            TORCH_CHECK(peer_u8 != pt.base[rank], "peer mapping aliases local heap");
            pt.base[r] = peer_u8;
        }
    } catch (...) {
        close_opened_peers(opened);
        throw;
    }
    h.peers = pt;
    h.rank = static_cast<int>(rank);
    h.opened_peers = std::move(opened);
    h.ready = true;
}

static void unregister_heap_op(int64_t heap_id) {
    std::optional<HeapHandle> removed;
    {
        std::lock_guard<std::mutex> lock(heap_mutex());
        auto it = heap_registry().find(heap_id);
        if (it != heap_registry().end()) {
            removed.emplace(std::move(it->second));
            heap_registry().erase(it);
        }
    }
    if (removed.has_value()) {
        const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(removed->device);
        const hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            // Cannot confirm local kernels finished; retain opened_peers, local_ptr,
            // and block_epoch rather than unmapping/freeing active GPU memory.
            std::lock_guard<std::mutex> lock(heap_mutex());
            retired_unsynced_heaps().push_back(std::move(*removed));
            C10_HIP_CHECK(sync_err);
        }
        close_opened_peers(removed->opened_peers);
        if (removed->local_ptr != nullptr) {
            // Keep exported uncached local memory alive in retired_local_heaps (same
            // safety rule as sgl_amd_mono.common.peer._RETIRED) so non-collective close
            // cannot hipFree memory while a peer still holds an imported IPC handle.
            std::lock_guard<std::mutex> lock(heap_mutex());
            retired_local_heaps().push_back(removed->local_ptr);
        }
    }
}

static HeapHandle &lookup_heap_locked(int64_t heap_id, c10::Device dev) {
    auto it = heap_registry().find(heap_id);
    TORCH_CHECK(it != heap_registry().end(), "Unknown or unregistered XgmiHeap id: ", heap_id);
    TORCH_CHECK(it->second.ready, "XgmiHeap peer mappings not opened for id: ", heap_id);
    TORCH_CHECK(it->second.device == dev, "Tensor device does not match registered XgmiHeap device");
    return it->second;
}

// 16 B uncached peer store (sc0 sc1 nt); empirical gfx950 Lamport protocol matching ll_hip.cuh.
__device__ inline void store_nt16(uint32x4 *ptr, uint32x4 val) {
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_nop 1"
                 ::"v"(ptr), "v"(val) : "memory");
}

// Pipelined bulk nontemporal load (compiler schedules full MLP across waves/peers!)
__device__ inline uint32x4 load_nt16_bulk(const uint32x4 *ptr) {
    return __builtin_nontemporal_load(ptr);
}

// Synchronized system-scope poll loads (for Lamport tags and arrival flags)
__device__ inline uint32x4 load_nt16_sync(const uint32x4 *ptr) {
    uint32x4 val;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_waitcnt vmcnt(0)"
                 : "=&v"(val) : "v"(ptr) : "memory");
    return val;
}

// Drain every wave's uncached XGMI stores first, then synchronize all waves in the block
// before any thread publishes the block arrival flag.
__device__ inline void drain_stores() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __threadfence_system();
    __syncthreads();
}

__device__ inline uint32x4 poll_lamport(const uint32x4 *src, uint32_t tag, uint64_t deadline) {
    uint64_t t0 = 0;
    for (uint32_t n = 1u;; ++n) {
        const uint32x4 pk = load_nt16_sync(src);
        if (pk.y == tag && pk.w == tag) {
            __threadfence_system();
            return pk;
        }
        __builtin_amdgcn_s_sleep(1);
        if ((n & 1023u) == 0u) {
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > deadline) {
                __builtin_trap();
            }
        }
    }
}

__device__ inline void poll_flag16(const uint32x4 *src, uint32x4 expected, uint64_t deadline) {
    uint64_t t0 = 0;
    for (uint32_t n = 1u;; ++n) {
        const uint32x4 got = load_nt16_sync(src);
        if (got.x == expected.x && got.y == expected.y &&
            got.z == expected.z && got.w == expected.w) {
            __threadfence_system();
            return;
        }
        __builtin_amdgcn_s_sleep(1);
        if ((n & 1023u) == 0u) {
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > deadline) {
                __builtin_trap();
            }
        }
    }
}

__device__ inline uint32_t advance_block_epoch(int32_t *block_epoch, int offset) {
    __shared__ uint32_t s_tag;
    if (threadIdx.x == 0) {
        const uint32_t cur = static_cast<uint32_t>(block_epoch[offset + blockIdx.x]) + 1u;
        if (cur == 0u || cur > MAX_BLOCK_EPOCH) {
            __builtin_trap();
        }
        block_epoch[offset + blockIdx.x] = static_cast<int32_t>(cur);
        s_tag = cur;
    }
    __syncthreads();
    return s_tag;
}

__device__ inline uint2 pack_bf16x4(float a0, float a1, float a2, float a3) {
    bf16_t b[4] = {
        static_cast<bf16_t>(a0),
        static_cast<bf16_t>(a1),
        static_cast<bf16_t>(a2),
        static_cast<bf16_t>(a3),
    };
    uint2 u;
    __builtin_memcpy(&u, b, 8);
    return u;
}

__device__ inline void unpack_bf16x4(uint32_t w0, uint32_t w1, float (&out)[4]) {
    bf16_t b[4];
    uint2 u = make_uint2(w0, w1);
    __builtin_memcpy(b, &u, 8);
    out[0] = static_cast<float>(b[0]);
    out[1] = static_cast<float>(b[1]);
    out[2] = static_cast<float>(b[2]);
    out[3] = static_cast<float>(b[3]);
}

__device__ inline void unpack_bf16x8(uint32x4 v, float (&out)[8]) {
    bf16_t b[8];
    __builtin_memcpy(b, &v, 16);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<float>(b[i]);
    }
}

__device__ inline uint32x4 pack_bf16x8(const float (&in)[8]) {
    bf16_t b[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        b[i] = static_cast<bf16_t>(in[i]);
    }
    uint32x4 v;
    __builtin_memcpy(&v, b, 16);
    return v;
}

// ============================================================================
// Kernel 0: Fast 1-launch RMSNorm for Decode latent [M, D] (eliminates 6 eager ops)
// ============================================================================
__global__ __launch_bounds__(256) void xgmi_fused_rmsnorm_bf16_kernel(
    const bf16_t *__restrict__ x,
    const bf16_t *__restrict__ w,
    bf16_t *__restrict__ out,
    int D,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int vecs = D >> 3;
    const uint32x4 *xr = reinterpret_cast<const uint32x4 *>(x + static_cast<int64_t>(row) * D);
    const uint32x4 *wr = reinterpret_cast<const uint32x4 *>(w);
    uint32x4 *orow = reinterpret_cast<uint32x4 *>(out + static_cast<int64_t>(row) * D);

    float sum_sq = 0.0f;
    for (int v = tid; v < vecs; v += 256) {
        float fv[8];
        unpack_bf16x8(xr[v], fv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sum_sq += fv[j] * fv[j];
        }
    }
    // Wave64 reduction + shared reduction across 4 waves
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset, 64);
    }
    __shared__ float s_wave[4];
    if ((tid & 63) == 0) {
        s_wave[tid >> 6] = sum_sq;
    }
    __syncthreads();
    __shared__ float s_inv;
    if (tid == 0) {
        float tot = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
        s_inv = rsqrtf(tot / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (int v = tid; v < vecs; v += 256) {
        float fv[8], wv[8];
        unpack_bf16x8(xr[v], fv);
        unpack_bf16x8(wr[v], wv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            fv[j] = fv[j] * inv * wv[j];
        }
        orow[v] = pack_bf16x8(fv);
    }
}

// ============================================================================
// Kernel 1: Decode 1/8 column AllGather (with optional zero-copy dual split)
// If out_b != nullptr: first split_cols of each rank's slice go to out_a
// [M, 8*split_cols] and the rest go to out_b [M, 8*(slice_cols-split_cols)].
// ============================================================================
__global__ __launch_bounds__(DECODE_THREADS) void xgmi_lamport_allgather_kernel(
    const bf16_t *__restrict__ x_slice,
    bf16_t *__restrict__ out_a,
    bf16_t *__restrict__ out_b, // nullable
    int M,
    int slice_cols,
    int split_cols,
    int rank,
    PeerTable peers,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int64_t reg_off = decode_off + static_cast<int64_t>(tag & 1u) * half_bytes + ag_sub_off;
    const int lines_per_row = slice_cols >> 2;
    const int split_lines = split_cols >> 2;
    const int b_slice_cols = slice_cols - split_cols;
    const int total_lines = M * lines_per_row;
    const int a_total_cols = (out_b == nullptr ? slice_cols : split_cols) * TP;
    const int b_total_cols = b_slice_cols * TP;
    const int tid = blockIdx.x * DECODE_THREADS + threadIdx.x;
    const int stride = DECODE_BLOCKS * DECODE_THREADS;

    auto write_val = [&](int row, int src, int l, uint2 v) {
        if (out_b == nullptr) {
            reinterpret_cast<uint2 *>(out_a + static_cast<int64_t>(row) * a_total_cols + src * slice_cols)[l] = v;
        } else if (l < split_lines) {
            reinterpret_cast<uint2 *>(out_a + static_cast<int64_t>(row) * a_total_cols + src * split_cols)[l] = v;
        } else {
            reinterpret_cast<uint2 *>(out_b + static_cast<int64_t>(row) * b_total_cols + src * b_slice_cols)[l - split_lines] = v;
        }
    };

    for (int idx = tid; idx < total_lines; idx += stride) {
        const int row = idx / lines_per_row;
        const int l = idx - row * lines_per_row;
        const uint2 v = reinterpret_cast<const uint2 *>(x_slice + static_cast<int64_t>(row) * slice_cols)[l];
        write_val(row, rank, l, v);
        const uint32x4 pkt = {v.x, tag, v.y, tag};
        const int64_t slot_idx = static_cast<int64_t>(rank) * total_lines + idx;
        #pragma unroll
        for (int dst = 0; dst < TP; ++dst) {
            if (dst != rank) {
                uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + reg_off) + slot_idx;
                store_nt16(dst_ptr, pkt);
            }
        }
    }

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + reg_off);
    for (int idx = tid; idx < total_lines; idx += stride) {
        const int row = idx / lines_per_row;
        const int l = idx - row * lines_per_row;
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            if (src != rank) {
                const int64_t slot_idx = static_cast<int64_t>(src) * total_lines + idx;
                const uint32x4 pkt = poll_lamport(own_ag + slot_idx, tag, deadline);
                write_val(row, src, l, make_uint2(pkt.x, pkt.z));
            }
        }
    }
}

// ============================================================================
// Kernel 2: Decode Deferred Top-K + Routed Latent AR + Shared RS/AR (Lamport)
// ============================================================================
__global__ __launch_bounds__(DECODE_THREADS) void xgmi_deferred_topk_lamport_ar_kernel(
    const bf16_t *__restrict__ expert_out,
    const int32_t *__restrict__ sorted_idx,
    const float *__restrict__ topk_weights,
    const bf16_t *__restrict__ shared_partial,
    bf16_t *__restrict__ latent_out,
    bf16_t *__restrict__ shared_out,
    int M,
    int top_k,
    int num_expert_rows,
    int D_lat,
    int D_sh,
    int gather_full_shared,
    int rank,
    PeerTable peers,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int64_t half_off = decode_off + static_cast<int64_t>(tag & 1u) * half_bytes;
    const int64_t rs_off = half_off;
    const int64_t ag_off = half_off + ag_sub_off;

    const int lat_slice = D_lat / TP;
    const int sh_slice = D_sh / TP;
    const int lat_lines = lat_slice >> 2;
    const int sh_lines = sh_slice >> 2;
    const int lat_slice_total = M * lat_lines;
    const int sh_slice_total = M * sh_lines;
    const int64_t rs_sh_off_vecs = static_cast<int64_t>(TP) * lat_slice_total;
    const int64_t ag_sh_off_vecs = static_cast<int64_t>(TP) * lat_slice_total;

    const int tid = blockIdx.x * DECODE_THREADS + threadIdx.x;
    const int stride = DECODE_BLOCKS * DECODE_THREADS;

    const int total_lat_all = M * (D_lat >> 2);
    for (int idx = tid; idx < total_lat_all; idx += stride) {
        const int row = idx / (D_lat >> 2);
        const int gl = idx - row * (D_lat >> 2);
        const int owner = gl / lat_lines;
        const int ll = gl - owner * lat_lines;
        const int col0 = gl << 2;

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const float *w_row = topk_weights + static_cast<int64_t>(row) * top_k;
        for (int k = 0; k < top_k; ++k) {
            int64_t exp_row;
            if (sorted_idx == nullptr) {
                exp_row = static_cast<int64_t>(row) * top_k + k;
            } else {
                const int sidx = sorted_idx[static_cast<int64_t>(row) * top_k + k];
                if (sidx < 0 || sidx >= num_expert_rows) {
                    continue;
                }
                exp_row = static_cast<int64_t>(sidx);
            }
            const float w = w_row[k];
            const uint2 raw = *reinterpret_cast<const uint2 *>(expert_out + exp_row * D_lat + col0);
            float vals[4];
            unpack_bf16x4(raw.x, raw.y, vals);
            acc[0] += w * vals[0];
            acc[1] += w * vals[1];
            acc[2] += w * vals[2];
            acc[3] += w * vals[3];
        }
        const uint2 packed = pack_bf16x4(acc[0], acc[1], acc[2], acc[3]);
        const uint32x4 pkt = {packed.x, tag, packed.y, tag};
        const int64_t dst_idx = static_cast<int64_t>(rank) * lat_slice_total + static_cast<int64_t>(row) * lat_lines + ll;
        uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[owner] + rs_off) + dst_idx;
        store_nt16(dst_ptr, pkt);
    }

    const int total_sh_all = M * (D_sh >> 2);
    for (int idx = tid; idx < total_sh_all; idx += stride) {
        const int row = idx / (D_sh >> 2);
        const int gl = idx - row * (D_sh >> 2);
        const int owner = gl / sh_lines;
        const int sl = gl - owner * sh_lines;
        const uint2 v = reinterpret_cast<const uint2 *>(shared_partial + static_cast<int64_t>(row) * D_sh)[gl];
        const uint32x4 pkt = {v.x, tag, v.y, tag};
        const int64_t dst_idx = rs_sh_off_vecs + static_cast<int64_t>(rank) * sh_slice_total +
                                static_cast<int64_t>(row) * sh_lines + sl;
        uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[owner] + rs_off) + dst_idx;
        store_nt16(dst_ptr, pkt);
    }

    const uint32x4 *own_rs = reinterpret_cast<const uint32x4 *>(peers.base[rank] + rs_off);
    for (int idx = tid; idx < lat_slice_total; idx += stride) {
        const int row = idx / lat_lines;
        const int ll = idx - row * lat_lines;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            const int64_t s_idx = static_cast<int64_t>(src) * lat_slice_total + idx;
            const uint32x4 pkt = poll_lamport(own_rs + s_idx, tag, deadline);
            float vals[4];
            unpack_bf16x4(pkt.x, pkt.z, vals);
            sum[0] += vals[0];
            sum[1] += vals[1];
            sum[2] += vals[2];
            sum[3] += vals[3];
        }
        const uint2 red = pack_bf16x4(sum[0], sum[1], sum[2], sum[3]);
        reinterpret_cast<uint2 *>(latent_out + static_cast<int64_t>(row) * D_lat + rank * lat_slice)[ll] = red;
        const uint32x4 ag_pkt = {red.x, tag, red.y, tag};
        const int64_t ag_idx = static_cast<int64_t>(rank) * lat_slice_total + idx;
        #pragma unroll
        for (int dst = 0; dst < TP; ++dst) {
            if (dst != rank) {
                uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + ag_off) + ag_idx;
                store_nt16(dst_ptr, ag_pkt);
            }
        }
    }

    for (int idx = tid; idx < sh_slice_total; idx += stride) {
        const int row = idx / sh_lines;
        const int sl = idx - row * sh_lines;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            const int64_t s_idx = rs_sh_off_vecs + static_cast<int64_t>(src) * sh_slice_total + idx;
            const uint32x4 pkt = poll_lamport(own_rs + s_idx, tag, deadline);
            float vals[4];
            unpack_bf16x4(pkt.x, pkt.z, vals);
            sum[0] += vals[0];
            sum[1] += vals[1];
            sum[2] += vals[2];
            sum[3] += vals[3];
        }
        const uint2 red = pack_bf16x4(sum[0], sum[1], sum[2], sum[3]);
        if (gather_full_shared) {
            reinterpret_cast<uint2 *>(shared_out + static_cast<int64_t>(row) * D_sh + rank * sh_slice)[sl] = red;
            const uint32x4 ag_pkt = {red.x, tag, red.y, tag};
            const int64_t ag_idx = ag_sh_off_vecs + static_cast<int64_t>(rank) * sh_slice_total + idx;
            #pragma unroll
            for (int dst = 0; dst < TP; ++dst) {
                if (dst != rank) {
                    uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + ag_off) + ag_idx;
                    store_nt16(dst_ptr, ag_pkt);
                }
            }
        } else {
            reinterpret_cast<uint2 *>(shared_out + static_cast<int64_t>(row) * sh_slice)[sl] = red;
        }
    }

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + ag_off);
    for (int idx = tid; idx < lat_slice_total; idx += stride) {
        const int row = idx / lat_lines;
        const int ll = idx - row * lat_lines;
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            if (src != rank) {
                const int64_t s_idx = static_cast<int64_t>(src) * lat_slice_total + idx;
                const uint32x4 pkt = poll_lamport(own_ag + s_idx, tag, deadline);
                reinterpret_cast<uint2 *>(latent_out + static_cast<int64_t>(row) * D_lat + src * lat_slice)[ll] = make_uint2(pkt.x, pkt.z);
            }
        }
    }

    if (gather_full_shared) {
        for (int idx = tid; idx < sh_slice_total; idx += stride) {
            const int row = idx / sh_lines;
            const int sl = idx - row * sh_lines;
            #pragma unroll
            for (int src = 0; src < TP; ++src) {
                if (src != rank) {
                    const int64_t s_idx = ag_sh_off_vecs + static_cast<int64_t>(src) * sh_slice_total + idx;
                    const uint32x4 pkt = poll_lamport(own_ag + s_idx, tag, deadline);
                    reinterpret_cast<uint2 *>(shared_out + static_cast<int64_t>(row) * D_sh + src * sh_slice)[sl] = make_uint2(pkt.x, pkt.z);
                }
            }
        }
    }
}

// ============================================================================
// Kernel 3: Decode 1/8 MoE Tail Column Gather (up_slice + shared + prefix -> stream)
// ============================================================================
__global__ __launch_bounds__(DECODE_THREADS) void xgmi_moetail_column_gather_kernel(
    const bf16_t *__restrict__ up_slice,
    const bf16_t *__restrict__ shared_slice,
    const bf16_t *__restrict__ prefix_slice,
    bf16_t *__restrict__ stream_out,
    int M,
    int slice_cols,
    int rank,
    PeerTable peers,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int64_t reg_off = decode_off + static_cast<int64_t>(tag & 1u) * half_bytes + ag_sub_off;
    const int lines_per_row = slice_cols >> 2;
    const int total_lines = M * lines_per_row;
    const int total_cols = slice_cols * TP;
    const int tid = blockIdx.x * DECODE_THREADS + threadIdx.x;
    const int stride = DECODE_BLOCKS * DECODE_THREADS;

    for (int idx = tid; idx < total_lines; idx += stride) {
        const int row = idx / lines_per_row;
        const int l = idx - row * lines_per_row;
        const int64_t off = static_cast<int64_t>(row) * slice_cols;
        const uint2 u_up = reinterpret_cast<const uint2 *>(up_slice + off)[l];
        const uint2 u_sh = reinterpret_cast<const uint2 *>(shared_slice + off)[l];
        float a[4], b[4];
        unpack_bf16x4(u_up.x, u_up.y, a);
        unpack_bf16x4(u_sh.x, u_sh.y, b);
        if (prefix_slice != nullptr) {
            const uint2 u_pr = reinterpret_cast<const uint2 *>(prefix_slice + off)[l];
            float c[4];
            unpack_bf16x4(u_pr.x, u_pr.y, c);
            a[0] += b[0] + c[0];
            a[1] += b[1] + c[1];
            a[2] += b[2] + c[2];
            a[3] += b[3] + c[3];
        } else {
            a[0] += b[0];
            a[1] += b[1];
            a[2] += b[2];
            a[3] += b[3];
        }
        const uint2 packed = pack_bf16x4(a[0], a[1], a[2], a[3]);
        reinterpret_cast<uint2 *>(stream_out + static_cast<int64_t>(row) * total_cols + rank * slice_cols)[l] = packed;
        const uint32x4 pkt = {packed.x, tag, packed.y, tag};
        const int64_t slot_idx = static_cast<int64_t>(rank) * total_lines + idx;
        #pragma unroll
        for (int dst = 0; dst < TP; ++dst) {
            if (dst != rank) {
                uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + reg_off) + slot_idx;
                store_nt16(dst_ptr, pkt);
            }
        }
    }

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + reg_off);
    for (int idx = tid; idx < total_lines; idx += stride) {
        const int row = idx / lines_per_row;
        const int l = idx - row * lines_per_row;
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            if (src != rank) {
                const int64_t slot_idx = static_cast<int64_t>(src) * total_lines + idx;
                const uint32x4 pkt = poll_lamport(own_ag + slot_idx, tag, deadline);
                reinterpret_cast<uint2 *>(stream_out + static_cast<int64_t>(row) * total_cols + src * slice_cols)[l] = make_uint2(pkt.x, pkt.z);
            }
        }
    }
}

// ============================================================================
// Kernel 4a: Prefill 32/64-block 7-Link-Parallel Chunked Reduce-Scatter
// Supports BOTH row sub-chunking [row_off : row_off + sub_rows) AND zero-copy
// column-chunked output placement out_shard[:, out_col_off : out_col_off + cols]
// so N-chunked o_proj GEMMs overlap 100% with 7-link-parallel XGMI Reduce-Scatter!
// ============================================================================
__global__ __launch_bounds__(RS_THREADS) void xgmi_rs_chunk_32cu_kernel(
    const bf16_t *__restrict__ partial,
    bf16_t *__restrict__ out_shard,
    int rows_per_rank,
    int row_off,
    int sub_rows,
    int cols,
    int out_col_off,
    int64_t out_stride_cols,
    int rank,
    PeerTable peers,
    int64_t rs_off,
    int64_t rs_slot_bytes,
    int64_t rs_flags_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const uint32x4 flag_pkt = {
        tag,
        (FLAG_OP_RS << 24) | static_cast<uint32_t>(rows_per_rank),
        (static_cast<uint32_t>(row_off) << 16) | static_cast<uint32_t>(sub_rows),
        (static_cast<uint32_t>(out_col_off) << 16) | static_cast<uint32_t>(cols)};
    const int slot = static_cast<int>(tag & (RS_RING_SLOTS - 1));
    const int64_t slot_off = rs_off + static_cast<int64_t>(slot) * rs_slot_bytes;
    const int64_t flag_base = rs_flags_off + static_cast<int64_t>(slot * RS_BLOCKS * TP + blockIdx.x * TP) * 16;

    const int vecs_per_row = cols >> 3;
    const int64_t out_vecs_stride = out_stride_cols >> 3;
    const int64_t out_vec_col0 = out_col_off >> 3;
    const int64_t total_vecs = static_cast<int64_t>(sub_rows) * vecs_per_row;
    const int64_t per_blk = (total_vecs + RS_BLOCKS - 1) / RS_BLOCKS;
    const int64_t v_begin = static_cast<int64_t>(blockIdx.x) * per_blk;
    const int64_t v_end = v_begin + per_blk < total_vecs ? v_begin + per_blk : total_vecs;

    // All 7 XGMI links fire concurrently via block-rotated peer order!
    for (int pi = 0; pi < TP - 1; ++pi) {
        const int p_off = 1 + ((pi + static_cast<int>(blockIdx.x)) % (TP - 1));
        const int dst = (rank + p_off) & (TP - 1);
        const int64_t src_row0 = static_cast<int64_t>(dst) * rows_per_rank + row_off;
        const uint32x4 *src_base = reinterpret_cast<const uint32x4 *>(partial + src_row0 * cols);
        uint32x4 *dst_base = reinterpret_cast<uint32x4 *>(peers.base[dst] + slot_off) +
                             static_cast<int64_t>(rank) * total_vecs;
        for (int64_t v = v_begin + threadIdx.x; v < v_end; v += RS_THREADS) {
            store_nt16(dst_base + v, src_base[v]);
        }
    }

    drain_stores();
    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int dst = threadIdx.x;
        uint32x4 *flag_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + flag_base) + rank;
        store_nt16(flag_ptr, flag_pkt);
    }

    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int src = threadIdx.x;
        const uint32x4 *flag_ptr = reinterpret_cast<const uint32x4 *>(peers.base[rank] + flag_base) + src;
        poll_flag16(flag_ptr, flag_pkt, deadline);
    }
    __syncthreads();

    const uint32x4 *own_slot = reinterpret_cast<const uint32x4 *>(peers.base[rank] + slot_off);
    const int64_t own_row0 = static_cast<int64_t>(rank) * rows_per_rank + row_off;
    const uint32x4 *local_src = reinterpret_cast<const uint32x4 *>(partial + own_row0 * cols);
    uint32x4 *out_base = reinterpret_cast<uint32x4 *>(out_shard + static_cast<int64_t>(row_off) * out_stride_cols);
    const bool direct_out = (out_stride_cols == cols && out_col_off == 0);

    for (int64_t v = v_begin + threadIdx.x; v < v_end; v += RS_THREADS) {
        uint32x4 raws[TP];
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            raws[src] = (src == rank)
                            ? local_src[v]
                            : load_nt16_bulk(own_slot + static_cast<int64_t>(src) * total_vecs + v);
        }
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            float vals[8];
            unpack_bf16x8(raws[src], vals);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] += vals[j];
            }
        }
        const uint32x4 packed = pack_bf16x8(acc);
        if (direct_out) {
            out_base[v] = packed;
        } else {
            const int64_t r = v / vecs_per_row;
            const int64_t cv = v - r * vecs_per_row;
            out_base[r * out_vecs_stride + out_vec_col0 + cv] = packed;
        }
    }
}

// ============================================================================
// Kernel 5: Prefill Fused E2M1 FP4 (or FP8) + Scale + TopK Pack & AllGather
// ============================================================================
__global__ __launch_bounds__(AG_THREADS) void xgmi_fp4_pack_allgather_kernel(
    const uint8_t *__restrict__ q_data,
    const uint8_t *__restrict__ q_scale,
    const int32_t *__restrict__ topk_ids,
    const float *__restrict__ topk_weights,
    uint8_t *__restrict__ q_data_all,
    uint8_t *__restrict__ q_scale_all,
    int32_t *__restrict__ topk_ids_all,
    float *__restrict__ topk_weights_all,
    int rows,
    int q_vecs,
    int s_vecs,
    int tk_vecs,
    int rank,
    PeerTable peers,
    int64_t ag_off,
    int64_t ag_half_bytes,
    int64_t ag_flags_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const uint32_t pack_geom =
        (static_cast<uint32_t>(q_vecs) << 16) |
        (static_cast<uint32_t>(s_vecs) << 8) |
        static_cast<uint32_t>(tk_vecs);
    const uint32x4 flag_pkt = {
        tag, FLAG_OP_AG_FP4, static_cast<uint32_t>(rows), pack_geom};
    const int par = static_cast<int>(tag & 1u);
    const int64_t half_off = ag_off + static_cast<int64_t>(par) * ag_half_bytes;
    const int64_t flag_base = ag_flags_off + static_cast<int64_t>(par * AG_BLOCKS * TP + blockIdx.x * TP) * 16;

    const int row_vecs = q_vecs + s_vecs + 2 * tk_vecs;
    const int64_t total_vecs = static_cast<int64_t>(rows) * row_vecs;
    const int64_t per_blk = (total_vecs + AG_BLOCKS - 1) / AG_BLOCKS;
    const int64_t v_begin = static_cast<int64_t>(blockIdx.x) * per_blk;
    const int64_t v_end = v_begin + per_blk < total_vecs ? v_begin + per_blk : total_vecs;

    const uint32x4 *qd_in = reinterpret_cast<const uint32x4 *>(q_data);
    const uint32x4 *qs_in = reinterpret_cast<const uint32x4 *>(q_scale);
    const uint32x4 *ti_in = reinterpret_cast<const uint32x4 *>(topk_ids);
    const uint32x4 *tw_in = reinterpret_cast<const uint32x4 *>(topk_weights);

    uint32x4 *qd_out = reinterpret_cast<uint32x4 *>(q_data_all);
    uint32x4 *qs_out = reinterpret_cast<uint32x4 *>(q_scale_all);
    uint32x4 *ti_out = reinterpret_cast<uint32x4 *>(topk_ids_all);
    uint32x4 *tw_out = reinterpret_cast<uint32x4 *>(topk_weights_all);

    for (int64_t idx = v_begin + threadIdx.x; idx < v_end; idx += AG_THREADS) {
        const int r = static_cast<int>(idx / row_vecs);
        const int lv = static_cast<int>(idx - static_cast<int64_t>(r) * row_vecs);
        const int dst_r = rank * rows + r;
        uint32x4 val;
        if (lv < q_vecs) {
            val = qd_in[static_cast<int64_t>(r) * q_vecs + lv];
            qd_out[static_cast<int64_t>(dst_r) * q_vecs + lv] = val;
        } else if (lv < q_vecs + s_vecs) {
            const int sv = lv - q_vecs;
            val = qs_in[static_cast<int64_t>(r) * s_vecs + sv];
            qs_out[static_cast<int64_t>(dst_r) * s_vecs + sv] = val;
        } else if (lv < q_vecs + s_vecs + tk_vecs) {
            const int tv = lv - q_vecs - s_vecs;
            val = ti_in[static_cast<int64_t>(r) * tk_vecs + tv];
            ti_out[static_cast<int64_t>(dst_r) * tk_vecs + tv] = val;
        } else {
            const int tv = lv - q_vecs - s_vecs - tk_vecs;
            val = tw_in[static_cast<int64_t>(r) * tk_vecs + tv];
            tw_out[static_cast<int64_t>(dst_r) * tk_vecs + tv] = val;
        }
        const int64_t peer_idx = static_cast<int64_t>(rank) * total_vecs + idx;
        #pragma unroll
        for (int pi = 0; pi < TP - 1; ++pi) {
            const int p_off = 1 + ((pi + static_cast<int>(blockIdx.x)) % (TP - 1));
            const int dst = (rank + p_off) & (TP - 1);
            uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + half_off) + peer_idx;
            store_nt16(dst_ptr, val);
        }
    }

    drain_stores();
    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int dst = threadIdx.x;
        uint32x4 *flag_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + flag_base) + rank;
        store_nt16(flag_ptr, flag_pkt);
    }

    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int src = threadIdx.x;
        const uint32x4 *flag_ptr = reinterpret_cast<const uint32x4 *>(peers.base[rank] + flag_base) + src;
        poll_flag16(flag_ptr, flag_pkt, deadline);
    }
    __syncthreads();

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + half_off);
    for (int p_off = 1; p_off < TP; ++p_off) {
        const int src = (rank + p_off) & (TP - 1);
        const uint32x4 *src_base = own_ag + static_cast<int64_t>(src) * total_vecs;
        for (int64_t idx = v_begin + threadIdx.x; idx < v_end; idx += AG_THREADS) {
            const int r = static_cast<int>(idx / row_vecs);
            const int lv = static_cast<int>(idx - static_cast<int64_t>(r) * row_vecs);
            const int dst_r = src * rows + r;
            const uint32x4 val = load_nt16_bulk(src_base + idx);
            if (lv < q_vecs) {
                qd_out[static_cast<int64_t>(dst_r) * q_vecs + lv] = val;
            } else if (lv < q_vecs + s_vecs) {
                qs_out[static_cast<int64_t>(dst_r) * s_vecs + (lv - q_vecs)] = val;
            } else if (lv < q_vecs + s_vecs + tk_vecs) {
                ti_out[static_cast<int64_t>(dst_r) * tk_vecs + (lv - q_vecs - s_vecs)] = val;
            } else {
                tw_out[static_cast<int64_t>(dst_r) * tk_vecs + (lv - q_vecs - s_vecs - tk_vecs)] = val;
            }
        }
    }
}

// ============================================================================
// Kernel 6: Prefill High-BW Row AllGather (full MLP bulk loads!)
// ============================================================================
__global__ __launch_bounds__(AG_THREADS) void xgmi_row_allgather_kernel(
    const bf16_t *__restrict__ x_shard,
    bf16_t *__restrict__ out_all,
    int rows,
    int cols,
    int rank,
    PeerTable peers,
    int64_t ag_off,
    int64_t ag_half_bytes,
    int64_t ag_flags_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const uint32x4 flag_pkt = {
        tag, FLAG_OP_AG_ROW, static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
    const int par = static_cast<int>(tag & 1u);
    const int64_t half_off = ag_off + static_cast<int64_t>(par) * ag_half_bytes;
    const int64_t flag_base = ag_flags_off + static_cast<int64_t>(par * AG_BLOCKS * TP + blockIdx.x * TP) * 16;

    const int vecs_per_row = cols >> 3;
    const int64_t total_vecs = static_cast<int64_t>(rows) * vecs_per_row;
    const int64_t per_blk = (total_vecs + AG_BLOCKS - 1) / AG_BLOCKS;
    const int64_t v_begin = static_cast<int64_t>(blockIdx.x) * per_blk;
    const int64_t v_end = v_begin + per_blk < total_vecs ? v_begin + per_blk : total_vecs;

    const uint32x4 *src_vec = reinterpret_cast<const uint32x4 *>(x_shard);
    uint32x4 *own_out = reinterpret_cast<uint32x4 *>(out_all) + static_cast<int64_t>(rank) * total_vecs;

    for (int64_t v = v_begin + threadIdx.x; v < v_end; v += AG_THREADS) {
        const uint32x4 val = src_vec[v];
        own_out[v] = val;
        const int64_t peer_idx = static_cast<int64_t>(rank) * total_vecs + v;
        #pragma unroll
        for (int pi = 0; pi < TP - 1; ++pi) {
            const int p_off = 1 + ((pi + static_cast<int>(blockIdx.x)) % (TP - 1));
            const int dst = (rank + p_off) & (TP - 1);
            uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + half_off) + peer_idx;
            store_nt16(dst_ptr, val);
        }
    }

    drain_stores();
    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int dst = threadIdx.x;
        uint32x4 *flag_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + flag_base) + rank;
        store_nt16(flag_ptr, flag_pkt);
    }

    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int src = threadIdx.x;
        const uint32x4 *flag_ptr = reinterpret_cast<const uint32x4 *>(peers.base[rank] + flag_base) + src;
        poll_flag16(flag_ptr, flag_pkt, deadline);
    }
    __syncthreads();

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + half_off);
    uint32x4 *out_base = reinterpret_cast<uint32x4 *>(out_all);
    for (int p_off = 1; p_off < TP; ++p_off) {
        const int src = (rank + p_off) & (TP - 1);
        const uint32x4 *src_buf = own_ag + static_cast<int64_t>(src) * total_vecs;
        uint32x4 *dst_out = out_base + static_cast<int64_t>(src) * total_vecs;
        for (int64_t v = v_begin + threadIdx.x; v < v_end; v += AG_THREADS) {
            dst_out[v] = load_nt16_bulk(src_buf + v);
        }
    }
}

// ============================================================================
// Host C++ Entrypoints (bound to registered HeapHandle metadata; zero raw offsets)
// ============================================================================
void fused_rmsnorm_bf16_op(
    const at::Tensor &x,
    const at::Tensor &w,
    at::Tensor &out,
    double eps) {
    TORCH_CHECK(x.is_contiguous() && x.dtype() == at::kBFloat16);
    TORCH_CHECK(w.is_contiguous() && w.dtype() == at::kBFloat16);
    TORCH_CHECK(out.is_contiguous() && out.dtype() == at::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && w.dim() == 1 && out.sizes() == x.sizes());
    TORCH_CHECK(x.device() == w.device() && x.device() == out.device());
    const int M = checked_int32_dim(x.size(0), "M");
    const int D = checked_int32_dim(x.size(1), "D");
    TORCH_CHECK(M >= 1 && (D & 7) == 0 && w.numel() == D);
    check_align(x, 16, "x");
    check_align(w, 16, "w");
    check_align(out, 16, "out");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    xgmi_fused_rmsnorm_bf16_kernel<<<M, 256, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(x.data_ptr()),
        reinterpret_cast<const bf16_t *>(w.data_ptr()),
        reinterpret_cast<bf16_t *>(out.data_ptr()),
        D, static_cast<float>(eps));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void lamport_allgather_op(
    int64_t heap_id,
    const at::Tensor &x_slice,
    at::Tensor &out_a,
    at::Tensor &out_b,
    int64_t split_cols,
    double deadline_s) {
    std::lock_guard<std::mutex> lock(heap_mutex());
    HeapHandle &h = lookup_heap_locked(heap_id, x_slice.device());
    TORCH_CHECK(next_epoch_valid(h.decode_launches),
                "XgmiHeap decode epoch exhausted; recreate XgmiHeap before further launches");
    TORCH_CHECK(x_slice.dim() == 2 && x_slice.is_contiguous() && x_slice.dtype() == at::kBFloat16);
    TORCH_CHECK(out_a.dim() == 2 && out_a.is_contiguous() && out_a.dtype() == at::kBFloat16 && out_a.device() == h.device);
    const int64_t M64 = x_slice.size(0);
    const int64_t slice_cols64 = x_slice.size(1);
    const bool has_out_b = out_b.numel() > 0;
    TORCH_CHECK(decode_ag_dims_valid(M64, slice_cols64, split_cols, has_out_b),
                "lamport_allgather payload exceeds decode AG region");
    const int M = static_cast<int>(M64);
    const int slice_cols = static_cast<int>(slice_cols64);
    bf16_t *b_ptr = nullptr;
    if (has_out_b) {
        TORCH_CHECK(out_a.size(0) == M64 && out_a.size(1) == split_cols * TP);
        TORCH_CHECK(out_b.dim() == 2 && out_b.is_contiguous() && out_b.dtype() == at::kBFloat16 && out_b.device() == h.device);
        TORCH_CHECK(out_b.size(0) == M64 && out_b.size(1) == (slice_cols64 - split_cols) * TP);
        check_align(out_b, 8, "out_b");
        b_ptr = reinterpret_cast<bf16_t *>(out_b.data_ptr());
    } else {
        TORCH_CHECK(out_a.size(0) == M64 && out_a.size(1) == slice_cols64 * TP);
    }
    check_align(x_slice, 8, "x_slice");
    check_align(out_a, 8, "out_a");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(h.device);
    const hipStream_t stream = checked_eager_stream(h.device);
    const uint64_t deadline_ticks = checked_deadline_ticks(deadline_s);
    xgmi_lamport_allgather_kernel<<<DECODE_BLOCKS, DECODE_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(x_slice.data_ptr()),
        reinterpret_cast<bf16_t *>(out_a.data_ptr()),
        b_ptr, M, slice_cols, static_cast<int>(split_cols), h.rank, h.peers,
        DECODE_REGION_OFF, DECODE_HALF_BYTES, DECODE_RS_BYTES,
        h.epoch_ptr, EPOCH_DECODE_OFF, deadline_ticks);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    ++h.decode_launches;
}

void deferred_topk_lamport_ar_op(
    int64_t heap_id,
    const at::Tensor &expert_out,
    const at::Tensor &sorted_idx,
    const at::Tensor &topk_weights,
    const at::Tensor &shared_partial,
    at::Tensor &latent_out,
    at::Tensor &shared_out,
    bool gather_full_shared,
    double deadline_s) {
    std::lock_guard<std::mutex> lock(heap_mutex());
    HeapHandle &h = lookup_heap_locked(heap_id, shared_partial.device());
    TORCH_CHECK(next_epoch_valid(h.decode_launches),
                "XgmiHeap decode epoch exhausted; recreate XgmiHeap before further launches");
    TORCH_CHECK(expert_out.is_contiguous() && expert_out.dtype() == at::kBFloat16 && expert_out.device() == h.device);
    TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.is_contiguous() && topk_weights.dtype() == at::kFloat && topk_weights.device() == h.device);
    TORCH_CHECK(shared_partial.dim() == 2 && shared_partial.is_contiguous() && shared_partial.dtype() == at::kBFloat16);
    TORCH_CHECK(latent_out.dim() == 2 && latent_out.is_contiguous() && latent_out.dtype() == at::kBFloat16 && latent_out.device() == h.device);
    TORCH_CHECK(shared_out.dim() == 2 && shared_out.is_contiguous() && shared_out.dtype() == at::kBFloat16 && shared_out.device() == h.device);
    const int64_t M64 = topk_weights.size(0);
    const int64_t top_k64 = topk_weights.size(1);
    const int64_t D_lat64 = latent_out.size(1);
    const int64_t D_sh64 = shared_partial.size(1);
    TORCH_CHECK(decode_ar_dims_valid(M64, top_k64, D_lat64, D_sh64, gather_full_shared),
                "deferred_topk_lamport_ar dimensions invalid or exceed decode heap regions");
    TORCH_CHECK(shared_partial.size(0) == M64 && latent_out.size(0) == M64 && shared_out.size(0) == M64);
    TORCH_CHECK(shared_out.size(1) == (gather_full_shared ? D_sh64 : D_sh64 / TP));
    const int M = static_cast<int>(M64);
    const int top_k = static_cast<int>(top_k64);
    const int D_lat = static_cast<int>(D_lat64);
    const int D_sh = static_cast<int>(D_sh64);
    const int32_t *sidx_ptr = nullptr;
    int num_expert_rows = 0;
    if (sorted_idx.numel() > 0) {
        TORCH_CHECK(sorted_idx.dim() == 2 && sorted_idx.is_contiguous() && sorted_idx.dtype() == at::kInt &&
                    sorted_idx.size(0) == M && sorted_idx.size(1) == top_k && sorted_idx.device() == h.device);
        TORCH_CHECK(expert_out.dim() == 2 && expert_out.size(1) == D_lat);
        num_expert_rows = checked_int32_dim(expert_out.size(0), "num_expert_rows");
        sidx_ptr = sorted_idx.data_ptr<int32_t>();
    } else {
        TORCH_CHECK(expert_out.dim() == 3 && expert_out.size(0) == M &&
                    expert_out.size(1) == top_k && expert_out.size(2) == D_lat);
        num_expert_rows = M * top_k;
    }
    check_align(expert_out, 8, "expert_out");
    check_align(shared_partial, 8, "shared_partial");
    check_align(latent_out, 8, "latent_out");
    check_align(shared_out, 8, "shared_out");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(h.device);
    const hipStream_t stream = checked_eager_stream(h.device);
    const uint64_t deadline_ticks = checked_deadline_ticks(deadline_s);
    xgmi_deferred_topk_lamport_ar_kernel<<<DECODE_BLOCKS, DECODE_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(expert_out.data_ptr()),
        sidx_ptr, topk_weights.data_ptr<float>(),
        reinterpret_cast<const bf16_t *>(shared_partial.data_ptr()),
        reinterpret_cast<bf16_t *>(latent_out.data_ptr()),
        reinterpret_cast<bf16_t *>(shared_out.data_ptr()),
        M, top_k, num_expert_rows, D_lat, D_sh, gather_full_shared ? 1 : 0,
        h.rank, h.peers, DECODE_REGION_OFF, DECODE_HALF_BYTES, DECODE_RS_BYTES,
        h.epoch_ptr, EPOCH_DECODE_OFF, deadline_ticks);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    ++h.decode_launches;
}

void moetail_column_gather_op(
    int64_t heap_id,
    const at::Tensor &up_slice,
    const at::Tensor &shared_slice,
    const at::Tensor &prefix_slice,
    at::Tensor &stream_out,
    double deadline_s) {
    std::lock_guard<std::mutex> lock(heap_mutex());
    HeapHandle &h = lookup_heap_locked(heap_id, up_slice.device());
    TORCH_CHECK(next_epoch_valid(h.decode_launches),
                "XgmiHeap decode epoch exhausted; recreate XgmiHeap before further launches");
    TORCH_CHECK(up_slice.dim() == 2 && up_slice.is_contiguous() && up_slice.dtype() == at::kBFloat16);
    TORCH_CHECK(shared_slice.dim() == 2 && shared_slice.is_contiguous() && shared_slice.dtype() == at::kBFloat16 && shared_slice.device() == h.device);
    TORCH_CHECK(stream_out.dim() == 2 && stream_out.is_contiguous() && stream_out.dtype() == at::kBFloat16 && stream_out.device() == h.device);
    const int64_t M64 = up_slice.size(0);
    const int64_t slice_cols64 = up_slice.size(1);
    TORCH_CHECK(decode_ag_dims_valid(M64, slice_cols64),
                "moetail_column_gather payload exceeds decode AG region");
    TORCH_CHECK(shared_slice.sizes() == up_slice.sizes());
    TORCH_CHECK(stream_out.size(0) == M64 && stream_out.size(1) == slice_cols64 * TP);
    const int M = static_cast<int>(M64);
    const int slice_cols = static_cast<int>(slice_cols64);
    const bf16_t *pref_ptr = nullptr;
    if (prefix_slice.numel() > 0) {
        TORCH_CHECK(prefix_slice.is_contiguous() && prefix_slice.dtype() == at::kBFloat16 &&
                    prefix_slice.sizes() == up_slice.sizes() && prefix_slice.device() == h.device);
        check_align(prefix_slice, 8, "prefix_slice");
        pref_ptr = reinterpret_cast<const bf16_t *>(prefix_slice.data_ptr());
    }
    check_align(up_slice, 8, "up_slice");
    check_align(shared_slice, 8, "shared_slice");
    check_align(stream_out, 8, "stream_out");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(h.device);
    const hipStream_t stream = checked_eager_stream(h.device);
    const uint64_t deadline_ticks = checked_deadline_ticks(deadline_s);
    xgmi_moetail_column_gather_kernel<<<DECODE_BLOCKS, DECODE_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(up_slice.data_ptr()),
        reinterpret_cast<const bf16_t *>(shared_slice.data_ptr()),
        pref_ptr, reinterpret_cast<bf16_t *>(stream_out.data_ptr()),
        M, slice_cols, h.rank, h.peers, DECODE_REGION_OFF, DECODE_HALF_BYTES, DECODE_RS_BYTES,
        h.epoch_ptr, EPOCH_DECODE_OFF, deadline_ticks);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    ++h.decode_launches;
}

void rs_chunk_32cu_op(
    int64_t heap_id,
    const at::Tensor &partial,
    at::Tensor &out_shard,
    int64_t rows_per_rank,
    int64_t row_off,
    int64_t sub_rows,
    int64_t out_col_off,
    double deadline_s) {
    std::lock_guard<std::mutex> lock(heap_mutex());
    HeapHandle &h = lookup_heap_locked(heap_id, partial.device());
    TORCH_CHECK(next_epoch_valid(h.rs_launches),
                "XgmiHeap RS epoch exhausted; recreate XgmiHeap before further launches");
    TORCH_CHECK(partial.dim() == 2 && partial.is_contiguous() && partial.dtype() == at::kBFloat16);
    TORCH_CHECK(out_shard.dim() == 2 && out_shard.is_contiguous() && out_shard.dtype() == at::kBFloat16 && out_shard.device() == h.device);
    const int64_t cols64 = partial.size(1);
    const int64_t out_stride_cols = out_shard.size(1);
    TORCH_CHECK(rows_per_rank >= 1 && rows_per_rank < (1ll << 24) &&
                    row_off >= 0 && row_off < 65536 &&
                    sub_rows >= 1 && sub_rows < 65536 &&
                    out_col_off >= 0 && out_col_off < 65536 &&
                    cols64 >= 8 && cols64 < 65536,
                "rs_chunk_32cu logical geometry exceeds 16-byte arrival flag range");
    TORCH_CHECK((cols64 & 7) == 0 && (out_col_off & 7) == 0 &&
                    (out_stride_cols & 7) == 0);
    TORCH_CHECK(partial.size(0) == rows_per_rank * TP &&
                    out_shard.size(0) == rows_per_rank);
    TORCH_CHECK(row_off + sub_rows <= rows_per_rank &&
                    out_col_off + cols64 <= out_stride_cols);
    const int cols = static_cast<int>(cols64);
    TORCH_CHECK(static_cast<int64_t>(sub_rows) * TP * cols * 2 <= RS_SLOT_BYTES);
    check_align(partial, 16, "partial");
    check_align(out_shard, 16, "out_shard");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(h.device);
    const hipStream_t stream = checked_eager_stream(h.device);
    const uint64_t deadline_ticks = checked_deadline_ticks(deadline_s);
    xgmi_rs_chunk_32cu_kernel<<<RS_BLOCKS, RS_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(partial.data_ptr()),
        reinterpret_cast<bf16_t *>(out_shard.data_ptr()),
        static_cast<int>(rows_per_rank), static_cast<int>(row_off), static_cast<int>(sub_rows),
        cols, static_cast<int>(out_col_off), out_stride_cols,
        h.rank, h.peers, RS_REGION_OFF, RS_SLOT_BYTES, RS_FLAGS_OFF,
        h.epoch_ptr, EPOCH_RS_OFF, deadline_ticks);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    ++h.rs_launches;
}

void fp4_pack_allgather_op(
    int64_t heap_id,
    const at::Tensor &q_data,
    const at::Tensor &q_scale,
    const at::Tensor &topk_ids,
    const at::Tensor &topk_weights,
    at::Tensor &q_data_all,
    at::Tensor &q_scale_all,
    at::Tensor &topk_ids_all,
    at::Tensor &topk_weights_all,
    double deadline_s) {
    std::lock_guard<std::mutex> lock(heap_mutex());
    HeapHandle &h = lookup_heap_locked(heap_id, q_data.device());
    TORCH_CHECK(next_epoch_valid(h.ag_launches),
                "XgmiHeap AG epoch exhausted; recreate XgmiHeap before further launches");
    TORCH_CHECK(q_data.dim() == 2 && q_scale.dim() == 2 && topk_ids.dim() == 2 && topk_weights.dim() == 2);
    TORCH_CHECK(q_data_all.dim() == 2 && q_scale_all.dim() == 2 && topk_ids_all.dim() == 2 && topk_weights_all.dim() == 2);
    TORCH_CHECK(q_scale.device() == h.device && topk_ids.device() == h.device && topk_weights.device() == h.device &&
                q_data_all.device() == h.device && q_scale_all.device() == h.device &&
                topk_ids_all.device() == h.device && topk_weights_all.device() == h.device);
    TORCH_CHECK(q_data.is_contiguous() && q_data.dtype() == at::kByte &&
                q_scale.is_contiguous() && q_scale.dtype() == at::kByte &&
                topk_ids.is_contiguous() && topk_ids.dtype() == at::kInt &&
                topk_weights.is_contiguous() && topk_weights.dtype() == at::kFloat);
    TORCH_CHECK(q_data_all.is_contiguous() && q_data_all.dtype() == at::kByte &&
                q_scale_all.is_contiguous() && q_scale_all.dtype() == at::kByte &&
                topk_ids_all.is_contiguous() && topk_ids_all.dtype() == at::kInt &&
                topk_weights_all.is_contiguous() && topk_weights_all.dtype() == at::kFloat);
    const int64_t rows64 = q_data.size(0);
    const int64_t q_bytes64 = q_data.size(1);
    const int64_t s_bytes64 = q_scale.size(1);
    const int64_t top_k64 = topk_ids.size(1);
    TORCH_CHECK(fp4_pack_dims_valid(rows64, q_bytes64, s_bytes64, top_k64),
                "fp4_pack_allgather dimensions invalid or exceed AG heap / 16-byte flag range");
    TORCH_CHECK(q_scale.size(0) == rows64 && topk_ids.size(0) == rows64 && topk_weights.sizes() == topk_ids.sizes());
    TORCH_CHECK(q_data_all.size(0) == rows64 * TP && q_data_all.size(1) == q_bytes64 &&
                q_scale_all.size(0) == rows64 * TP && q_scale_all.size(1) == s_bytes64 &&
                topk_ids_all.size(0) == rows64 * TP && topk_ids_all.size(1) == top_k64 &&
                topk_weights_all.size(0) == rows64 * TP && topk_weights_all.size(1) == top_k64);
    const int rows = static_cast<int>(rows64);
    const int q_bytes = static_cast<int>(q_bytes64);
    const int s_bytes = static_cast<int>(s_bytes64);
    const int top_k = static_cast<int>(top_k64);
    check_align(q_data, 16, "q_data");
    check_align(q_scale, 16, "q_scale");
    check_align(topk_ids, 16, "topk_ids");
    check_align(topk_weights, 16, "topk_weights");
    check_align(q_data_all, 16, "q_data_all");
    check_align(q_scale_all, 16, "q_scale_all");
    check_align(topk_ids_all, 16, "topk_ids_all");
    check_align(topk_weights_all, 16, "topk_weights_all");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(h.device);
    const hipStream_t stream = checked_eager_stream(h.device);
    const uint64_t deadline_ticks = checked_deadline_ticks(deadline_s);
    xgmi_fp4_pack_allgather_kernel<<<AG_BLOCKS, AG_THREADS, 0, stream>>>(
        q_data.data_ptr<uint8_t>(), q_scale.data_ptr<uint8_t>(),
        topk_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
        q_data_all.data_ptr<uint8_t>(), q_scale_all.data_ptr<uint8_t>(),
        topk_ids_all.data_ptr<int32_t>(), topk_weights_all.data_ptr<float>(),
        rows, q_bytes >> 4, s_bytes >> 4, (top_k * 4) >> 4,
        h.rank, h.peers, AG_REGION_OFF, AG_HALF_BYTES, AG_FLAGS_OFF,
        h.epoch_ptr, EPOCH_AG_OFF, deadline_ticks);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    ++h.ag_launches;
}

void row_allgather_op(
    int64_t heap_id,
    const at::Tensor &x_shard,
    at::Tensor &out_all,
    double deadline_s) {
    std::lock_guard<std::mutex> lock(heap_mutex());
    HeapHandle &h = lookup_heap_locked(heap_id, x_shard.device());
    TORCH_CHECK(next_epoch_valid(h.ag_launches),
                "XgmiHeap AG epoch exhausted; recreate XgmiHeap before further launches");
    TORCH_CHECK(x_shard.dim() == 2 && x_shard.is_contiguous() && x_shard.dtype() == at::kBFloat16);
    TORCH_CHECK(out_all.dim() == 2 && out_all.is_contiguous() && out_all.dtype() == at::kBFloat16 && out_all.device() == h.device);
    const int64_t rows64 = x_shard.size(0);
    const int64_t cols64 = x_shard.size(1);
    TORCH_CHECK(row_ag_dims_valid(rows64, cols64),
                "row_allgather dimensions invalid or exceed AG heap region");
    TORCH_CHECK(out_all.size(0) == rows64 * TP && out_all.size(1) == cols64);
    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);
    check_align(x_shard, 16, "x_shard");
    check_align(out_all, 16, "out_all");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(h.device);
    const hipStream_t stream = checked_eager_stream(h.device);
    const uint64_t deadline_ticks = checked_deadline_ticks(deadline_s);
    xgmi_row_allgather_kernel<<<AG_BLOCKS, AG_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(x_shard.data_ptr()),
        reinterpret_cast<bf16_t *>(out_all.data_ptr()),
        rows, cols, h.rank, h.peers, AG_REGION_OFF, AG_HALF_BYTES, AG_FLAGS_OFF,
        h.epoch_ptr, EPOCH_AG_OFF, deadline_ticks);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    ++h.ag_launches;
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("create_local_heap(Tensor block_epoch, int heap_bytes) -> (int, Tensor)");
    m.impl("create_local_heap", &create_local_heap_op);

    m.def("open_peer_heaps(int heap_id, Tensor peer_handles_cpu, int rank) -> ()");
    m.impl("open_peer_heaps", &open_peer_heaps_op);

    m.def("unregister_heap(int heap_id) -> ()");
    m.impl("unregister_heap", &unregister_heap_op);

    m.def("fused_rmsnorm_bf16(Tensor x, Tensor w, Tensor(a!) out, float eps) -> ()");
    m.impl("fused_rmsnorm_bf16", at::kCUDA, &fused_rmsnorm_bf16_op);

    m.def("lamport_allgather(int heap_id, Tensor x_slice, Tensor(a!) out_a, Tensor(b!) out_b, "
          "int split_cols, float deadline_s) -> ()");
    m.impl("lamport_allgather", at::kCUDA, &lamport_allgather_op);

    m.def("deferred_topk_lamport_ar(int heap_id, Tensor expert_out, Tensor sorted_idx, "
          "Tensor topk_weights, Tensor shared_partial, Tensor(a!) latent_out, Tensor(b!) shared_out, "
          "bool gather_full_shared, float deadline_s) -> ()");
    m.impl("deferred_topk_lamport_ar", at::kCUDA, &deferred_topk_lamport_ar_op);

    m.def("moetail_column_gather(int heap_id, Tensor up_slice, Tensor shared_slice, "
          "Tensor prefix_slice, Tensor(a!) stream_out, float deadline_s) -> ()");
    m.impl("moetail_column_gather", at::kCUDA, &moetail_column_gather_op);

    m.def("rs_chunk_32cu(int heap_id, Tensor partial, Tensor(a!) out_shard, int rows_per_rank, "
          "int row_off, int sub_rows, int out_col_off, float deadline_s) -> ()");
    m.impl("rs_chunk_32cu", at::kCUDA, &rs_chunk_32cu_op);

    m.def("fp4_pack_allgather(int heap_id, Tensor q_data, Tensor q_scale, Tensor topk_ids, "
          "Tensor topk_weights, Tensor(a!) q_data_all, Tensor(b!) q_scale_all, "
          "Tensor(c!) topk_ids_all, Tensor(d!) topk_weights_all, float deadline_s) -> ()");
    m.impl("fp4_pack_allgather", at::kCUDA, &fp4_pack_allgather_op);

    m.def("row_allgather(int heap_id, Tensor x_shard, Tensor(a!) out_all, float deadline_s) -> ()");
    m.impl("row_allgather", at::kCUDA, &row_allgather_op);
}
