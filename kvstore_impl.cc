// kvstore_impl.cc -- larger-than-memory KV store: slot cache over an append-only record log.
//
// Design (candidate B, v0):
//  * Index: 64 B buckets of 8 x 8 B entries, two-choice placement (partial-key: b2 = b1 ^ f(tag)),
//    plus a class-keyed overflow table. An entry names a *class* (b1, 21-bit tag) and holds the
//    head of that class's chain: a resident slot id or a disk record pointer. Keys of one class are
//    chained (resident entry mode / record.prev), so a blind Upsert never has to read the disk to resolve a tag
//    match: it pushes a new head whose prev is the old head.
//  * Slots: fixed 112 B in-memory records (key, 32-bit control word = seqlock version + size code +
//    flags, 100 B value), updated in place under the seqlock. At most one slot per class, always the
//    class head. The head's chain continuation, or its own clean record D, lives in its index entry.
//  * Log: 64 MiB O_DIRECT segment files; records in 128 B units (a record <= 512 B never straddles a
//    512 B sector), each with a crc32c. Background CLOCK evictors batch victims into 1 MiB appends.
//  * Cold reads: per-session io_uring (raw syscalls), 512 B reads, completed in CompletePending.
//  * Checkpoint: quiesce sessions, append dirty slots, fsync segments + dir, publish a manifest by
//    temp + fsync + rename + dir fsync. Recovery scans the log up to the manifest tail (max seq per key).
#include "kvstore_interface.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <mutex>
#include <sched.h>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#if defined(__linux__) && __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#define KV_HAVE_URING 1
#else
#define KV_HAVE_URING 0
#endif

namespace {

// ───────────────────────────── small helpers ─────────────────────────────
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

inline uint64_t mix64(uint64_t z) {
    z ^= z >> 33;
    z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 33;
    z *= 0xc4ceb9fe1a85ec53ULL;
    z ^= z >> 33;
    return z;
}

// Fatal store-state error with a full message (no errno: the cause is the data, not a syscall).
[[noreturn]] void die_state(const std::string& dir, const char* what) {
    std::fprintf(stderr, "kvstore: FATAL: storage at %s: %s; refusing to open (no files were changed)\n", dir.c_str(),
                 what);
    std::fflush(stderr);
    std::abort();
}

// Exactly "seg_NNNNNN.log" (6 digits): the only names the store creates for log segments.
bool parse_seg_name(const char* n, unsigned* idx) {
    if (std::strlen(n) != 14 || std::memcmp(n, "seg_", 4) != 0 || std::memcmp(n + 10, ".log", 4) != 0) return false;
    unsigned v = 0;
    for (int i = 4; i < 10; i++) {
        if (n[i] < '0' || n[i] > '9') return false;
        v = v * 10 + (unsigned)(n[i] - '0');
    }
    *idx = v;
    return true;
}

[[noreturn]] void die(const char* what, long long a = 0) {
    std::fprintf(stderr, "kvstore: FATAL: %s (%lld, errno %d: %s)\n", what, a, errno, std::strerror(errno));
    std::fflush(stderr);
    std::abort();
}

struct Crc32cTable {
    uint32_t t[256];
    Crc32cTable() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
            t[i] = c;
        }
    }
};
const Crc32cTable kCrcTab;

inline uint32_t crc32c(uint32_t crc, const uint8_t* p, size_t n) {
#if defined(__SSE4_2__)
    uint64_t c = crc;
    while (n >= 8) {
        uint64_t v;
        std::memcpy(&v, p, 8);
        c = __builtin_ia32_crc32di(c, v);
        p += 8;
        n -= 8;
    }
    uint32_t c32 = (uint32_t)c;
    while (n--) c32 = __builtin_ia32_crc32qi(c32, *p++);
    return c32;
#else
    while (n--) crc = kCrcTab.t[(crc ^ *p++) & 0xff] ^ (crc >> 8);
    return crc;
#endif
}

inline uint64_t rd40(const uint8_t* p) {
    uint64_t v = 0;
    std::memcpy(&v, p, 5);
    return v;
}
inline void wr40(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 5); }

void* map_anon(size_t bytes, bool huge) {
    if (bytes == 0) bytes = 4096;
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) die("mmap failed", (long long)bytes);
#ifdef MADV_HUGEPAGE
    if (huge && bytes >= (8u << 20)) ::madvise(p, bytes, MADV_HUGEPAGE);
#else
    (void)huge;
#endif
    return p;
}

uint8_t* aligned_buf(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, (bytes + 4095) & ~size_t(4095)) != 0) die("posix_memalign", (long long)bytes);
    std::memset(p, 0, bytes);
    return (uint8_t*)p;
}

inline uint64_t round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }
inline uint64_t round_dn(uint64_t x, uint64_t a) { return x / a * a; }

// ───────────────────────────── constants ─────────────────────────────
constexpr uint64_t kSegShift = 26;
constexpr uint64_t kSegSize = 1ULL << kSegShift;  // 64 MiB segments
constexpr uint32_t kMaxSeg = 1u << 16;            // 4 TiB of log
constexpr uint64_t kUnit = 128;                   // record unit
constexpr uint64_t kSector = 512;
constexpr uint32_t kHdr = 27;                     // record header bytes
constexpr uint32_t kSlotVal = 100;                // value bytes held inline by a slot
constexpr uint64_t kLogStart = 4096;              // address 0 is the null pointer
constexpr size_t kBatch = 1u << 20;               // eviction / checkpoint append size
constexpr uint32_t kMaxPend = 512;                // async reads in flight per session
// Each cold read is submitted at once: the caller's in-flight reads are capped by its own ring, so
// throughput = in-flight / latency, and a queued-but-unsubmitted read only adds latency (the
// enter costs ~1.5 us of a thread that is otherwise waiting on its ring).
constexpr uint32_t kSubmitBatch = 1;
constexpr uint32_t kStripes = 1u << 14;
constexpr uint32_t kNoSlot = 0xFFFFFFFFu;
constexpr uint64_t kNoInstall = ~0ULL;              // chain hit was not the head record
constexpr uint64_t kHandChunk = 512;              // CLOCK slots claimed per hand step
constexpr size_t kHelpBatch = 64u << 10;          // worker-assisted eviction append size

// Record pointer (40 bits): [0,35) address in 128 B units, [35,39) size code, bit 39 = T ("the
// target's prev field is not part of this chain"). Size code c: 0..8 = c+1 sectors (c = 0: one
// sector of 4 units, or unknown size <= 1 sector, as in a resident entry's A); 12 = a 1-unit
// tombstone; 13..15 = a 1-sector record of c-12 units. The code makes the log bytes a head record
// occupies known without reading it (GC trigger accounting, kvstore 44dd84cc3248).
constexpr uint64_t kPtrT = 1ULL << 39;
constexpr uint64_t kPtrMask = (1ULL << 40) - 1;
constexpr uint32_t kCodeTomb = 12;
inline uint32_t ptr_code(uint64_t p) { return (uint32_t)((p >> 35) & 15); }
inline uint64_t ptr_byte(uint64_t p) { return (p & ((1ULL << 35) - 1)) * kUnit; }
inline uint32_t ptr_nsect(uint64_t p) { uint32_t c = ptr_code(p); return c >= kCodeTomb ? 1 : c + 1; }
inline uint32_t rec_nsect(uint32_t bytes) { return bytes <= kSector ? 1 : (bytes + kSector - 1) / kSector; }
inline uint64_t mk_ptr(uint64_t off, uint32_t bytes, bool tomb = false) {
    uint64_t c;
    if (bytes > kSector) c = rec_nsect(bytes) - 1;
    else if (tomb) c = kCodeTomb;
    else {
        uint32_t u = (bytes + kUnit - 1) / kUnit;  // 1..4
        c = u >= 4 ? 0 : kCodeTomb + u;
    }
    return (off / kUnit) | (c << 35);
}
// Log bytes a record of `bytes` occupies (place(): a single-sector record takes its units, a larger one
// whole sectors). The unit of the GC trigger's live accounting.
inline uint64_t rec_charge(uint32_t bytes) {
    return bytes <= kSector ? (bytes + kUnit - 1) / kUnit * kUnit : (uint64_t)rec_nsect(bytes) * kSector;
}
// Live log bytes of the record p names, from its size code alone: 0 for a tombstone (counted dead when
// written), exact for every code mk_ptr makes, and an upper bound (one sector) for code 0.
inline uint64_t ptr_live_charge(uint64_t p) {
    uint32_t c = ptr_code(p);
    if (c == kCodeTomb) return 0;
    if (c > kCodeTomb) return (uint64_t)(c - kCodeTomb) * kUnit;
    return (uint64_t)(c + 1) * kSector;
}
constexpr uint64_t kMaxRecCharge = ((kHdr + GenValue::kMaxSize + kSector - 1) / kSector) * kSector;

// Index entry. Two forms, told apart by bit 62 (kIsSlot); 0 = empty.
//  * Disk entry: bit63 valid, bit61 secondary bucket, [40,61) 21-bit tag, [0,40) record pointer (0 =
//    no record). A false tag match here would cost a disk read, so the tag stays long.
//  * Resident entry (the class head is a slot): [0,29) A, a record address in 128 B units (1-sector
//    record; 0 = none), [29,61) slot id * 31 + short tag (tag % 31), mode in bits {61, 63}. A short-tag
//    match is confirmed against the slot's key (a DRAM read, never an I/O); the bucket side (kSec) is
//    recomputed from the entry's position when the entry goes back to disk form.
//    Modes (chain = the class continuation after the head slot; it is also the prev of a write-back):
//      kMChain   chain = record A, none if A == 0 (A is the previous class head on disk)
//      kMChainT  chain = record A, T        (A's own prev is not part of the chain)
//      kMD       A = the head key's own record D; chain = D's on-disk prev (starts at D)
//      kMDSolo   A = D, no chain
//    A slot without kDirty is always in a D mode and equals its record D exactly (clean): eviction
//    points the index back at D with no write. Pointers that do not fit A (record > 512 B, or log
//    beyond 64 GiB) are never put in an entry: such writes go straight to the log, such records stay
//    dirty.
constexpr uint64_t kValid = 1ULL << 63;
constexpr uint64_t kIsSlot = 1ULL << 62;
constexpr uint64_t kSec = 1ULL << 61;
constexpr int kTagShift = 40;
constexpr uint64_t kTagMask = (1ULL << 21) - 1;
constexpr uint64_t kPay = (1ULL << 40) - 1;
constexpr uint64_t kHdrBits = kValid | kSec | (kTagMask << kTagShift);  // disk-entry header
constexpr int kAddrBits = 29, kSidShift = 29;
// Bits [29,61) hold one 32-bit mixed-radix field: slot id * kNT + short tag (tag mod kNT). With
// kNT = 31 the slot-id cap is floor(2^32 / 31) ~ 138.5M (vs 2^27 for 27 + 5 split bits) at the same
// 1/31 short-tag false-match rate; extracting both is a constant division (multiply + shift).
constexpr uint32_t kNT = 31;
constexpr uint64_t kAddrMask = (1ULL << kAddrBits) - 1;
constexpr uint64_t kSidMask = 0xFFFFFFFFULL << kSidShift;  // the whole (slot id, short tag) field
constexpr uint64_t kMaxSlots = 0xFFFFFFFFULL / kNT;           // ids [0, kMaxSlots): field < 2^32
static_assert(kAddrBits <= kSidShift && kSidShift + 32 <= 61, "resident entry layout");
enum : uint32_t { kMChain = 0, kMChainT = 1, kMD = 2, kMDSolo = 3 };
inline uint32_t res_field(uint64_t e) { return (uint32_t)(e >> kSidShift); }
inline uint32_t res_slot(uint64_t e) { return res_field(e) / kNT; }
inline uint64_t res_addr(uint64_t e) { return e & kAddrMask; }
inline uint32_t tag5(uint64_t tag21) { return (uint32_t)(tag21 & kTagMask) % kNT; }  // short tag
inline uint32_t res_tag5(uint64_t e) { return res_field(e) % kNT; }
inline uint32_t res_mode(uint64_t e) { return (uint32_t)((e >> 61) & 1) | (uint32_t)((e >> 62) & 2); }
inline uint64_t mk_res(uint32_t mode, uint32_t si, uint64_t addr, uint32_t t5) {
    return kIsSlot | ((uint64_t)(mode & 1) << 61) | ((uint64_t)(mode >> 1) << 63) |
           ((uint64_t)(si * kNT + t5) << kSidShift) | (addr & kAddrMask);
}
inline uint64_t res_with_slot(uint64_t e, uint32_t si) {
    return (e & ~kSidMask) | ((uint64_t)(si * kNT + res_tag5(e)) << kSidShift);
}
// A record pointer fits A if it names a 1-sector record below 64 GiB.
inline bool ptr_fits(uint64_t p) { return ptr_nsect(p) == 1 && (p & ((1ULL << 35) - 1)) <= kAddrMask; }
// Mode of a fresh head pushed over disk payload p (requires p == 0 or ptr_fits(p)).
inline uint32_t push_mode(uint64_t p) { return (p & kPtrT) ? kMChainT : kMChain; }
// Chain continuation of a resident entry (record pointer, 0 = none): the prev of its write-back.
inline uint64_t chain_start(uint64_t e) {
    switch (res_mode(e)) {
        case kMChain: case kMD: return res_addr(e);
        case kMChainT: return res_addr(e) | kPtrT;
        default: return 0;  // kMDSolo
    }
}
inline bool res_has_d(uint64_t e) { uint32_t m = res_mode(e); return m == kMD || m == kMDSolo; }

// Slot concurrency state lives outside the 108 B slot (kvstore iteration 13):
//  * Side lines: one 64 B line holds the kSideLocks (4) stripe lock words and the flag bytes of
//    kSideSlots (48) consecutive slots, so a slot's lock and flags cost one cache miss together and a
//    sequential sweep reads one line per 48 slots.
//  * Lock + version: a striped seqlock (one 32-bit word per 12 consecutive slots): bit0 = lock, [1,32)
//    version (+1 per modification of any slot of the stripe, wrapping). A reader validates its slot's
//    stripe word around the snapshot; a writer holds the stripe lock. No code path holds two stripe
//    locks: publish (the only lock once taken while another was held) is lock-free.
//  * Flags: one side byte per slot: live, dirty, tomb, the value-size kind (short / big) and a 2-bit
//    CLOCK counter (inserts start at 1, accesses set 3 by fetch_or, the sweep decrements by CAS, a victim
//    is at 0). live/dirty/tomb/size change only under the stripe lock (CAS loop, so a racing ref update
//    or sweep decrement is never lost into them); ref bits change lock-free.
//  * Value size: kind 0 = a 100 B inline value; kShort = inline value of < 100 B whose size is the last
//    value byte (byte 99, the top byte of word 26); kBig = a large value whose size is in the first
//    continuation unit's word 1.
//  * Write-behind validation is per slot, not by version (a shared stripe would abort commits for
//    other slots' writes): an encoder that writes a slot behind and commits later (evictor, cleaner,
//    tombstone reclaim) takes the slot's kWbI bit exclusively at encode (a slot with kWbI set is
//    skipped by every other such encoder) and clears kWbM; every modification of the slot (unlock_mod)
//    sets kWbM while kWbI is set; the owner's commit requires kWbI && !kWbM (no modification since
//    its encode) plus unchanged content and entry, and clears both bits whatever the outcome. Frees
//    and fills keep both bits, so a recycled slot stays owned until the stale commit fails.
// Version wraparound: every optimistic reader bounds its own read window (kReadWindowTicks, far below
// the time 2^31 modifications take). A shared stripe only adds spurious version changes, which make
// a reader retry.
constexpr uint32_t kLockBit = 1;
constexpr uint32_t kVerMask = ~1u;
// Flag byte: live, dirty, the value kind (2 bits: 100 B inline / short / big / tombstone), the
// write-behind pair (kWbI, kWbM) and the CLOCK counter.
constexpr uint32_t kLive = 1u, kDirty = 2u, kWbI = 4u, kWbM = 32u;
constexpr uint32_t kShort = 8u, kBig = 16u, kTomb = 24u, kKind = 24u;
constexpr uint32_t kRefMask = 3u << 6, kRef1 = 1u << 6, kRefMax = 3u << 6;
constexpr uint32_t kFlagMask = kLive | kDirty | kRefMask;
inline bool is_clean(uint32_t w) { return !(w & kDirty); }
inline bool is_tomb(uint32_t w) { return (w & kKind) == kTomb; }
inline uint32_t size_bits(uint32_t size) { return size > 100 ? kBig : size == 100 ? 0 : kShort; }  // value kind

// A slot is one 108 B unit of the arena: 27 32-bit words (odd slots start at 4 mod 8, so every
// optimistic access is a 4 B relaxed atomic; no 8 B atomic and no RMW ever touches slot bytes).
// Head unit: words 0-1 = key (lo, hi), words 2-26 = value bytes [0,100). A value of <= kSlotVal bytes
// lives inline (size < 100: its size is value byte 99). A larger value keeps bytes [0,96) in words
// 3-26 with word 2 = the first continuation unit. A continuation unit holds the next unit id in word
// 0, the value size in word 1 (first continuation only) and kContVal bytes in words 2-26. Writers hold
// the head's stripe lock or own the unit (unpublished).
constexpr uint32_t kBigHead = 96;
constexpr uint32_t kContVal = 100;
constexpr uint32_t kMaxUnits = 1 + (GenValue::kMaxSize - kBigHead + kContVal - 1) / kContVal;  // 41
constexpr uint32_t kSlotWords = 27;
struct Slot {
    uint32_t w[kSlotWords];
};
static_assert(sizeof(Slot) == 108, "slot layout");
static_assert(alignof(Slot) == 4, "slot layout");
constexpr uint32_t kSideSlots = 48, kSideLocks = 4;
struct alignas(64) SideLine {
    std::atomic<uint32_t> lk[kSideLocks];  // stripe seqlocks: slots [12k, 12k+12) of the line
    std::atomic<uint8_t> fl[kSideSlots];   // flag bytes
};
static_assert(sizeof(SideLine) == 64, "side line layout");

template <class T>
inline T ald(const T* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }
template <class T>
inline void ast(T* p, T v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }

// Read-window clock for optimistic readers. A modification takes >= ~20 cycles (a locked CAS plus a
// store), so 2^19 of them take >= 10M cycles; a read window under 2^20 ticks cannot see a wrap.
#if defined(__x86_64__) || defined(__i386__)
// No fences: rdtsc may drift from the loads it brackets by at most the reorder window (a few us),
// far below the 2^20-tick bound and the >= 10M cycles a wrap takes.
inline uint64_t win_begin() { return __builtin_ia32_rdtsc(); }
inline uint64_t win_end() { return __builtin_ia32_rdtsc(); }
constexpr uint64_t kReadWindowTicks = 1ULL << 20;
#else
inline uint64_t win_begin() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline uint64_t win_end() { return win_begin(); }
constexpr uint64_t kReadWindowTicks = 100000;  // 100 us
#endif

inline uint32_t units_for(uint32_t size) {
    return size <= kSlotVal ? 1 : 1 + (size - kBigHead + kContVal - 1) / kContVal;
}
inline uint64_t slot_key(const Slot* s) { return (uint64_t)ald(&s->w[0]) | ((uint64_t)ald(&s->w[1]) << 32); }
inline void set_slot_key(Slot* s, uint64_t k) {
    ast(&s->w[0], (uint32_t)k);
    ast(&s->w[1], (uint32_t)(k >> 32));
}
inline uint32_t* cont_words(Slot* c) { return &c->w[2]; }
inline const uint32_t* cont_words(const Slot* c) { return &c->w[2]; }
inline uint32_t ld_part(const uint8_t* p, uint32_t n) {
    uint32_t w = 0;
    std::memcpy(&w, p, n);
    return w;
}
// n bytes of src -> 4 B words at dst (relaxed atomic stores; the last word zero-padded).
inline void store_words(uint32_t* dst, const uint8_t* src, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32_t w;
        std::memcpy(&w, src + i, 4);
        ast(dst + i / 4, w);
    }
    if (i < n) ast(dst + i / 4, ld_part(src + i, n - i));
}
// ceil4(n) bytes from 4 B words at src -> dst (relaxed atomic loads).
inline void load_words(const uint32_t* src, uint8_t* dst, uint32_t n) {
    for (uint32_t i = 0; i < n; i += 4) {
        uint32_t w = ald(src + i / 4);
        std::memcpy(dst + i, &w, 4);
    }
}

// ───────────────────────────── records ─────────────────────────────
// [0,8) key | [8,16) seq(56) | magic(8) | [16,20) crc32c | [20,22) size | [22,27) prev | value
struct RecView {
    uint64_t key, seq, prev;
    uint32_t size;
    bool tomb;
    const uint8_t* val;
};

inline uint32_t enc_record(uint8_t* d, uint64_t key, uint64_t seq, bool tomb, uint64_t prev, const uint8_t* val,
                           uint32_t size) {
    uint64_t sf = (seq & ((1ULL << 56) - 1)) | ((uint64_t)(0xA0u | (tomb ? 1u : 0u)) << 56);
    std::memcpy(d, &key, 8);
    std::memcpy(d + 8, &sf, 8);
    uint16_t sz = (uint16_t)size;
    std::memcpy(d + 20, &sz, 2);
    wr40(d + 22, prev & kPtrMask);
    if (size) std::memcpy(d + kHdr, val, size);
    uint32_t c = crc32c(0xFFFFFFFFu, d, 16);
    c = crc32c(c, d + 20, 7 + size);
    std::memcpy(d + 16, &c, 4);
    return kHdr + size;
}

inline bool parse_record(const uint8_t* p, size_t avail, RecView& r) {
    if (avail < kHdr) return false;
    uint64_t sf;
    std::memcpy(&sf, p + 8, 8);
    uint32_t magic = (uint32_t)(sf >> 56);
    if ((magic & 0xFE) != 0xA0) return false;
    uint16_t sz;
    std::memcpy(&sz, p + 20, 2);
    if (sz > GenValue::kMaxSize || kHdr + (size_t)sz > avail) return false;
    uint32_t c = crc32c(0xFFFFFFFFu, p, 16);
    c = crc32c(c, p + 20, 7 + sz);
    uint32_t stored;
    std::memcpy(&stored, p + 16, 4);
    if (c != stored) return false;
    std::memcpy(&r.key, p, 8);
    r.seq = sf & ((1ULL << 56) - 1);
    r.tomb = magic & 1;
    r.size = sz;
    r.prev = rd40(p + 22);
    r.val = p + kHdr;
    return true;
}

// Where a record of `bytes` goes in an append buffer at position `pos`.
inline uint64_t place(uint64_t pos, uint32_t bytes) {
    uint64_t span = round_up(bytes, kUnit);
    if (bytes <= kSector) {
        if ((pos % kSector) + span > kSector) pos = round_up(pos, kSector);
    } else {
        pos = round_up(pos, kSector);
    }
    return pos;
}

// Digest term of the record at segment offset so with stored crc32c `crc`.
inline uint64_t rec_dig(uint32_t crc, uint64_t so) { return mix64(((uint64_t)crc << 32) ^ (so / kUnit) ^ 0x5EC0D16E57ULL); }
// Records of an append buffer buf[0, len) written at segment offset so: the same walk scan_seg does
// (a unit that starts a well-formed header is a record, anything else a padding unit); the buffer
// holds only records we encoded, so the stored crc is taken as is.
inline void buf_digest(const uint8_t* buf, uint64_t len, uint64_t so, uint64_t& cnt, uint64_t& dig) {
    uint64_t u = 0;
    while (u + kHdr <= len) {
        const uint8_t* p = buf + u;
        if ((p[15] & 0xFE) == 0xA0) {
            uint16_t sz;
            std::memcpy(&sz, p + 20, 2);
            if (sz <= GenValue::kMaxSize && u + kHdr + sz <= len) {
                uint32_t c;
                std::memcpy(&c, p + 16, 4);
                cnt++;
                dig += rec_dig(c, so + u);
                u += round_up(kHdr + (uint64_t)sz, kUnit);
                continue;
            }
        }
        u += kUnit;
    }
}

// ───────────────────────────── io_uring ─────────────────────────────
#if KV_HAVE_URING
struct Ring {
    int fd = -1;
    unsigned *sq_head = nullptr, *sq_tail = nullptr, *sq_mask = nullptr, *sq_array = nullptr;
    unsigned *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
    io_uring_sqe* sqes = nullptr;
    io_uring_cqe* cqes = nullptr;
    void* sq_map = nullptr;
    void* cq_map = nullptr;
    size_t sq_len = 0, cq_len = 0, sqe_len = 0;
    bool init(unsigned entries) {
        io_uring_params p{};
        fd = (int)syscall(__NR_io_uring_setup, entries, &p);
        if (fd < 0) return false;
        sq_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_len = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        bool single = p.features & IORING_FEAT_SINGLE_MMAP;
        if (single) sq_len = cq_len = std::max(sq_len, cq_len);
        sq_map = mmap(nullptr, sq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq_map == MAP_FAILED) { sq_map = nullptr; close(); return false; }
        cq_map = single ? sq_map
                        : mmap(nullptr, cq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (cq_map == MAP_FAILED) { cq_map = nullptr; close(); return false; }
        sqe_len = p.sq_entries * sizeof(io_uring_sqe);
        sqes = (io_uring_sqe*)mmap(nullptr, sqe_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                                   IORING_OFF_SQES);
        if (sqes == MAP_FAILED) { sqes = nullptr; close(); return false; }
        char* sq = (char*)sq_map;
        char* cq = (char*)cq_map;
        sq_head = (unsigned*)(sq + p.sq_off.head);
        sq_tail = (unsigned*)(sq + p.sq_off.tail);
        sq_mask = (unsigned*)(sq + p.sq_off.ring_mask);
        sq_array = (unsigned*)(sq + p.sq_off.array);
        cq_head = (unsigned*)(cq + p.cq_off.head);
        cq_tail = (unsigned*)(cq + p.cq_off.tail);
        cq_mask = (unsigned*)(cq + p.cq_off.ring_mask);
        cqes = (io_uring_cqe*)(cq + p.cq_off.cqes);
        return true;
    }
    void prep_read(int f, void* buf, unsigned len, uint64_t off, uint64_t ud) {
        unsigned t = *sq_tail, i = t & *sq_mask;
        io_uring_sqe* e = &sqes[i];
        std::memset(e, 0, sizeof *e);
        e->opcode = IORING_OP_READ;
        e->fd = f;
        e->addr = (uint64_t)buf;
        e->len = len;
        e->off = off;
        e->user_data = ud;
        sq_array[i] = i;
        __atomic_store_n(sq_tail, t + 1, __ATOMIC_RELEASE);
    }
    int enter(unsigned submit, unsigned wait) {
        for (;;) {
            int r = (int)syscall(__NR_io_uring_enter, fd, submit, wait, wait ? IORING_ENTER_GETEVENTS : 0, nullptr, 0);
            if (r < 0 && (errno == EINTR || errno == EAGAIN || errno == EBUSY)) {
                if (errno != EINTR) std::this_thread::yield();
                continue;
            }
            return r;
        }
    }
    template <class F>
    unsigned reap(F f) {
        unsigned h = *cq_head, t = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE), n = 0;
        while (h != t) {
            io_uring_cqe c = cqes[h & *cq_mask];
            h++;
            n++;
            __atomic_store_n(cq_head, h, __ATOMIC_RELEASE);
            f(c);
            t = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
        }
        return n;
    }
    void close() {
        if (sqes) munmap(sqes, sqe_len);
        if (cq_map && cq_map != sq_map) munmap(cq_map, cq_len);
        if (sq_map) munmap(sq_map, sq_len);
        sqes = nullptr;
        sq_map = cq_map = nullptr;
        if (fd >= 0) ::close(fd);
        fd = -1;
    }
};
#endif

// ───────────────────────────── per-thread state ─────────────────────────────
// Single-writer counter (its session's thread), read by stats from any thread: relaxed atomic
// load + store, no read-modify-write on the hot path.
struct Ctr {
    uint64_t v = 0;
    void operator++(int) { ast(&v, ald(&v) + 1); }
    void operator+=(uint64_t n) { ast(&v, ald(&v) + n); }
    operator uint64_t() const { return ald(&v); }
};
struct Counters {
    Ctr read_mem, read_disk_found, read_notfound, ios, hops, installs;
    Ctr ups_inplace, ups_push, slow_writes, rmw_hit, rmw_miss, slot_waits, help_evicts;
    Ctr depth_sum, enters;  // in-flight reads summed at each submit; io_uring_enter calls
    Ctr reaped;             // reads finished (ios - reaped = reads in flight)
    Ctr deletes;
    Ctr live_sub;  // estimated log bytes that stopped being current (GC trigger accounting)
    // Run-window cold-read latency, log2(ns) buckets: submit -> reap, and the reap lag (an upper bound
    // on CQE-posted -> reaped: the time since this session last drained its CQ). done is set at reap.
    static constexpr int kHB = 40;
    Ctr lat_n, lat_sum_ns, lag_sum_ns, lat_h[kHB], lag_h[kHB];
};

struct Pend {
    int64_t sidx;                  // re-read sampler slot of the key (-1: not sampled)
    ReadSlot* rs;
    uint64_t key;
    uint64_t p;                    // record pointer being read
    std::atomic<uint64_t>* ep;     // class entry
    uint64_t head_e;               // entry value when p was the head record (0: no install)
    uint64_t base;                 // logical byte offset of the read
    uint32_t len;
    uint8_t* buf;
    bool heap;
    int64_t t_sub;                 // steady ns at submit (latency histogram)
    uint64_t gen;                  // gc_gen_ when p was resolved (a GC since: p's chain is gone)
};

struct Victim {
    uint32_t si, boff, bytes;
    std::atomic<uint64_t>* ep;
    uint64_t e;      // resident entry when encoded
    uint64_t chain;  // prev written into the record
    uint64_t hdr;    // disk-entry header of the class at ep
    uint32_t ver;    // stripe version when encoded (diagnostic; commits validate with kWbI/kWbM)
};

// Test seam: KVSTORE_TEST_COMMIT_STALL_US stalls the evictor's and cleaner's commits (after their
// batch write, before re-validating the slots) to widen the encode->commit window. Read once.
static uint32_t commit_stall_us() {
    static const uint32_t us = [] {
        const char* v = std::getenv("KVSTORE_TEST_COMMIT_STALL_US");
        return v ? (uint32_t)std::strtoul(v, nullptr, 10) : 0u;
    }();
    return us;
}
static inline void commit_stall() {
    if (uint32_t us = commit_stall_us()) std::this_thread::sleep_for(std::chrono::microseconds(us));
}

// Write-behind context (a cleaner thread, or the stopping thread for a small pre-run drain).
struct CleanCtx {
    std::vector<Victim> vic;
    uint8_t* buf = nullptr;
    uint64_t pos = 0;
    alignas(8) uint8_t tmp[GenValue::kMaxSize + 8];
};

struct Session {
    std::atomic<uint32_t> in_op{0};
    char pad0[60];
    std::vector<uint32_t> fcache;  // free slots owned by this thread
#if KV_HAVE_URING
    Ring ring;
#endif
    bool ring_ok = false, ring_tried = false;
    uint8_t* bufs = nullptr;  // kMaxPend * rd_unit
    std::vector<Pend> pend;
    std::vector<uint32_t> pfree;
    uint32_t queued = 0, npend = 0;
    int64_t last_poll_ns = 0;  // when this session last drained its CQ
    uint8_t* sbuf = nullptr;  // sync read buffer
    uint8_t* wbuf = nullptr;  // sync record write buffer
    GenValue* tmpv = nullptr;
    alignas(8) uint8_t gbuf[GenValue::kMaxSize + 8];  // value gather buffer (lock holder)
    uint32_t ids[kMaxUnits];                         // units of the chain being written
    uint32_t oids[kMaxUnits];                        // units of the chain being replaced
    uint8_t* ebuf = nullptr;  // worker-assisted eviction buffer
    std::vector<Victim> evic;
    std::vector<uint32_t> efreed;
    Counters c;
    std::vector<uint32_t> tpend;  // resident tombstone heads with no chain, awaiting reclaim (Delete)
    uint8_t* tbuf = nullptr;      // their append buffer (allocated on the first batch)
};

struct TLS {
    uint64_t owner = 0;
    Session* s = nullptr;
};
thread_local TLS tls;
std::atomic<uint64_t> g_instance_ids{1};

struct OvTable {
    std::atomic<uint64_t>* cls = nullptr;
    std::atomic<uint64_t>* ent = nullptr;
    uint64_t mask = 0, count = 0;
};

struct Cls {
    uint64_t b1, b2, id;
    uint32_t tag;
};

struct Manifest {  // version 1 (read only): the log is the contiguous range [kLogStart, tail)
    uint64_t magic, version, tail, next_seq, nb, reserved;
    uint32_t align, crc;
};
constexpr uint64_t kManifestMagic = 0x4b56534c4f544231ULL;  // "KVSLOTB1"
// Version 2 (written): the checkpointed log is the set of segments in `live` (a bitmap; segment
// numbers are reused after garbage collection frees them), each read whole, except the active
// segment, read up to active_off. Records past active_off, and segment files outside the set,
// are post-checkpoint writes (or freed segments) and are dropped by recovery.
constexpr uint32_t kSegWords = kMaxSeg / 64;
struct ManifestV2 {
    uint64_t magic, version, active_seg, active_off, next_seq, nb;
    uint32_t align, pad;
    uint64_t live[kSegWords];
    uint32_t crc, pad2;
};
struct ManifestSeg {
    uint64_t seg, cnt, dig;
};
inline uint64_t seg_start(uint64_t seg) { return seg == 0 ? kLogStart : 0; }  // address 0 = null

// ───────────────────────────── the store ─────────────────────────────
class SlotStore final : public IKVStore {
public:
    SlotStore() : id_(g_instance_ids.fetch_add(1)) {
        for (uint32_t i = 0; i < kMaxSeg; i++) {
            fds_[i].store(-1, std::memory_order_relaxed);
            segdirty_[i].store(0, std::memory_order_relaxed);
            seg_cnt_[i].store(0, std::memory_order_relaxed);
            seg_dig_[i].store(0, std::memory_order_relaxed);
        }
    }
    ~SlotStore() override;

    void Init(size_t hash_table_size, size_t log_size_bytes) override {
        InitExtended(hash_table_size, log_size_bytes, 0, nullptr);
    }
    void InitExtended(size_t hash_table_size, size_t log_size_bytes, size_t mem_budget_bytes,
                      const char* storage_path) override;
    void StartSession() override { (void)sess(); }
    void StopSession() override;
    void Refresh() override {}
    bool Read(uint64_t key, GenValue& out) override;
    OpStatus ReadAsync(ReadSlot* slot) override;
    void CompletePending(bool wait) override;
    void Upsert(uint64_t key, const GenValue& value) override;
    void RMW(uint64_t key, const uint8_t* mod_data, size_t mod_size) override;
    bool Delete(uint64_t key) override;
    void Checkpoint() override;
    CacheStats GetCacheStats() const override;

private:
    // ── sessions / quiesce
    Session* sess() {
        if (tls.owner == id_) return tls.s;
        return sess_slow();
    }
    Session* sess_slow();
    struct OpGuard {
        Session* s;
        OpGuard(SlotStore* kv, Session* ss) : s(ss) {
            for (;;) {
                s->in_op.store(1, std::memory_order_seq_cst);
                if (!kv->ckpt_flag_.load(std::memory_order_seq_cst)) break;
                s->in_op.store(0, std::memory_order_seq_cst);
                while (kv->ckpt_flag_.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        }
        ~OpGuard() { s->in_op.store(0, std::memory_order_release); }
    };

    // ── index
    Cls classify(uint64_t key) const {
        uint64_t h = mix64(key);
        Cls c;
        c.b1 = h & bmask_;
        c.tag = (uint32_t)(h >> 43) & (uint32_t)kTagMask;
        uint64_t x = mix64((uint64_t)c.tag * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL) & bmask_;
        c.b2 = c.b1 ^ (x ? x : 1);
        c.id = (c.b1 << 21) | c.tag;
        return c;
    }
    std::atomic<uint64_t>* find_entry(const Cls& c) const;
    int res_class_is(const std::atomic<uint64_t>* ep, uint64_t x, const Cls& c) const;
    // Start the slot reads of a bucket's short-tag candidates together (their misses overlap).
    void prefetch_candidates(const std::atomic<uint64_t>* B, int n, uint32_t t5) const {
        for (int i = 0; i < n; i++) {
            uint64_t x = B[i].load(std::memory_order_relaxed);
            if ((x & kIsSlot) && res_tag5(x) == t5) {
                const uint32_t si = res_slot(x);
                const uint8_t* p = arena_ + (uint64_t)si * sizeof(Slot);
                __builtin_prefetch(p);
                __builtin_prefetch(p + sizeof(Slot) - 1);
                uint32_t pos;
                __builtin_prefetch(&side_line(si, pos));
            }
        }
    }
    std::atomic<uint64_t>* find_own_entry(uint64_t key, uint32_t si, Cls* cout) const;
    uint64_t disk_hdr(const std::atomic<uint64_t>* ep, const Cls& c) const {
        uintptr_t d = (uintptr_t)ep - (uintptr_t)bk_;  // wraps huge when ep is below bk_
        uint64_t i = d / sizeof(*bk_);
        bool in_bk = d < nb_ * 64;
        return kValid | ((uint64_t)c.tag << kTagShift) | ((in_bk && (i >> 3) == c.b2) ? kSec : 0);
    }
    bool in_bk(const std::atomic<uint64_t>* ep) const { return (uintptr_t)ep - (uintptr_t)bk_ < nb_ * 64; }
    // Entry reclaim: the evictor frees a bucket entry (stores 0) when it evicts a tombstone head with
    // no chain (its class holds only that deleted key; the tombstone is in the log). The position may
    // then be taken by another class, so a writer that loaded disk-form e from an ep it found earlier
    // checks that e is still its class's entry before pushing onto it (bucket + side + tag identify
    // the class; resident entries are confirmed by the slot key as before).
    bool stale_disk(const std::atomic<uint64_t>* ep, uint64_t e, const Cls& c) const {
        return !(e & kIsSlot) && (e & kHdrBits) != disk_hdr(ep, c);
    }
    std::atomic<uint64_t>* create_entry(const Cls& c);
    std::atomic<uint64_t>* ov_find(uint64_t id) const;
    std::atomic<uint64_t>* ov_insert(uint64_t id, uint64_t e, uint64_t b1);
    std::atomic<uint64_t>* entry_for(const Cls& c) {
        std::atomic<uint64_t>* ep = find_entry(c);
        return ep ? ep : create_entry(c);
    }

    // ── slots
    Slot* slot(uint64_t i) const { return reinterpret_cast<Slot*>(arena_ + i * sizeof(Slot)); }
    // A stripe serves 12 consecutive slots: sharing costs only reader retries and lock waits (write-
    // behind commits validate per slot, kWbI/kWbM).
    // Writer: start the slot's line fetches (for ownership) before taking the stripe lock, so the
    // critical section does not wait on them (the lock lives in a side line, not the slot).
    void prefetch_w(const Slot* s) const {
        __builtin_prefetch(s, 1);
        __builtin_prefetch((const uint8_t*)s + sizeof(Slot) - 1, 1);
        __builtin_prefetch(&lockw(s), 1);  // the side line (lock + flag byte), for ownership
    }
    uint32_t sid(const Slot* s) const { return (uint32_t)(((const uint8_t*)s - arena_) / sizeof(Slot)); }
    SideLine& side_line(uint32_t si, uint32_t& pos) const {
        const uint32_t l = si / kSideSlots;
        pos = si - l * kSideSlots;
        return side_[l];
    }
    std::atomic<uint32_t>& lockw(uint32_t si) const {
        uint32_t pos;
        SideLine& l = side_line(si, pos);
        return l.lk[pos / (kSideSlots / kSideLocks)];
    }
    std::atomic<uint32_t>& lockw(const Slot* s) const { return lockw(sid(s)); }
    std::atomic<uint8_t>& flg(uint32_t si) const {
        uint32_t pos;
        SideLine& l = side_line(si, pos);
        return l.fl[pos];
    }
    // Seqlock writer side: the release fence after taking the lock keeps the data stores that follow
    // from becoming visible before the locked word (readers validate with an acquire fence).
    uint32_t lock_slot(const Slot* s) const {
        std::atomic<uint32_t>& L = lockw(s);
        for (int n = 0;; n++) {
            uint32_t v = L.load(std::memory_order_relaxed);
            if (!(v & kLockBit) &&
                L.compare_exchange_weak(v, v | kLockBit, std::memory_order_acquire, std::memory_order_relaxed)) {
                std::atomic_thread_fence(std::memory_order_release);
                return v;
            }
            if (n > 64) std::this_thread::yield(); else cpu_relax();
        }
    }
    bool try_lock_slot(const Slot* s, uint32_t& v) const {
        std::atomic<uint32_t>& L = lockw(s);
        v = L.load(std::memory_order_relaxed);
        if (v & kLockBit) return false;
        if (!L.compare_exchange_strong(v, v | kLockBit, std::memory_order_acquire, std::memory_order_relaxed))
            return false;
        std::atomic_thread_fence(std::memory_order_release);
        return true;
    }
    // Lock holder: replace the flag bits in `mask` (a CAS loop: a racing ref update or sweep decrement
    // never overwrites them, and they never overwrite it). No store when the bits already match, so
    // repeated overwrites of a hot key leave its flag line shared in the readers' caches.
    void set_ctl(const Slot* s, uint32_t mask, uint32_t bits) const {
        std::atomic<uint8_t>& f = flg(sid(s));
        uint8_t w = f.load(std::memory_order_relaxed);
        while ((w & mask) != bits &&
               !f.compare_exchange_weak(w, (uint8_t)((w & ~mask) | bits), std::memory_order_relaxed)) {
        }
    }
    uint32_t ctl(const Slot* s) const { return flg(sid(s)).load(std::memory_order_relaxed); }
    // Unlock after a modification: the stripe version advanced (wrapping).
    // A write-behind in flight on the slot (kWbI) learns of the modification (kWbM).
    void unlock_mod(const Slot* s, uint32_t v) const {
        std::atomic<uint8_t>& f = flg(sid(s));
        const uint8_t w = f.load(std::memory_order_relaxed);
        if ((w & (kWbI | kWbM)) == kWbI) f.fetch_or((uint8_t)kWbM, std::memory_order_relaxed);
        lockw(s).store((v + 2) & kVerMask, std::memory_order_release);
    }
    // Write-behind ownership (lock holder): take it at encode (false: another encoder owns the slot),
    // test it at commit, and release it after the commit attempt.
    bool wb_begin(const Slot* s) const {
        if (ctl(s) & kWbI) return false;
        set_ctl(s, kWbI | kWbM, kWbI);
        return true;
    }
    bool wb_unmodified(const Slot* s) const { return (ctl(s) & (kWbI | kWbM)) == kWbI; }
    void wb_end(const Slot* s) const { set_ctl(s, kWbI | kWbM, 0); }
    void unlock_nomod(const Slot* s, uint32_t v) const { lockw(s).store(v, std::memory_order_release); }
    void mark_dirty(uint32_t si) {
        std::atomic<uint8_t>& m = dmark_[si / kHandChunk];
        if (!m.load(std::memory_order_relaxed)) m.store(1, std::memory_order_relaxed);
    }
    // Value size of head s whose flags are w; > kMaxSize if a (torn) size or large-value chain is bad.
    uint32_t value_size(const Slot* s, uint32_t w) const {
        const uint32_t k = w & kKind;
        if (k == kTomb) return 0;
        if (k == 0) return kSlotVal;
        if (k == kShort) {
            uint32_t b = ald(&s->w[kSlotWords - 1]) >> 24;
            return b < kSlotVal ? b : GenValue::kMaxSize + 1;
        }
        uint32_t id = ald(&s->w[2]);
        if (id >= nslots_) return GenValue::kMaxSize + 1;
        uint32_t sz = ald(&slot(id)->w[1]);
        return (sz > kSlotVal && sz <= GenValue::kMaxSize) ? sz : GenValue::kMaxSize + 1;
    }
    bool slot_matches(const Slot* s, const uint8_t* rec, uint8_t* tmp) const;
    uint32_t alloc_slot(Session* S, bool blocking);
    bool refill(Session* S);
    bool alloc_chain(Session* S, uint32_t nu, bool blocking, uint32_t* ids);
    void fill_chain(const uint32_t* ids, uint32_t nu, uint64_t key, const uint8_t* data, uint32_t size, uint32_t flags);
    // Set kLive (plus `extra` flags) on a filled head the index already names. Lock-free: the head's
    // contents were completed (and its stripe version bumped) by fill_chain before the entry store
    // that made it reachable, and no lock holder changes the flags of a head that is not live, so
    // the fetch_or is the only flag change; a reader that sees it not live re-reads the entry. Being
    // lock-free, publish never takes a second stripe lock under a held one (replace_head).
    void publish(uint32_t head, uint32_t extra = 0) {
        flg(head).fetch_or((uint8_t)(kLive | extra), std::memory_order_release);
    }
    void free_ids(Session* S, const uint32_t* ids, uint32_t n) {
        for (uint32_t i = 0; i < n; i++) free_local(S, ids[i]);
    }
    // GC accounting: a clean head (w, locked) is about to be modified: its record D stops being current.
    void note_clean_stale(Session* S, const Slot* s, uint32_t w) const {
        // A tombstone's record was counted dead when it was written (dead_bytes_ / write accounting).
        if (is_clean(w) && !is_tomb(w)) S->c.live_sub += rec_charge(kHdr + value_size(s, w));
    }
    // ... a disk head p is replaced by a push. `stale` = the live charge of the key's record the push
    // makes non-current when the caller read it (Delete, RMW), or kBlindPush (blind Upsert): then p is
    // assumed to be the key's record and charged by its size code (exact, or an upper bound). The
    // assumption can only be wrong in a multi-key class, by at most kMaxRecCharge per blind push;
    // maybe_gc bounds that error with the blind-push count (ups_push).
    static constexpr uint64_t kBlindPush = ~0ULL;
    static void note_push_stale(Session* S, uint64_t p, uint64_t stale) {
        if (stale == kBlindPush) stale = p ? ptr_live_charge(p) : 0;
        S->c.live_sub += stale;
    }
    // Disk payload that replaces a clean head (locked slot s, flags w, resident entry e): its record D,
    // with D's size code (the slot holds D's value), T when D's on-disk prev is not chain.
    uint64_t clean_disk_ptr(uint64_t e, const Slot* s, uint32_t w) const {
        const bool tomb = is_tomb(w);
        return mk_ptr(res_addr(e) * kUnit, kHdr + (tomb ? 0 : value_size(s, w)), tomb) |
               (res_mode(e) == kMDSolo ? kPtrT : 0);
    }
    uint32_t chain_conts(const Slot* s, uint32_t* ids) const;
    bool copy_value(const Slot* s, uint32_t size, uint8_t* out) const;
    void write_small(Slot* s, const uint8_t* data, uint32_t size);
    bool replace_head(Session* S, std::atomic<uint64_t>* ep, uint64_t e, uint64_t key, const uint8_t* exp_val,
                      uint32_t exp_size, bool exp_tomb);
    bool push_head(Session* S, std::atomic<uint64_t>* ep, uint64_t e, uint64_t key, const uint8_t* data, uint32_t size,
                   bool tomb, uint32_t nu, bool& have, uint32_t ref, uint64_t stale);
    void free_local(Session* S, uint32_t si);
    void link_chain(const uint32_t* ids, size_t n);
    uint64_t free_slots() const {
        uint64_t f = fresh_.load(std::memory_order_relaxed);
        return (f < nslots_ ? nslots_ - f : 0) + recycled_.load(std::memory_order_relaxed);
    }
    void kick_evictor();

    // ── log
    uint64_t log_alloc(uint64_t len);
    int seg_fd(uint64_t seg);
    void open_seg(uint64_t seg, bool create);
    void log_write(uint64_t off, const uint8_t* buf, uint64_t len);
    uint64_t write_record_sync(Session* S, uint64_t key, bool tomb, uint64_t prev, const uint8_t* val, uint32_t size);
    const uint8_t* read_record_sync(uint8_t* buf, uint64_t p, RecView& r);
    void read_range(uint64_t p, uint64_t& base, uint32_t& len) const;
    uint64_t next_seq() { return seq_.fetch_add(1, std::memory_order_relaxed); }

    // ── operations
    enum class LR { Hit, NotFound, Disk };
    LR lookup_mem(uint64_t key, GenValue* out, std::atomic<uint64_t>** ep_out, uint64_t* e_out, uint64_t* p_out);
    bool chain_lookup_sync(Session* S, uint64_t p, uint64_t key, GenValue* out, uint64_t* hit_prev = nullptr,
                           uint64_t* hit_charge = nullptr);
    void put_value(Session* S, uint64_t key, const uint8_t* data, uint32_t size);
    void flush_slot_sync(Session* S, uint32_t si, std::atomic<uint64_t>* ep, uint64_t e);
    void try_install(Session* S, std::atomic<uint64_t>* ep, uint64_t head_e, uint64_t p, uint64_t key,
                     const uint8_t* val, uint32_t size, uint64_t rec_prev);
    bool ensure_ring(Session* S);
    void submit_read(Session* S, uint32_t id);
    void process(Session* S, uint32_t id, int res, int64_t now, int64_t poll_prev);
    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    void drive(Session* S, bool wait);
    void release_session_io(Session* S);

    // ── background / durability
    size_t evict_batch(uint8_t* buf, size_t cap, size_t want, std::vector<Victim>& vic, std::vector<uint32_t>& freed,
                       bool allow_write, uint64_t scan_limit);
    // `live`: rec_charge sum of the batch's non-tombstone records; the rest of the appended length
    // (tombstones, alignment and placement gaps) is counted dead at once.
    uint64_t write_victims(uint8_t* buf, uint64_t pos, uint64_t live);
    // rec_charge of the records of v[0, n) in buf that are not tombstones.
    static uint64_t live_charge(const uint8_t* rec, uint32_t bytes) { return (rec[15] & 1) ? 0 : rec_charge(bytes); }
    static uint64_t batch_live(const uint8_t* buf, const Victim* v, size_t n) {
        uint64_t t = 0;
        for (size_t i = 0; i < n; i++) t += live_charge(buf + v[i].boff, v[i].bytes);
        return t;
    }
    void reclaim_tombs(Session* S);
    static constexpr size_t kTombBatch = 512;  // 512 x 128 B tombstone units: one 64 KiB append
    void push_chains(const std::vector<uint32_t>& freed, size_t from);
    bool help_evict(Session* S);
    void evictor_main();
    void cleaner_main(int idx);
    bool clean_chunk(CleanCtx& C, uint64_t start, uint64_t end);
    void clean_flush(CleanCtx& C);
    void drain_dirty();
    void monitor_reads();
    void sample_touch(uint64_t key);
    int64_t sample_cold(uint64_t key);
    void sample_done(int64_t i) {
        if (i >= 0) samp_st_[i].fetch_and((uint8_t)~2u, std::memory_order_relaxed);
    }
    int64_t sample_slot(uint64_t key);
    void print_run_stats(const char* when);
    struct SegRange {
        uint32_t seg;
        uint64_t end;  // bytes of the segment that belong to the checkpointed log
        uint64_t cnt = 0, dig = 0;  // expected record count / digest of [seg_start, end) (chk)
        bool chk = false;
    };
    void recover(const std::vector<SegRange>& segs, uint64_t active_seg, uint64_t active_off, uint64_t next_seq);
    uint64_t rebuild_index(const std::vector<SegRange>& segs, uint64_t need, uint64_t next_seq);
    template <class F>
    void scan_seg(const SegRange& sr, uint8_t* buf, size_t chunk, F&& f);
    void write_manifest();
    void sync_segments();
    uint32_t new_seg_locked();
    void log_switch();
    uint64_t live_estimate() const;
    // Exact live accounting base: live log bytes = `live` now (quiesced: recovery, GC, head walk).
    void rebase_live(uint64_t live) {
        const Counters t = sum_counters();
        live_base_ = live;
        log_base_ = log_bytes_.load();
        sub_base_ = t.live_sub;
        push_base_ = t.ups_push;
        dead_base_ = dead_bytes_.load();
    }
    uint64_t live_heads() const;
    uint64_t maybe_gc();
    uint64_t gc_run();
    void reset_mem();
    void print_stats(const char* tag);
    Counters sum_counters() const;

    // ── members
    const uint64_t id_;
    size_t budget_ = 0;
    std::string dir_;
    int dirfd_ = -1;
    int lockfd_ = -1;  // flock(LOCK_EX) on <dir>/LOCK for the store's lifetime (released by close / death)
    bool temp_dir_ = false;
    bool inited_ = false;

    std::atomic<uint64_t>* bk_ = nullptr;
    uint64_t nb_ = 0, bmask_ = 0;
    std::atomic<uint64_t>* ovbits_ = nullptr;
    OvTable ov_[24];
    std::atomic<int> nov_{0};
    std::mutex ov_mu_;
    uint64_t ovcap0_ = 1024;
    // Overflow reclaim: a reclaimed overflow entry is zeroed (dead) and skipped by ov_find; its slot
    // is not reused in place (no position ever changes class while ops run). Once dead slots reach
    // the live count, compact_ov() rebuilds the tables with the store quiesced, so the overflow index
    // stays proportional to the live classes under insert/delete churn.
    std::atomic<uint64_t> ov_total_{0}, ov_dead_{0};
    void note_reclaim(const std::atomic<uint64_t>* ep) {
        if (!in_bk(ep)) ov_dead_.fetch_add(1, std::memory_order_relaxed);
    }
    void maybe_compact_ov() {
        uint64_t d = ov_dead_.load(std::memory_order_relaxed);
        if (__builtin_expect(d >= kOvCompactMin, 0) && 2 * d >= ov_total_.load(std::memory_order_relaxed)) compact_ov();
    }
    static constexpr uint64_t kOvCompactMin = 16384;
    void compact_ov();
    void quiesce_all();
    void resume_all();
    bool delete_op(uint64_t key);
    std::atomic<uint32_t>* stripes_ = nullptr;

    uint8_t* arena_ = nullptr;
    uint64_t nslots_ = 0;
    SideLine* side_ = nullptr;  // stripe seqlocks and flag bytes, kSideSlots slots per line
    uint64_t nside_ = 0;
    std::atomic<uint64_t> fresh_{0};
    std::atomic<uint64_t> hand_{0};
    std::mutex pool_mu_;
    std::vector<std::pair<uint32_t, uint32_t>> chains_;  // (head, count), linked through val[0..4)
    std::atomic<uint64_t> recycled_{0};
    uint64_t low_wm_ = 0, high_wm_ = 0;

    std::atomic<int> fds_[kMaxSeg];
    std::atomic<uint8_t> segdirty_[kMaxSeg];
    // Per segment: number of records and the order-independent digest (sum of rec_dig) of every
    // record appended to it (log_write), reset when the number is (re)used. A checkpoint persists
    // both for its range; recovery and GC re-derive them from the scan and fail loudly on a
    // mismatch, so a lost/zeroed record inside the checkpointed range is never taken for padding.
    std::atomic<uint64_t> seg_cnt_[kMaxSeg];
    std::atomic<uint64_t> seg_dig_[kMaxSeg];
    std::mutex seg_mu_;
    std::atomic<uint64_t> tail_{kLogStart};  // next append address; always inside the active segment
    std::atomic<uint64_t> seq_{1};
    // Segments in use (checkpointed or written since), guarded by alloc_mu_. A segment switch takes
    // the lowest free number, so addresses stay proportional to the live log, not to total writes.
    std::mutex alloc_mu_;
    uint64_t seg_used_[kSegWords] = {};
    // Log garbage collection (at Checkpoint, see maybe_gc): allocated log bytes vs an estimate of
    // live bytes = live_base_ (exact at the last GC / recovery) + bytes appended since - bytes that
    // stopped being current since (per-session live_sub, approximate) - bytes dead when appended
    // (dead_bytes_: tombstones, padding, aborted commits). A GC re-bases it to the exact copied size.
    std::atomic<uint64_t> log_bytes_{0};
    // Appended bytes known dead when written (batch writers: tombstones, padding, records whose commit
    // aborted); one add per batch, never per op. Session writes count theirs in live_sub.
    std::atomic<uint64_t> dead_bytes_{0};
    uint64_t live_base_ = 0, log_base_ = 0, sub_base_ = 0, dead_base_ = 0;  // guarded by ckpt_mu_ (set at open / GC)
    uint64_t push_base_ = 0;  // blind pushes (ups_push) at the last exact base
    uint64_t st_live_walks_ = 0;
    std::atomic<uint64_t> gc_gen_{0};
    uint64_t st_gc_runs_ = 0, st_gc_segs_ = 0;
    uint32_t align_ = 512;
    uint32_t rd_unit_ = 512;

    std::vector<std::thread> evth_;
    std::atomic<bool> ev_stop_{false}, ev_pause_{false};
    std::atomic<int> ev_paused_{0};
    std::atomic<int> ev_sleeping_{0};
    std::mutex ev_mu_;
    std::condition_variable ev_cv_;

    std::atomic<uint32_t> ckpt_flag_{0};
    std::mutex ckpt_mu_;

    mutable std::mutex sess_mu_;
    std::vector<Session*> sessions_;  // every session ever made (bounded: stopped ones are reused)
    std::vector<Session*> idle_;      // stopped sessions, ready for the next StartSession
    std::unordered_map<std::thread::id, Session*> by_thread_;

    std::atomic<uint64_t> st_evict_{0}, st_evict_abort_{0}, st_bytes_written_{0}, st_flush_sync_{0}, st_ckpt_recs_{0}, st_ov_compact_{0};
    std::atomic<uint64_t> st_drop_clean_{0}, st_evict_dirty_{0}, st_cleaned_{0}, st_clean_abort_{0};
    std::atomic<bool> run_started_{false}, run_ready_{false}, run_printed_{false};
    uint64_t run_resident_ = 0, run_evict_ = 0, run_bytes_ = 0, run_ios_ = 0, run_drop_ = 0, run_dirty_ = 0;
    uint64_t run_waits_ = 0, run_helps_ = 0, run_depth_ = 0, run_enters_ = 0, run_clean0_ = 0, run_ops_ = 0;
    std::chrono::steady_clock::time_point run_t0_;
    std::atomic<int64_t> last_io_ns_{0};  // steady-clock ns when the read counter last moved
    std::atomic<bool> reads_busy_{false};

    // Write-behind cleaner: its hand runs ahead of the CLOCK hand (same slot order, at most one
    // revolution ahead), so the records the CLOCK reaches next are already clean.
    std::atomic<uint64_t> chand_{0};
    int ncleaners_ = 0;

    // Pre-run drain: when the last session stops before any async read, the cleaners write every
    // dirty slot behind (one bounded sweep) so a following read phase evicts with no writes.
    // dcur_ = (generation << 40) | next slot to claim; ddone_ = (generation << 40) | chunks done.
    static constexpr int kDGenShift = 40;
    static constexpr uint64_t kDPosMask = (1ULL << kDGenShift) - 1;
    std::mutex drain_mu_;
    uint64_t dgen_ = 0, drained_writes_ = ~0ULL;  // guarded by drain_mu_
    std::atomic<uint64_t> dcur_{0}, dend_{0}, ddone_{0};
    // Dirty marks: one byte per kHandChunk slots, set when a slot of the chunk becomes dirty, cleared
    // by the drain; the drain sweeps only the marked chunks (dlist_), so its cost tracks the dirty work.
    std::atomic<uint8_t>* dmark_ = nullptr;
    std::atomic<uint32_t>* dlist_ = nullptr;
    uint64_t nchunks_ = 0;
    CleanCtx* drain_ctx_ = nullptr;  // guarded by drain_mu_

    // Re-read sampler (observability): keys with hash & kSampMask == 0 are tracked through the run.
    // State bit0: touched in the run; bit1: a cold read in flight.
    static constexpr uint64_t kSampMask = 2047;
    std::atomic<uint64_t>* samp_key_ = nullptr;
    std::atomic<uint8_t>* samp_st_ = nullptr;
    uint64_t samp_cap_ = 0;
    std::atomic<uint64_t> st_reread_{0}, st_dup_inflight_{0}, st_samp_cold_{0};
};

// ───────────────────────────── init / teardown ─────────────────────────────
void SlotStore::InitExtended(size_t hash_table_size, size_t log_size_bytes, size_t mem_budget_bytes,
                             const char* storage_path) {
    (void)log_size_bytes;
    if (inited_) return;
    inited_ = true;
    budget_ = mem_budget_bytes;

    // Storage directory.
    if (storage_path && *storage_path) {
        dir_ = storage_path;
        ::mkdir(dir_.c_str(), 0755);
    } else {
        const char* t = std::getenv("TMPDIR");
        std::string tmpl = std::string(t && *t ? t : "/tmp") + "/kvstore.XXXXXX";
        std::vector<char> b(tmpl.begin(), tmpl.end());
        b.push_back(0);
        if (!mkdtemp(b.data())) die("mkdtemp");
        dir_ = b.data();
        temp_dir_ = true;
    }
    dirfd_ = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dirfd_ < 0) die("open storage dir");
    // Exclusive open: one store per directory. Taken before anything in the directory is read,
    // created or removed; the kernel drops it when the fd closes (destructor) or the process dies.
    lockfd_ = ::openat(dirfd_, "LOCK", O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (lockfd_ < 0) die("open LOCK file");
    if (::flock(lockfd_, LOCK_EX | LOCK_NB) != 0) {
        int err = errno;
        std::fprintf(stderr,
                     "kvstore: FATAL: storage at %s is already open by another kvstore instance (flock LOCK: %s); "
                     "refusing a second open\n",
                     dir_.c_str(), std::strerror(err));
        std::fflush(stderr);
        std::abort();
    }

    // O_DIRECT granularity: 512 B if the device takes it, else 4 KiB.
    {
        ::unlinkat(dirfd_, ".align_probe", 0);  // never write through a planted name
        int fd = ::openat(dirfd_, ".align_probe", O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_DIRECT, 0644);
        align_ = 4096;
        if (fd >= 0) {
            uint8_t* b = aligned_buf(4096);
            if (::pwrite(fd, b, 512, 512) == 512 && ::pread(fd, b, 512, 512) == 512) align_ = 512;
            free(b);
            ::close(fd);
        }
        ::unlinkat(dirfd_, ".align_probe", 0);
    }
    rd_unit_ = align_;

    // Index: 2 x hint entries, capped at a quarter of the budget.
    uint64_t hint = hash_table_size ? hash_table_size : (1u << 20);
    uint64_t entries = 16;
    while (entries < 2 * hint) entries <<= 1;
    nb_ = entries / 8;
    if (budget_) {
        while (nb_ > 64 && nb_ * 64 > budget_ / 4) nb_ >>= 1;
    }
    if (nb_ < 64) nb_ = 64;
    bmask_ = nb_ - 1;
    bk_ = (std::atomic<uint64_t>*)map_anon(nb_ * 64, true);
    ovbits_ = (std::atomic<uint64_t>*)map_anon((nb_ + 63) / 64 * 8, false);
    uint64_t ovcap = 1024;
    while (ovcap < nb_ * 8 / 32) ovcap <<= 1;
    ov_[0].cls = (std::atomic<uint64_t>*)map_anon(ovcap * 8, true);
    ov_[0].ent = (std::atomic<uint64_t>*)map_anon(ovcap * 8, true);
    ov_[0].mask = ovcap - 1;
    ovcap0_ = ovcap;
    nov_.store(1);
    stripes_ = (std::atomic<uint32_t>*)map_anon(kStripes * 4, false);
    samp_cap_ = 1024;
    while (samp_cap_ < 4 * hint / (kSampMask + 1)) samp_cap_ <<= 1;
    samp_key_ = (std::atomic<uint64_t>*)map_anon(samp_cap_ * 8, false);
    samp_st_ = (std::atomic<uint8_t>*)map_anon(samp_cap_, false);

    // Slots: whatever the budget leaves after the index and a reserve.
    uint64_t fixed = nb_ * 64 + (nb_ + 63) / 64 * 8 + ovcap * 16 + kStripes * 4 + samp_cap_ * 9;
    if (budget_) {
        // Scaled slack (capped: the real overhead does not grow past ~170 MiB) + fixed (sessions,
        // rings, buffers); the fixed part shrinks with small budgets. Budgets <= 6 GiB: budget/64.
        uint64_t reserve = std::min<uint64_t>(budget_ / 64, 96ULL << 20) + std::min<uint64_t>(16ULL << 20, budget_ / 8);
        uint64_t avail = budget_ > fixed + reserve ? budget_ - fixed - reserve : 0;
        // Per slot: 108 B of arena + 64/48 B of side line (flag byte and stripe lock share), plus one
        // side line of rounding.
        avail = avail > sizeof(SideLine) ? avail - sizeof(SideLine) : 0;
        avail -= avail / 8192;  // the drain's chunk marks and list
        nslots_ = avail * kSideSlots / (sizeof(Slot) * kSideSlots + sizeof(SideLine));
    } else {
        nslots_ = nb_ * 8;
    }
    if (nslots_ < 4096) nslots_ = 4096;
    if (nslots_ > kMaxSlots) nslots_ = kMaxSlots;  // slot id * kNT + short tag fits 32 bits
    nside_ = (nslots_ + kSideSlots - 1) / kSideSlots;
    side_ = (SideLine*)map_anon(nside_ * sizeof(SideLine), true);
    nchunks_ = (nslots_ + kHandChunk - 1) / kHandChunk;
    dmark_ = (std::atomic<uint8_t>*)map_anon(nchunks_, false);
    dlist_ = (std::atomic<uint32_t>*)map_anon(nchunks_ * 4, false);
    low_wm_ = std::max<uint64_t>(nslots_ / 256, 2048);
    if (low_wm_ > nslots_ / 4) low_wm_ = nslots_ / 4;
    high_wm_ = 2 * low_wm_;

    // Recover or start fresh. A fresh store writes an empty MANIFEST before its first segment, so
    // every directory holding segments also holds a MANIFEST: a missing, unreadable, short or
    // corrupt MANIFEST next to data is damage, and the open fails loudly without touching a file.
    // The slot arena is mapped before recovery: recovery builds its key map inside it (budgeted,
    // still empty) and releases it to zero pages before the slot code runs.
    arena_ = (uint8_t*)map_anon(nslots_ * sizeof(Slot), true);

    bool have = false;
    {
        int fd = ::openat(dirfd_, "MANIFEST", O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            ManifestV2* m2 = new ManifestV2();
            ssize_t got = ::pread(fd, m2, sizeof *m2, 0);
            std::vector<uint8_t> trailer;
            if (got == (ssize_t)sizeof *m2 && m2->version == 3) {
                const size_t tmax = 8 + (size_t)kMaxSeg * sizeof(ManifestSeg) + 4 + 1;
                trailer.resize(tmax);
                size_t tg = 0;
                for (;;) {
                    ssize_t r = ::pread(fd, trailer.data() + tg, tmax - tg, (off_t)(sizeof *m2 + tg));
                    if (r < 0 && errno == EINTR) continue;
                    if (r <= 0) break;
                    tg += (size_t)r;
                    if (tg == tmax) break;
                }
                trailer.resize(tg);
            }
            ::close(fd);
            if (got < 16) die_state(dir_, "MANIFEST is short or unreadable (damaged)");
            if (m2->magic != kManifestMagic) die_state(dir_, "MANIFEST has a bad magic (damaged or foreign)");
            std::vector<SegRange> segs;
            if (m2->version == 1) {
                Manifest m;
                if (got != (ssize_t)sizeof m) die_state(dir_, "MANIFEST is short or unreadable (damaged)");
                std::memcpy(&m, m2, sizeof m);
                if (crc32c(0xFFFFFFFFu, (const uint8_t*)&m, offsetof(Manifest, crc)) != m.crc)
                    die_state(dir_, "MANIFEST fails its checksum (damaged)");
                if (m.tail < kLogStart || ((m.tail - 1) >> kSegShift) >= kMaxSeg)
                    die_state(dir_, "MANIFEST holds an impossible log tail");
                uint64_t nseg = m.tail > kLogStart ? ((m.tail - 1) >> kSegShift) + 1 : 1;
                for (uint64_t sg = 0; sg < nseg; sg++)
                    segs.push_back({(uint32_t)sg, std::min<uint64_t>(m.tail - (sg << kSegShift), kSegSize)});
                uint64_t act = nseg - 1;
                recover(segs, act, m.tail - (act << kSegShift), m.next_seq);
            } else if (m2->version == 2 || m2->version == 3) {
                if (got != (ssize_t)sizeof *m2) die_state(dir_, "MANIFEST is short or unreadable (damaged)");
                if (crc32c(0xFFFFFFFFu, (const uint8_t*)m2, offsetof(ManifestV2, crc)) != m2->crc)
                    die_state(dir_, "MANIFEST fails its checksum (damaged)");
                if (m2->active_seg >= kMaxSeg || m2->active_off < seg_start(m2->active_seg) ||
                    m2->active_off >= kSegSize || !((m2->live[m2->active_seg / 64] >> (m2->active_seg % 64)) & 1))
                    die_state(dir_, "MANIFEST holds an impossible active segment");
                for (uint32_t sg = 0; sg < kMaxSeg; sg++)
                    if ((m2->live[sg / 64] >> (sg % 64)) & 1)
                        segs.push_back({sg, sg == m2->active_seg ? m2->active_off : kSegSize});
                if (m2->version == 3) {
                    // Digest trailer: exactly one entry per live segment, in order.
                    uint64_t n = 0;
                    if (trailer.size() >= 12) std::memcpy(&n, trailer.data(), 8);
                    if (trailer.size() < 12 || n != segs.size() || trailer.size() != 8 + n * sizeof(ManifestSeg) + 4)
                        die_state(dir_, "MANIFEST digest trailer is short or malformed (damaged)");
                    uint32_t tc;
                    std::memcpy(&tc, trailer.data() + trailer.size() - 4, 4);
                    if (crc32c(0xFFFFFFFFu, trailer.data(), trailer.size() - 4) != tc)
                        die_state(dir_, "MANIFEST digest trailer fails its checksum (damaged)");
                    for (uint64_t i = 0; i < n; i++) {
                        ManifestSeg e;
                        std::memcpy(&e, trailer.data() + 8 + i * sizeof(ManifestSeg), sizeof e);
                        if (e.seg != segs[i].seg) die_state(dir_, "MANIFEST digest trailer names the wrong segment (damaged)");
                        segs[i].cnt = e.cnt;
                        segs[i].dig = e.dig;
                        segs[i].chk = true;
                    }
                }
                recover(segs, m2->active_seg, m2->active_off, m2->next_seq);
            } else {
                die_state(dir_, "MANIFEST holds an unsupported version");
            }
            delete m2;
            have = true;
        } else if (errno != ENOENT) {
            die("open MANIFEST (storage present but unreadable; refusing to open)");
        }
    }
    if (!have) {
        seg_used_[0] = 1;  // the log starts in segment 0 (created by the first append)
        bool data = false;
        if (DIR* d = ::opendir(dir_.c_str())) {
            while (dirent* de = ::readdir(d)) {
                unsigned sidx;
                if (parse_seg_name(de->d_name, &sidx)) data = true;
            }
            ::closedir(d);
        }
        if (data) die_state(dir_, "log segments exist but the MANIFEST is missing (damaged)");
        write_manifest();  // empty checkpoint: tail = kLogStart
    }

    int ncpu = (int)std::thread::hardware_concurrency();
    {
        cpu_set_t cs;
        if (sched_getaffinity(0, sizeof cs, &cs) == 0) ncpu = CPU_COUNT(&cs);
    }
    int ne = std::max(1, std::min(4, ncpu / 8));
    ncleaners_ = std::max(1, std::min(4, ncpu / 8));
    for (int k = 0; k < ne; k++) evth_.emplace_back([this] { evictor_main(); });
    for (int k = 0; k < ncleaners_; k++) evth_.emplace_back([this, k] { cleaner_main(k); });

    std::fprintf(stderr,
                 "kvstore: init budget=%zu index=%llu MiB (%llu buckets) overflow=%llu slots=%llu (%llu MiB) "
                 "align=%u evictors=%d cleaners=%d recovered=%d\n",
                 budget_, (unsigned long long)(nb_ * 64 >> 20), (unsigned long long)nb_, (unsigned long long)ovcap,
                 (unsigned long long)nslots_, (unsigned long long)(nslots_ * sizeof(Slot) >> 20), align_, ne,
                 ncleaners_, have ? 1 : 0);
}

SlotStore::~SlotStore() {
    ev_stop_.store(true);
    {
        std::lock_guard<std::mutex> g(ev_mu_);
        ev_cv_.notify_all();
    }
    for (auto& t : evth_) t.join();
    {
        std::lock_guard<std::mutex> g(sess_mu_);
        for (Session* s : sessions_) {
            release_session_io(s);
            free(s->sbuf);
            free(s->wbuf);
            free(s->tbuf);
            delete s->tmpv;
            delete s;
        }
        sessions_.clear();
    }
    for (uint32_t i = 0; i < kMaxSeg; i++) {
        int fd = fds_[i].load();
        if (fd >= 0) ::close(fd);
    }
    if (arena_) munmap(arena_, nslots_ * sizeof(Slot));
    if (side_) munmap(side_, nside_ * sizeof(SideLine));
    if (bk_) munmap(bk_, nb_ * 64);
    if (ovbits_) munmap(ovbits_, (nb_ + 63) / 64 * 8);
    for (int i = 0; i < nov_.load(); i++) {
        munmap(ov_[i].cls, (ov_[i].mask + 1) * 8);
        munmap(ov_[i].ent, (ov_[i].mask + 1) * 8);
    }
    if (stripes_) munmap(stripes_, kStripes * 4);
    if (dmark_) munmap(dmark_, nchunks_);
    if (dlist_) munmap(dlist_, nchunks_ * 4);
    if (drain_ctx_) {
        free(drain_ctx_->buf);
        delete drain_ctx_;
    }
    if (samp_key_) munmap(samp_key_, samp_cap_ * 8);
    if (samp_st_) munmap(samp_st_, samp_cap_);
    if (temp_dir_) {
        if (DIR* d = ::opendir(dir_.c_str())) {
            while (dirent* de = ::readdir(d))
                if (de->d_name[0] != '.') ::unlinkat(dirfd_, de->d_name, 0);
            ::closedir(d);
        }
        ::rmdir(dir_.c_str());
    }
    if (lockfd_ >= 0) ::close(lockfd_);  // releases the flock
    if (dirfd_ >= 0) ::close(dirfd_);
}

Session* SlotStore::sess_slow() {
    std::lock_guard<std::mutex> g(sess_mu_);
    auto it = by_thread_.find(std::this_thread::get_id());
    Session* s;
    if (it != by_thread_.end()) {
        s = it->second;
    } else if (!idle_.empty()) {
        s = idle_.back();
        idle_.pop_back();
        by_thread_[std::this_thread::get_id()] = s;
    } else {
        s = new Session();
        s->sbuf = aligned_buf(16384);
        s->wbuf = aligned_buf(8192);
        s->tmpv = new GenValue();
        s->fcache.reserve(512);
        sessions_.push_back(s);
        by_thread_[std::this_thread::get_id()] = s;
    }
    tls.owner = id_;
    tls.s = s;
    return s;
}

void SlotStore::release_session_io(Session* S) {
#if KV_HAVE_URING
    if (S->ring_ok) S->ring.close();
#endif
    S->ring_ok = false;
    S->ring_tried = false;
    free(S->bufs);
    S->bufs = nullptr;
    free(S->ebuf);
    S->ebuf = nullptr;
    S->pend.clear();
    S->pend.shrink_to_fit();
    S->pfree.clear();
    S->pfree.shrink_to_fit();
}

void SlotStore::StopSession() {
    Session* S = sess();
    CompletePending(true);
    // Return this thread's free slots and I/O resources; the session object is kept for reuse.
    // Under the op guard: a log GC (Checkpoint, quiesced) resets the slot pools and these caches.
    {
    OpGuard og(this, S);
    if (!S->fcache.empty()) {
        std::lock_guard<std::mutex> g(pool_mu_);
        // Link the cached ids into chains.
        size_t i = 0;
        while (i < S->fcache.size()) {
            size_t n = std::min<size_t>(256, S->fcache.size() - i);
            link_chain(&S->fcache[i], n);
            chains_.push_back({S->fcache[i], (uint32_t)n});
            recycled_.fetch_add(n);
            i += n;
        }
        S->fcache.clear();
    }
    S->tpend.clear();  // still dirty tombstones: the evictor writes and reclaims them
    }
    // compact_ov walks every session's pend under sess_mu_ (outside any op of ours, so the op
    // guard does not exclude it): detach pend/pfree under that lock before they are freed.
    {
        std::vector<Pend> dead_pend;
        std::vector<uint32_t> dead_pfree;
        {
            std::lock_guard<std::mutex> g(sess_mu_);
            dead_pend.swap(S->pend);
            dead_pfree.swap(S->pfree);
        }
    }
    release_session_io(S);
    bool last;
    {
        std::lock_guard<std::mutex> g(sess_mu_);
        by_thread_.erase(std::this_thread::get_id());
        // The session stays in sessions_ (owned, its counters keep counting) and goes to the idle
        // list: the next StartSession on any thread reuses it, so sessions_ is bounded by the peak
        // number of concurrent sessions. Checkpoint may walk it at any time: idle has in_op == 0.
        idle_.push_back(S);
        last = by_thread_.empty();
    }
    tls.owner = 0;
    tls.s = nullptr;
    // Last session out of a write (load) phase: write every dirty slot behind before returning, so
    // a read phase that follows starts with every resident record clean (evictions cost no write).
    if (last && !run_started_.load(std::memory_order_acquire)) drain_dirty();
}

// ───────────────────────────── index ─────────────────────────────
// Is the resident entry *ep (value x when the caller loaded it) the entry of class c? Confirmed with
// the head slot's key: 1 yes, 0 no, -1 the entry changed (reload it). An entry position never changes
// class while it is non-zero (a reclaimed entry is zeroed first), and an entry names a slot only while that slot holds a key of
// the entry's class, so a key read inside a stable read window, with the entry still x, decides it.
int SlotStore::res_class_is(const std::atomic<uint64_t>* ep, uint64_t x, const Cls& c) const {
    // The lock need not be free: a key changes only inside a fill (an unpublished slot, which no entry
    // names), and every fill ends with a version bump, so an equal word around the read suffices.
    const Slot* s = slot(res_slot(x));
    const std::atomic<uint32_t>& L = lockw(res_slot(x));
    for (;;) {
        uint64_t t0 = win_begin();
        uint32_t v1 = L.load(std::memory_order_acquire);
        uint64_t k = slot_key(s);
        std::atomic_thread_fence(std::memory_order_acquire);
        uint64_t x2 = ep->load(std::memory_order_acquire);
        uint32_t v2 = L.load(std::memory_order_relaxed);
        if (v2 != v1 || win_end() - t0 >= kReadWindowTicks) continue;
        if (x2 != x) return -1;
        return classify(k).id == c.id ? 1 : 0;
    }
}

std::atomic<uint64_t>* SlotStore::find_entry(const Cls& c) const {
    const uint64_t mask = kValid | kIsSlot | kSec | (kTagMask << kTagShift);
    const uint64_t want = kValid | ((uint64_t)c.tag << kTagShift);
    const uint32_t t5 = tag5(c.tag);
    __builtin_prefetch(bk_ + c.b2 * 8);
    for (int side = 0; side < 2; side++) {
        std::atomic<uint64_t>* B = bk_ + (side ? c.b2 : c.b1) * 8;
        const uint64_t w = side ? (want | kSec) : want;
        prefetch_candidates(B, 8, t5);
        for (int i = 0; i < 8; i++) {
            for (;;) {
                uint64_t x = B[i].load(std::memory_order_acquire);
                if (!(x & kIsSlot)) {
                    if ((x & mask) == w) return &B[i];
                    break;
                }
                if (res_tag5(x) != t5) break;
                int r = res_class_is(&B[i], x, c);
                if (r == 1) return &B[i];
                if (r == 0) break;
            }
        }
    }
    if ((ovbits_[c.b1 >> 6].load(std::memory_order_acquire) >> (c.b1 & 63)) & 1) return ov_find(c.id);
    return nullptr;
}

// The entry naming slot si, whose key the caller holds stable (slot locked): no key check needed,
// only si's class entry can name it. nullptr if no entry names si (not a published head).
std::atomic<uint64_t>* SlotStore::find_own_entry(uint64_t key, uint32_t si, Cls* cout) const {
    const Cls c = classify(key);
    if (cout) *cout = c;
    for (int side = 0; side < 2; side++) {
        std::atomic<uint64_t>* B = bk_ + (side ? c.b2 : c.b1) * 8;
        for (int i = 0; i < 8; i++) {
            uint64_t x = B[i].load(std::memory_order_acquire);
            if ((x & kIsSlot) && res_slot(x) == si) return &B[i];
        }
    }
    if ((ovbits_[c.b1 >> 6].load(std::memory_order_acquire) >> (c.b1 & 63)) & 1) {
        std::atomic<uint64_t>* ep = ov_find(c.id);
        if (ep) {
            uint64_t x = ep->load(std::memory_order_acquire);
            if ((x & kIsSlot) && res_slot(x) == si) return ep;
        }
    }
    return nullptr;
}

std::atomic<uint64_t>* SlotStore::ov_find(uint64_t id) const {
    int n = nov_.load(std::memory_order_acquire);
    for (int k = 0; k < n; k++) {
        const OvTable& t = ov_[k];
        for (uint64_t i = mix64(id) & t.mask;; i = (i + 1) & t.mask) {
            uint64_t c = t.cls[i].load(std::memory_order_acquire);
            if (c == 0) break;
            if (c == id + 1 && t.ent[i].load(std::memory_order_acquire) != 0) return &t.ent[i];  // 0: dead
        }
    }
    return nullptr;
}

std::atomic<uint64_t>* SlotStore::ov_insert(uint64_t id, uint64_t e, uint64_t b1) {
    std::lock_guard<std::mutex> g(ov_mu_);
    int n = nov_.load(std::memory_order_relaxed);
    OvTable* t = &ov_[n - 1];
    if ((t->count + 1) * 4 > (t->mask + 1) * 3) {
        if (n >= 24) die("overflow index exhausted", n);
        uint64_t cap = (t->mask + 1) * 2;
        OvTable& nt = ov_[n];
        nt.cls = (std::atomic<uint64_t>*)map_anon(cap * 8, true);
        nt.ent = (std::atomic<uint64_t>*)map_anon(cap * 8, true);
        nt.mask = cap - 1;
        nt.count = 0;
        nov_.store(n + 1, std::memory_order_release);
        t = &nt;
    }
    uint64_t i = mix64(id) & t->mask;
    while (t->cls[i].load(std::memory_order_relaxed) != 0) i = (i + 1) & t->mask;
    t->ent[i].store(e, std::memory_order_relaxed);
    t->cls[i].store(id + 1, std::memory_order_release);
    t->count++;
    ov_total_.fetch_add(1, std::memory_order_relaxed);
    ovbits_[b1 >> 6].fetch_or(1ULL << (b1 & 63), std::memory_order_release);
    return &t->ent[i];
}

std::atomic<uint64_t>* SlotStore::create_entry(const Cls& c) {
    std::atomic<uint32_t>& lk = stripes_[mix64(c.id ^ 0x5bd1e995ULL) & (kStripes - 1)];
    for (int n = 0;; n++) {
        uint32_t z = 0;
        if (lk.compare_exchange_weak(z, 1, std::memory_order_acquire)) break;
        if (n > 64) std::this_thread::yield(); else cpu_relax();
    }
    std::atomic<uint64_t>* r = find_entry(c);
    if (!r) {
        const uint64_t e0 = kValid | ((uint64_t)c.tag << kTagShift);
        std::atomic<uint64_t>* B1 = bk_ + c.b1 * 8;
        std::atomic<uint64_t>* B2 = bk_ + c.b2 * 8;
        int n1 = 0, n2 = 0;
        for (int i = 0; i < 8; i++) {
            n1 += B1[i].load(std::memory_order_relaxed) != 0;
            n2 += B2[i].load(std::memory_order_relaxed) != 0;
        }
        std::atomic<uint64_t>* order[2] = {B1, B2};
        uint64_t evals[2] = {e0, e0 | kSec};
        if (n2 < n1) {
            std::swap(order[0], order[1]);
            std::swap(evals[0], evals[1]);
        }
        for (int b = 0; b < 2 && !r; b++)
            for (int i = 0; i < 8 && !r; i++) {
                uint64_t z = 0;
                if (order[b][i].load(std::memory_order_relaxed) == 0 &&
                    order[b][i].compare_exchange_strong(z, evals[b], std::memory_order_acq_rel))
                    r = &order[b][i];
            }
        if (!r) r = ov_insert(c.id, e0, c.b1);
    }
    lk.store(0, std::memory_order_release);
    return r;
}

// ───────────────────────────── slots ─────────────────────────────
void SlotStore::kick_evictor() {
    if (ev_sleeping_.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> g(ev_mu_);
        ev_cv_.notify_all();
    }
}

bool SlotStore::refill(Session* S) {
    {
        std::unique_lock<std::mutex> g(pool_mu_, std::defer_lock);
        if (recycled_.load(std::memory_order_relaxed) > 0) {
            g.lock();
            if (!chains_.empty()) {
                auto ch = chains_.back();
                chains_.pop_back();
                recycled_.fetch_sub(ch.second, std::memory_order_relaxed);
                g.unlock();
                uint32_t si = ch.first;
                for (uint32_t j = 0; j < ch.second; j++) {
                    S->fcache.push_back(si);
                    if (j + 1 < ch.second) si = ald(&slot(si)->w[2]);
                }
                if (free_slots() < low_wm_) kick_evictor();
                return true;
            }
        }
    }
    if (fresh_.load(std::memory_order_relaxed) < nslots_) {
        uint64_t f = fresh_.fetch_add(256, std::memory_order_relaxed);
        if (f < nslots_) {
            uint64_t e = std::min<uint64_t>(f + 256, nslots_);
            for (uint64_t i = e; i > f; i--) S->fcache.push_back((uint32_t)(i - 1));
            if (free_slots() < low_wm_) kick_evictor();
            return true;
        }
    }
    return false;
}

uint32_t SlotStore::alloc_slot(Session* S, bool blocking) {
    for (int n = 0;; n++) {
        if (!S->fcache.empty()) {
            uint32_t si = S->fcache.back();
            S->fcache.pop_back();
            return si;
        }
        if (refill(S)) continue;
        kick_evictor();
        if (!blocking) return kNoSlot;
        S->c.slot_waits++;
        if (help_evict(S)) continue;
        if (n < 16) cpu_relax(); else std::this_thread::yield();
    }
}

// Free-list links live in word 2 (a free unit is never live, so readers ignore it).
void SlotStore::link_chain(const uint32_t* ids, size_t n) {
    for (size_t j = 0; j + 1 < n; j++) ast(&slot(ids[j])->w[2], ids[j + 1]);
}

void SlotStore::free_local(Session* S, uint32_t si) {
    S->fcache.push_back(si);
    if (S->fcache.size() >= 4096) {  // do not hoard: hand a chain back to the shared pool
        size_t n = 1024, from = S->fcache.size() - n;
        link_chain(&S->fcache[from], n);
        {
            std::lock_guard<std::mutex> g(pool_mu_);
            chains_.push_back({S->fcache[from], (uint32_t)n});
        }
        recycled_.fetch_add(n, std::memory_order_relaxed);
        S->fcache.resize(from);
    }
}

bool SlotStore::alloc_chain(Session* S, uint32_t nu, bool blocking, uint32_t* ids) {
    for (uint32_t i = 0; i < nu; i++) {
        uint32_t si = alloc_slot(S, blocking);
        if (si == kNoSlot) {
            free_ids(S, ids, i);
            return false;
        }
        ids[i] = si;
    }
    return true;
}

// Write a value into units the caller owns (allocated, unpublished). The head is left *not live*:
// the caller publishes it (sets kLive) only after the index points at it.
void SlotStore::fill_chain(const uint32_t* ids, uint32_t nu, uint64_t key, const uint8_t* data, uint32_t size,
                           uint32_t flags) {
    if (nu > 1) {
        uint32_t off = kBigHead;
        for (uint32_t j = 1; j < nu; j++) {
            Slot* c = slot(ids[j]);
            ast(&c->w[0], j + 1 < nu ? ids[j + 1] : kNoSlot);
            if (j == 1) ast(&c->w[1], size);
            uint32_t m = std::min(kContVal, size - off);
            store_words(cont_words(c), data + off, m);
            off += m;
        }
    }
    Slot* s = slot(ids[0]);
    uint32_t v = lock_slot(s);
    set_slot_key(s, key);
    if (nu > 1) {
        ast(&s->w[2], ids[1]);
        store_words(&s->w[3], data, kBigHead);
    } else {
        write_small(s, data, size);
    }
    set_ctl(s, kFlagMask | kKind, (flags & kFlagMask & ~kLive) | (is_tomb(flags) ? kTomb : size_bits(size)));
    unlock_mod(s, v);
    if (flags & kDirty) mark_dirty(ids[0]);
}

// An inline value (size <= kSlotVal) into words 2-26; a size < 100 goes into value byte 99. The
// caller sets the size kind (size_bits) in the flags.
void SlotStore::write_small(Slot* s, const uint8_t* data, uint32_t size) {
    uint32_t* v = &s->w[2];
    const uint32_t full = size / 4;
    for (uint32_t i = 0; i < full; i++) {
        uint32_t x;
        std::memcpy(&x, data + 4 * i, 4);
        ast(v + i, x);
    }
    if (size == kSlotVal) return;
    constexpr uint32_t kLast = kSlotWords - 3;  // value word 24: bytes [96,100)
    uint32_t i = full;
    if (size % 4) {
        ast(v + i, ld_part(data + 4 * i, size % 4) | (i == kLast ? size << 24 : 0u));
        i++;
    }
    if (i <= kLast) ast(v + kLast, size << 24);
}

// Continuation units of head s (caller holds the lock). Returns their count.
uint32_t SlotStore::chain_conts(const Slot* s, uint32_t* ids) const {
    uint32_t w = ctl(s);
    if ((w & kKind) != kBig) return 0;
    uint32_t sz = value_size(s, w);
    if (sz > GenValue::kMaxSize) die("corrupt continuation chain (size)", sz);
    uint32_t n = units_for(sz) - 1;
    uint32_t id = ald(&s->w[2]);
    for (uint32_t j = 0; j < n; j++) {
        if (id >= nslots_) die("corrupt continuation chain", id);
        ids[j] = id;
        id = ald(&slot(id)->w[0]);
    }
    return n;
}

// Copy a value of `size` bytes out of head s into out (needs ceil4(size) bytes of room). Works
// optimistically (validate the head's stripe version afterwards); false if the chain was torn.
bool SlotStore::copy_value(const Slot* s, uint32_t size, uint8_t* out) const {
    if (size <= kSlotVal) {
        load_words(&s->w[2], out, size);
        return true;
    }
    if (size > GenValue::kMaxSize) return false;
    uint32_t id = ald(&s->w[2]);
    load_words(&s->w[3], out, kBigHead);
    uint32_t off = kBigHead;
    while (off < size) {
        if (id >= nslots_) return false;
        const Slot* c = slot(id);
        uint32_t m = std::min(kContVal, size - off);
        load_words(cont_words(c), out + off, m);
        off += m;
        id = ald(&c->w[0]);
    }
    return true;
}

// Does locked head s hold exactly the record encoded at rec (key, tombstone, size, value)? Used to
// re-validate a slot after a write-behind instead of trusting a (wrapping) version number.
bool SlotStore::slot_matches(const Slot* s, const uint8_t* rec, uint8_t* tmp) const {
    uint32_t w = ctl(s);
    uint64_t rk, sf;
    uint16_t rsz;
    std::memcpy(&rk, rec, 8);
    std::memcpy(&sf, rec + 8, 8);
    std::memcpy(&rsz, rec + 20, 2);
    bool rtomb = (sf >> 56) & 1;
    if (!(w & kLive) || slot_key(s) != rk || is_tomb(w) != rtomb) return false;
    if (rtomb) return true;
    uint32_t sz = value_size(s, w);
    if (sz != rsz) return false;
    return copy_value(s, sz, tmp) && std::memcmp(tmp, rec + kHdr, sz) == 0;
}

// Swap the published head of key's class (a slot of the same key, entry value e) for the
// unpublished chain S->ids[0..nu). exp_val: if set, the head must still hold exactly that value
// (exp_size bytes, or a tombstone if exp_tomb). False: retry.
bool SlotStore::replace_head(Session* S, std::atomic<uint64_t>* ep, uint64_t e, uint64_t key, const uint8_t* exp_val,
                             uint32_t exp_size, bool exp_tomb) {
    uint32_t si = res_slot(e);
    Slot* s = slot(si);
    uint32_t v = lock_slot(s);
    uint32_t w = ctl(s);
    bool ok = (w & kLive) && ep->load(std::memory_order_acquire) == e && slot_key(s) == key;
    if (ok && exp_val) {
        if (is_tomb(w) != exp_tomb) ok = false;
        else if (!exp_tomb) {
            uint32_t sz = value_size(s, w);
            ok = sz == exp_size && copy_value(s, sz, S->gbuf) && std::memcmp(S->gbuf, exp_val, sz) == 0;
        }
    }
    if (!ok) {
        unlock_nomod(s, v);
        return false;
    }
    note_clean_stale(S, s, w);
    uint32_t k = chain_conts(s, S->oids);
    ep->store(res_with_slot(e, S->ids[0]), std::memory_order_release);  // same mode: same chain / D
    publish(S->ids[0], kRefMax | kDirty);
    set_ctl(s, kFlagMask, 0);
    unlock_mod(s, v);
    free_local(S, si);
    free_ids(S, S->oids, k);
    return true;
}

// ───────────────────────────── log ─────────────────────────────
void SlotStore::open_seg(uint64_t seg, bool create) {
    if (seg >= kMaxSeg) die("log exceeds maximum size", (long long)seg);
    std::lock_guard<std::mutex> g(seg_mu_);
    if (fds_[seg].load(std::memory_order_relaxed) >= 0) return;
    char name[64];
    std::snprintf(name, sizeof name, "seg_%06u.log", (unsigned)seg);
    int fd = ::openat(dirfd_, name, O_RDWR | O_DIRECT | O_NOFOLLOW | O_CLOEXEC | (create ? O_CREAT : 0), 0644);
    if (fd < 0) die("open segment", (long long)seg);
    fds_[seg].store(fd, std::memory_order_release);
}

int SlotStore::seg_fd(uint64_t seg) {
    int fd = fds_[seg].load(std::memory_order_acquire);
    if (fd < 0) {
        open_seg(seg, true);
        fd = fds_[seg].load(std::memory_order_acquire);
    }
    return fd;
}

// The lowest free segment number, marked used, its file created empty (caller holds alloc_mu_).
// A number is free only once its old file is unlinked and a manifest without it is durable.
uint32_t SlotStore::new_seg_locked() {
    for (uint32_t w = 0; w < kSegWords; w++) {
        if (seg_used_[w] == ~0ULL) continue;
        uint32_t sg = w * 64 + (uint32_t)__builtin_ctzll(~seg_used_[w]);
        seg_used_[w] |= 1ULL << (sg % 64);
        char name[64];
        std::snprintf(name, sizeof name, "seg_%06u.log", (unsigned)sg);
        int fd = ::openat(dirfd_, name, O_RDWR | O_DIRECT | O_NOFOLLOW | O_CLOEXEC | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) die("create segment", (long long)sg);
        {
            std::lock_guard<std::mutex> g(seg_mu_);
            int old = fds_[sg].exchange(fd, std::memory_order_acq_rel);
            if (old >= 0) ::close(old);
        }
        segdirty_[sg].store(1, std::memory_order_relaxed);
        seg_cnt_[sg].store(0, std::memory_order_relaxed);
        seg_dig_[sg].store(0, std::memory_order_relaxed);
        return sg;
    }
    die("log exceeds maximum size (no free segment)");
}

// Reserve len bytes of log. The tail never reaches a segment's end (strict fit), so tail_'s segment
// is always the active one. A reservation that does not fit moves the log to a fresh segment and
// extends the passed one to its full size (a zero hole), so recovery can tell a complete passed
// segment from a truncated one; the next Checkpoint fsyncs the new size (segdirty_).
uint64_t SlotStore::log_alloc(uint64_t len) {
    uint64_t cur = tail_.load(std::memory_order_relaxed);
    for (;;) {
        if ((cur & (kSegSize - 1)) + len < kSegSize) {
            if (tail_.compare_exchange_weak(cur, cur + len, std::memory_order_acq_rel)) {
                (void)seg_fd(cur >> kSegShift);
                return cur;
            }
            continue;
        }
        std::lock_guard<std::mutex> g(alloc_mu_);
        cur = tail_.load(std::memory_order_acquire);
        if ((cur & (kSegSize - 1)) + len < kSegSize) continue;  // another thread switched
        const uint64_t ps = cur >> kSegShift;
        const uint32_t ns = new_seg_locked();
        if (::ftruncate(seg_fd(ps), (off_t)kSegSize) != 0) die("extend segment", (long long)ps);
        segdirty_[ps].store(1, std::memory_order_relaxed);
        const uint64_t off = ((uint64_t)ns << kSegShift) + seg_start(ns);
        // Only a switcher (under alloc_mu_) changes the segment: a failed CAS is an allocation that
        // still fit in ps; retry against the new tail.
        while (!tail_.compare_exchange_weak(cur, off + len, std::memory_order_acq_rel)) {
        }
        return off;
    }
}

// Move the log to a fresh segment now (quiesced callers: GC, so its copies never land in a segment
// it is about to free).
void SlotStore::log_switch() {
    std::lock_guard<std::mutex> g(alloc_mu_);
    const uint64_t ps = tail_.load() >> kSegShift;
    const uint32_t ns = new_seg_locked();
    if (::ftruncate(seg_fd(ps), (off_t)kSegSize) != 0) die("extend segment", (long long)ps);
    segdirty_[ps].store(1, std::memory_order_relaxed);
    tail_.store(((uint64_t)ns << kSegShift) + seg_start(ns));
}

void SlotStore::log_write(uint64_t off, const uint8_t* buf, uint64_t len) {
    uint64_t seg = off >> kSegShift;
    int fd = seg_fd(seg);
    uint64_t so = off & (kSegSize - 1), done = 0;
    while (done < len) {
        ssize_t r = ::pwrite(fd, buf + done, len - done, (off_t)(so + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            die("log write", (long long)off);
        }
        done += (uint64_t)r;
    }
    uint64_t cnt = 0, dig = 0;
    buf_digest(buf, len, so, cnt, dig);
    seg_cnt_[seg].fetch_add(cnt, std::memory_order_relaxed);
    seg_dig_[seg].fetch_add(dig, std::memory_order_relaxed);
    segdirty_[seg].store(1, std::memory_order_relaxed);
    st_bytes_written_.fetch_add(len, std::memory_order_relaxed);
    log_bytes_.fetch_add(len, std::memory_order_relaxed);
}

uint64_t SlotStore::write_record_sync(Session* S, uint64_t key, bool tomb, uint64_t prev, const uint8_t* val,
                                      uint32_t size) {
    uint32_t bytes = kHdr + size;
    uint64_t len = round_up(bytes, align_);
    std::memset(S->wbuf, 0, len);
    enc_record(S->wbuf, key, next_seq(), tomb, prev, val, size);
    uint64_t off = log_alloc(len);
    log_write(off, S->wbuf, len);
    S->c.slow_writes++;
    S->c.live_sub += len - (tomb ? 0 : rec_charge(bytes));  // padding (and a tombstone) is dead at once
    return mk_ptr(off, bytes, tomb);
}

void SlotStore::read_range(uint64_t p, uint64_t& base, uint32_t& len) const {
    uint64_t start = ptr_byte(p);
    uint32_t ns = ptr_nsect(p);
    base = round_dn(start, align_);
    uint64_t end = ns == 1 ? round_dn(start, kSector) + kSector : start + (uint64_t)ns * kSector;
    len = (uint32_t)(round_up(end, align_) - base);
}

const uint8_t* SlotStore::read_record_sync(uint8_t* buf, uint64_t p, RecView& r) {
    uint64_t base;
    uint32_t len;
    read_range(p, base, len);
    int fd = seg_fd(base >> kSegShift);
    uint64_t so = base & (kSegSize - 1);
    uint32_t done = 0;
    while (done < len) {
        ssize_t n = ::pread(fd, buf + done, len - done, (off_t)(so + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            die("log read", (long long)base);
        }
        if (n == 0) break;
        done += (uint32_t)n;
    }
    uint64_t at = ptr_byte(p) - base;
    if (at >= done || !parse_record(buf + at, done - at, r)) die("corrupt record (crc/format mismatch) at log offset",
                                                                 (long long)ptr_byte(p));
    return buf + at;
}

// ───────────────────────────── lookups ─────────────────────────────
// Resolve `key` as far as memory allows. Disk: *p_out is the first record pointer to read; *e_out is
// the entry value if that record is the class head (install allowed), else 0.
SlotStore::LR SlotStore::lookup_mem(uint64_t key, GenValue* out, std::atomic<uint64_t>** ep_out, uint64_t* e_out,
                                    uint64_t* p_out) {
    const Cls c = classify(key);
    const uint64_t mask = kValid | kIsSlot | kSec | (kTagMask << kTagShift);
    const uint64_t want = kValid | ((uint64_t)c.tag << kTagShift);
    const uint32_t t5 = tag5(c.tag);
    __builtin_prefetch(bk_ + c.b2 * 8);
    // Candidate entries: the two buckets, then the overflow entry. A resident candidate is decided by
    // one snapshot of its head slot (key, value) taken while the entry is unchanged.
    for (int side = 0; side < 3; side++) {
        std::atomic<uint64_t>* B;
        int nb;
        if (side < 2) {
            B = bk_ + (side ? c.b2 : c.b1) * 8;
            nb = 8;
        } else {
            if (!((ovbits_[c.b1 >> 6].load(std::memory_order_acquire) >> (c.b1 & 63)) & 1)) break;
            B = ov_find(c.id);
            if (!B) break;
            nb = 1;
        }
        const uint64_t wd = side == 1 ? (want | kSec) : want;
        if (side < 2) prefetch_candidates(B, 8, t5);
        for (int i = 0; i < nb; i++) {
            std::atomic<uint64_t>* ep = &B[i];
            for (;;) {
                uint64_t e = ep->load(std::memory_order_acquire);
                if (!(e & kIsSlot)) {
                    if (side < 2 && (e & mask) != wd) break;  // not ours (overflow: found by class id)
                    if (side == 2 && !e) break;
                    *ep_out = ep;
                    uint64_t p = e & kPay;
                    if (!p) return LR::NotFound;
                    *p_out = p;
                    *e_out = e;
                    return LR::Disk;
                }
                if (side < 2 && res_tag5(e) != t5) break;
                const uint32_t si = res_slot(e);
                Slot* s = slot(si);
                std::atomic<uint32_t>& L = lockw(si);
                uint32_t fl;
                uint64_t sk;
                bool match, torn;
                for (;;) {
                    uint64_t t0 = win_begin();
                    uint32_t v1 = L.load(std::memory_order_acquire);
                    if (v1 & kLockBit) {
                        cpu_relax();
                        continue;
                    }
                    fl = flg(si).load(std::memory_order_relaxed);
                    sk = slot_key(s);
                    match = (fl & kLive) && sk == key;
                    torn = false;
                    if (match && out && !is_tomb(fl)) {
                        uint32_t sz = value_size(s, fl);
                        torn = !copy_value(s, sz, out->data);
                        out->size = sz;
                    }
                    std::atomic_thread_fence(std::memory_order_acquire);
                    if (L.load(std::memory_order_relaxed) == v1 && win_end() - t0 < kReadWindowTicks) break;
                }
                // The entry must still be e: then the snapshot is of e's head and chain_start(e) is its
                // chain. Not live: recycled, or not yet published; re-read the entry.
                if (ep->load(std::memory_order_acquire) != e || !(fl & kLive)) continue;
                if (match) {
                    if (torn) die("torn value chain under a stable slot version", (long long)res_slot(e));
                    if ((fl & kRefMask) != kRefMax) flg(si).fetch_or((uint8_t)kRefMax, std::memory_order_relaxed);
                    *ep_out = ep;
                    return is_tomb(fl) ? LR::NotFound : LR::Hit;
                }
                if (side < 2 && classify(sk).id != c.id) break;  // another class's entry (short tag)
                // Another key of our class heads it.
                *ep_out = ep;
                uint64_t prev = chain_start(e);
                if (!prev) return LR::NotFound;
                *p_out = prev;
                *e_out = 0;
                return LR::Disk;
            }
        }
    }
    return LR::NotFound;
}

// Walk a disk chain from p. *hit_prev is set only when the hit is the first record (p itself): only
// then may the caller install it as the class head (a deeper hit would unlink the members before it).
bool SlotStore::chain_lookup_sync(Session* S, uint64_t p, uint64_t key, GenValue* out, uint64_t* hit_prev,
                                  uint64_t* hit_charge) {
    bool first = true;
    if (hit_prev) *hit_prev = kNoInstall;
    if (hit_charge) *hit_charge = 0;
    while (p) {
        RecView r;
        read_record_sync(S->sbuf, p, r);
        if (r.key == key) {
            if (r.tomb) return false;
            if (hit_charge) *hit_charge = rec_charge(kHdr + r.size);
            if (out) {
                std::memcpy(out->data, r.val, r.size);
                out->size = r.size;
            }
            if (hit_prev && first) *hit_prev = r.prev;
            return true;
        }
        if (p & kPtrT) return false;
        p = r.prev;
        first = false;
    }
    return false;
}

// Cache the record p just read (key's class head, entry value head_e) in a slot: clean, with D = p
// in the entry. A record that does not fit an entry is cached dirty (no chain) only when it is solo.
void SlotStore::try_install(Session* S, std::atomic<uint64_t>* ep, uint64_t head_e, uint64_t p, uint64_t key,
                            const uint8_t* val, uint32_t size, uint64_t rec_prev) {
    if (!head_e || rec_prev == kNoInstall || (head_e & kIsSlot) || (head_e & kPay) != p) return;
    if (ep->load(std::memory_order_relaxed) != head_e) return;
    const bool solo = (p & kPtrT) || rec_prev == 0;
    const bool fits = ptr_fits(p & ~kPtrT);
    if (!fits && !solo) return;
    uint32_t nu = units_for(size);
    if (!alloc_chain(S, nu, false, S->ids)) return;
    const uint32_t t5 = tag5(head_e >> kTagShift);
    uint64_t ne;
    if (fits) {
        fill_chain(S->ids, nu, key, val, size, kRefMax);
        ne = mk_res(solo ? kMDSolo : kMD, S->ids[0], p & kAddrMask, t5);
    } else {
        fill_chain(S->ids, nu, key, val, size, kRefMax | kDirty);
        ne = mk_res(kMChain, S->ids[0], 0, t5);  // A = 0: no chain
    }
    uint64_t exp = head_e;
    if (ep->compare_exchange_strong(exp, ne, std::memory_order_acq_rel)) {
        publish(S->ids[0]);
        S->c.installs++;
    } else {
        free_ids(S, S->ids, nu);
    }
}

bool SlotStore::Read(uint64_t key, GenValue& out) {
    Session* S = sess();
    OpGuard g(this, S);
    std::atomic<uint64_t>* ep = nullptr;
    uint64_t e = 0, p = 0;
    LR lr = lookup_mem(key, &out, &ep, &e, &p);
    if (lr == LR::Hit) {
        S->c.read_mem++;
        sample_touch(key);
        return true;
    }
    if (lr == LR::NotFound) {
        S->c.read_notfound++;
        sample_touch(key);
        return false;
    }
    S->c.ios++;
    sample_done(sample_cold(key));
    uint64_t rprev = 0;
    bool found = chain_lookup_sync(S, p, key, &out, &rprev);
    S->c.reaped++;
    if (found) {
        S->c.read_disk_found++;
        try_install(S, ep, e, p, key, out.data, out.size, rprev);
    } else {
        S->c.read_notfound++;
    }
    return found;
}

// ───────────────────────────── async reads ─────────────────────────────
bool SlotStore::ensure_ring(Session* S) {
    if (S->ring_ok) return true;
    if (S->ring_tried) return false;
    S->ring_tried = true;
#if KV_HAVE_URING
#ifdef KVSTORE_SYNC_IO
    return false;  // sanitizer builds: kernel-side buffer writes are invisible to the sanitizer
#endif
    if (std::getenv("KVSTORE_NO_URING")) return false;
    if (!S->ring.init(kMaxPend)) return false;
    S->bufs = aligned_buf((size_t)kMaxPend * rd_unit_);
    S->pend.assign(kMaxPend, Pend{});
    S->pfree.clear();
    for (uint32_t i = 0; i < kMaxPend; i++) S->pfree.push_back(kMaxPend - 1 - i);
    S->queued = S->npend = 0;
    S->ring_ok = true;
    return true;
#else
    return false;
#endif
}

void SlotStore::submit_read(Session* S, uint32_t id) {
#if KV_HAVE_URING
    Pend& pd = S->pend[id];
    read_range(pd.p, pd.base, pd.len);
    if (pd.len <= rd_unit_) {
        pd.buf = S->bufs + (size_t)id * rd_unit_;
        pd.heap = false;
    } else {
        pd.buf = aligned_buf(pd.len);
        pd.heap = true;
    }
    int fd = seg_fd(pd.base >> kSegShift);
    pd.t_sub = now_ns();
    S->ring.prep_read(fd, pd.buf, pd.len, pd.base & (kSegSize - 1), id);
    S->c.ios++;
    S->c.depth_sum += S->npend;
    if (++S->queued >= kSubmitBatch) {
        S->c.enters++;
        S->ring.enter(S->queued, 0);
        S->queued = 0;
    }
#else
    (void)S;
    (void)id;
#endif
}

void SlotStore::process(Session* S, uint32_t id, int res, int64_t now, int64_t poll_prev) {
    Pend& pd = S->pend[id];
    S->c.reaped++;
    if (run_started_.load(std::memory_order_relaxed) && !run_printed_.load(std::memory_order_relaxed)) {
        int64_t lat = now - pd.t_sub, lag = now - std::max(poll_prev, pd.t_sub);
        if (lat < 1) lat = 1;
        if (lag < 1) lag = 1;
        S->c.lat_n++;
        S->c.lat_sum_ns += (uint64_t)lat;
        S->c.lag_sum_ns += (uint64_t)lag;
        S->c.lat_h[std::min(Counters::kHB - 1, 63 - __builtin_clzll((uint64_t)lat))]++;
        S->c.lag_h[std::min(Counters::kHB - 1, 63 - __builtin_clzll((uint64_t)lag))]++;
    }
    uint64_t at = ptr_byte(pd.p) - pd.base;
    ReadSlot* rs = pd.rs;
    bool finished = true;
    // A log GC ran since pd.p was resolved: the buffer (read from the old file) is still a valid
    // record of the old log, but its chain (prev) and the failed read's retry address are gone.
    // Serve a direct hit or a chain end as of the read's issue; anything else re-resolves the key
    // from the current index.
    const bool stale = pd.gen != gc_gen_.load(std::memory_order_relaxed);
    RecView r;
    if (stale && (res < 0 || (uint64_t)res <= at || !parse_record(pd.buf + at, (size_t)res - at, r) ||
                  (r.key != pd.key && !(pd.p & kPtrT) && r.prev != 0))) {
        std::atomic<uint64_t>* ep = nullptr;
        uint64_t e = 0, p = 0;
        LR lr = lookup_mem(pd.key, &rs->out, &ep, &e, &p);
        if (lr == LR::Disk) {
            if (pd.heap) free(pd.buf);
            pd.p = p;
            pd.ep = ep;
            pd.head_e = e;
            pd.gen = gc_gen_.load(std::memory_order_relaxed);
            submit_read(S, id);
            return;
        }
        if (lr == LR::Hit) {
            rs->status = OpStatus::Ok;
            S->c.read_mem++;
        } else {
            rs->status = OpStatus::NotFound;
            S->c.read_notfound++;
        }
        if (pd.heap) free(pd.buf);
        pd.buf = nullptr;
        sample_done(pd.sidx);
        rs->done.store(1, std::memory_order_release);
        S->pfree.push_back(id);
        S->npend--;
        return;
    }
    if (res < 0) {  // retry synchronously
        errno = -res;
        int fd = seg_fd(pd.base >> kSegShift);
        ssize_t n = ::pread(fd, pd.buf, pd.len, (off_t)(pd.base & (kSegSize - 1)));
        if (n < 0) die("async read failed", res);
        res = (int)n;
    }
    if ((uint64_t)res <= at || !parse_record(pd.buf + at, (size_t)res - at, r))
        die("corrupt record (crc/format mismatch) at log offset", (long long)ptr_byte(pd.p));
    if (r.key == pd.key) {
        if (r.tomb) {
            rs->status = OpStatus::NotFound;
            S->c.read_notfound++;
        } else {
            std::memcpy(rs->out.data, r.val, r.size);
            rs->out.size = r.size;
            rs->status = OpStatus::Ok;
            S->c.read_disk_found++;
            try_install(S, pd.ep, pd.head_e, pd.p, pd.key, r.val, r.size, r.prev);
        }
    } else if ((pd.p & kPtrT) || r.prev == 0) {
        rs->status = OpStatus::NotFound;
        S->c.read_notfound++;
    } else {
        finished = false;
        uint64_t nx = r.prev;
        if (pd.heap) free(pd.buf);
        pd.p = nx;
        pd.head_e = 0;
        S->c.hops++;
        submit_read(S, id);
    }
    if (finished) {
        if (pd.heap) free(pd.buf);
        pd.buf = nullptr;
        sample_done(pd.sidx);
        rs->done.store(1, std::memory_order_release);
        S->pfree.push_back(id);
        S->npend--;
    }
}

void SlotStore::drive(Session* S, bool wait) {
#if KV_HAVE_URING
    if (!S->ring_ok) return;
    for (;;) {
        unsigned want = (wait && S->npend) ? 1 : 0;
        if (S->queued || want) {
            S->c.enters++;
            S->ring.enter(S->queued, want);
            S->queued = 0;
        }
        int64_t now = now_ns(), prev = S->last_poll_ns;
        S->ring.reap([&](const io_uring_cqe& c) { process(S, (uint32_t)c.user_data, c.res, now, prev); });
        S->last_poll_ns = now;
        if (!wait || S->npend == 0) break;
    }
    if (S->queued) {
        S->c.enters++;
        S->ring.enter(S->queued, 0);
        S->queued = 0;
    }
#else
    (void)S;
    (void)wait;
#endif
}

OpStatus SlotStore::ReadAsync(ReadSlot* rs) {
    Session* S = sess();
    if (!run_started_.load(std::memory_order_relaxed) && !run_started_.exchange(true)) {
        run_t0_ = std::chrono::steady_clock::now();
        run_resident_ = std::min<uint64_t>(fresh_.load(), nslots_) - recycled_.load();
        run_evict_ = st_evict_.load();
        run_bytes_ = st_bytes_written_.load();
        Counters c0 = sum_counters();
        run_ios_ = c0.ios;
        run_waits_ = c0.slot_waits;
        run_helps_ = c0.help_evicts;
        run_depth_ = c0.depth_sum;
        run_enters_ = c0.enters;
        run_ops_ = c0.read_mem + c0.read_disk_found + c0.read_notfound + c0.ups_inplace + c0.ups_push + c0.rmw_hit +
                   c0.rmw_miss;
        run_drop_ = st_drop_clean_.load();
        run_dirty_ = st_evict_dirty_.load();
        run_clean0_ = st_cleaned_.load();
        run_ready_.store(true, std::memory_order_release);
    }
    OpGuard g(this, S);
    std::atomic<uint64_t>* ep = nullptr;
    uint64_t e = 0, p = 0;
    LR lr = lookup_mem(rs->key, &rs->out, &ep, &e, &p);
    if (lr == LR::Hit || lr == LR::NotFound) {
        if (lr == LR::Hit) S->c.read_mem++; else S->c.read_notfound++;
        sample_touch(rs->key);
        rs->status = lr == LR::Hit ? OpStatus::Ok : OpStatus::NotFound;
        rs->done.store(1, std::memory_order_release);
        return rs->status;
    }
    if (!ensure_ring(S)) {
        S->c.ios++;
        sample_done(sample_cold(rs->key));
        uint64_t rprev = 0;
        bool found = chain_lookup_sync(S, p, rs->key, &rs->out, &rprev);
        S->c.reaped++;
        if (found) {
            S->c.read_disk_found++;
            try_install(S, ep, e, p, rs->key, rs->out.data, rs->out.size, rprev);
        } else {
            S->c.read_notfound++;
        }
        rs->status = found ? OpStatus::Ok : OpStatus::NotFound;
        rs->done.store(1, std::memory_order_release);
        return rs->status;
    }
    while (S->pfree.empty()) drive(S, true);
    uint32_t id = S->pfree.back();
    S->pfree.pop_back();
    S->npend++;
    Pend& pd = S->pend[id];
    pd.rs = rs;
    pd.key = rs->key;
    pd.p = p;
    pd.ep = ep;
    pd.head_e = e;
    pd.gen = gc_gen_.load(std::memory_order_relaxed);
    pd.buf = nullptr;
    pd.heap = false;
    pd.sidx = sample_cold(rs->key);
    submit_read(S, id);
    return OpStatus::Pending;
}

void SlotStore::CompletePending(bool wait) {
    Session* S = sess();
    if (!S->ring_ok || (S->npend == 0 && S->queued == 0)) return;
    OpGuard g(this, S);
    drive(S, wait);
}

// ───────────────────────────── writes ─────────────────────────────
// Evict a class-head slot synchronously (class collision path). Caller holds no lock.
void SlotStore::flush_slot_sync(Session* S, uint32_t si, std::atomic<uint64_t>* ep, uint64_t e) {
    Slot* s = slot(si);
    uint32_t v = lock_slot(s);
    uint32_t w = ctl(s);
    if (!(w & kLive) || ep->load(std::memory_order_acquire) != e) {
        unlock_nomod(s, v);
        return;
    }
    const uint64_t hdr = disk_hdr(ep, classify(slot_key(s)));
    uint64_t ptr;
    if (is_clean(w)) {
        if (!res_has_d(e)) die("clean slot without a disk record", (long long)si);
        ptr = clean_disk_ptr(e, s, w);  // the disk copy is current: no write
    } else {
        bool tomb = is_tomb(w);
        uint32_t sz = tomb ? 0 : value_size(s, w);
        if (!tomb) copy_value(s, sz, S->gbuf);
        ptr = write_record_sync(S, slot_key(s), tomb, chain_start(e), S->gbuf, sz);
    }
    uint32_t k = chain_conts(s, S->oids);
    ep->store(hdr | ptr, std::memory_order_release);
    set_ctl(s, kFlagMask, 0);
    unlock_mod(s, v);
    free_local(S, si);
    free_ids(S, S->oids, k);
    st_flush_sync_.fetch_add(1, std::memory_order_relaxed);
}

// Make (key, value) the head of key's class whose entry is the disk entry e. The new head slot
// chains to the old head through its entry (mode); an old head that does not fit an entry is
// chained by writing the new record straight to the log instead. False: the entry moved, retry.
// `have`: S->ids[0..nu) already holds the filled, unpublished chain (consumed on success).
bool SlotStore::push_head(Session* S, std::atomic<uint64_t>* ep, uint64_t e, uint64_t key, const uint8_t* data,
                          uint32_t size, bool tomb, uint32_t nu, bool& have, uint32_t ref, uint64_t stale) {
    const uint64_t p = e & kPay;
    if (p && !ptr_fits(p)) {
        uint64_t np = write_record_sync(S, key, tomb, p, data, size);
        uint64_t exp = e;
        if (!ep->compare_exchange_strong(exp, (e & kHdrBits) | np, std::memory_order_acq_rel)) {
            if (!tomb) S->c.live_sub += rec_charge(kHdr + size);  // never referenced: dead
            return false;
        }
        note_push_stale(S, p, stale);
        if (have) free_ids(S, S->ids, nu);
        have = false;
        return true;
    }
    if (!have) {
        alloc_chain(S, nu, true, S->ids);
        fill_chain(S->ids, nu, key, data, size, kDirty | ref | (tomb ? kTomb : 0));
        have = true;
    }
    uint64_t ne = mk_res(push_mode(p), S->ids[0], p & kAddrMask, tag5(e >> kTagShift));
    uint64_t exp = e;
    if (!ep->compare_exchange_strong(exp, ne, std::memory_order_acq_rel)) return false;
    publish(S->ids[0]);
    note_push_stale(S, p, stale);
    have = false;
    return true;
}

// Make (key, data, size) the head of key's class. Small overwrites of a resident small value are
// done in place under the slot lock; everything else writes a fresh chain of units and swaps it in.
void SlotStore::put_value(Session* S, uint64_t key, const uint8_t* data, uint32_t size) {
    const Cls c = classify(key);
    std::atomic<uint64_t>* ep = entry_for(c);
    const uint32_t nu = units_for(size);
    bool have = false;  // S->ids[0..nu) holds a filled, unpublished chain
    for (;;) {
        uint64_t e = ep->load(std::memory_order_acquire);
        if (__builtin_expect(stale_disk(ep, e, c), 0)) {  // entry reclaimed since we found it
            ep = entry_for(c);
            continue;
        }
        if (e & kIsSlot) {
            uint32_t si = res_slot(e);
            Slot* s = slot(si);
            prefetch_w(s);
            uint32_t v = lock_slot(s);
            uint32_t w = ctl(s);
            if (!(w & kLive) || ep->load(std::memory_order_acquire) != e) {
                unlock_nomod(s, v);
                continue;
            }
            if (slot_key(s) != key) {
                unlock_nomod(s, v);
                flush_slot_sync(S, si, ep, e);
                continue;
            }
            if (nu == 1 && (w & kKind) != kBig) {
                note_clean_stale(S, s, w);
                write_small(s, data, size);
                set_ctl(s, kKind | kDirty | kRefMask, size_bits(size) | kDirty | kRefMax);
                unlock_mod(s, v);
                mark_dirty(si);
                if (have) free_ids(S, S->ids, nu);
                S->c.ups_inplace++;
                return;
            }
            unlock_nomod(s, v);
            if (!have) {
                alloc_chain(S, nu, true, S->ids);
                fill_chain(S->ids, nu, key, data, size, kDirty | kRef1);
                have = true;
            }
            if (replace_head(S, ep, e, key, nullptr, 0, false)) {
                S->c.ups_inplace++;
                return;
            }
            continue;
        }
        if (push_head(S, ep, e, key, data, size, false, nu, have, kRef1, kBlindPush)) {
            S->c.ups_push++;
            return;
        }
    }
}

void SlotStore::Upsert(uint64_t key, const GenValue& value) {
    Session* S = sess();
    OpGuard g(this, S);
    uint32_t size = std::min<uint32_t>(value.size, GenValue::kMaxSize);
    sample_touch(key);
    put_value(S, key, value.data, size);
}

void SlotStore::RMW(uint64_t key, const uint8_t* mod_data, size_t mod_size) {
    Session* S = sess();
    OpGuard g(this, S);
    uint64_t delta = 0;
    std::memcpy(&delta, mod_data, std::min<size_t>(mod_size, 8));
    sample_touch(key);
    const Cls c = classify(key);
    std::atomic<uint64_t>* ep = entry_for(c);
    GenValue* t = S->tmpv;
    auto apply = [&](bool found) {
        if (!found) {
            t->size = 8;
            std::memcpy(t->data, &delta, 8);
            return;
        }
        if (t->size < 8) {
            std::memset(t->data + t->size, 0, 8 - t->size);
            t->size = 8;
        }
        uint64_t x;
        std::memcpy(&x, t->data, 8);
        x += delta;
        std::memcpy(t->data, &x, 8);
    };
    for (;;) {
        uint64_t e = ep->load(std::memory_order_acquire);
        uint32_t old_size = 0;
        uint64_t hit_charge = 0;  // live charge of the key's disk record the push replaces
        if (__builtin_expect(stale_disk(ep, e, c), 0)) {  // entry reclaimed since we found it
            ep = entry_for(c);
            continue;
        }
        if (e & kIsSlot) {
            uint32_t si = res_slot(e);
            Slot* s = slot(si);
            prefetch_w(s);
            uint32_t v = lock_slot(s);
            uint32_t w = ctl(s);
            if (!(w & kLive) || ep->load(std::memory_order_acquire) != e) {
                unlock_nomod(s, v);
                continue;
            }
            if (slot_key(s) != key) {
                unlock_nomod(s, v);
                flush_slot_sync(S, si, ep, e);
                continue;
            }
            bool tomb = is_tomb(w);
            old_size = tomb ? 0 : value_size(s, w);
            t->size = old_size;
            if (!tomb) copy_value(s, t->size, t->data);
            if (old_size > kSlotVal) std::memcpy(S->sbuf, t->data, old_size);  // expected value for the swap
            apply(!tomb);
            if (t->size <= kSlotVal && old_size <= kSlotVal) {
                note_clean_stale(S, s, w);
                write_small(s, t->data, t->size);
                set_ctl(s, kKind | kDirty | kRefMask, size_bits(t->size) | kDirty | kRefMax);
                unlock_mod(s, v);
                mark_dirty(si);
                S->c.rmw_hit++;
                return;
            }
            // Large value: build the new chain unlocked, swap it in only if the head still holds the
            // value we read (compared by content, not by version).
            unlock_nomod(s, v);
        } else {
            apply(chain_lookup_sync(S, e & kPay, key, t, nullptr, &hit_charge));
        }
        uint32_t nu = units_for(t->size);
        bool have = false;
        if (e & kIsSlot) {
            alloc_chain(S, nu, true, S->ids);
            fill_chain(S->ids, nu, key, t->data, t->size, kDirty | kRefMax);
            have = true;
            if (replace_head(S, ep, e, key, S->sbuf, old_size, false)) {
                S->c.rmw_hit++;
                return;
            }
        } else if (push_head(S, ep, e, key, t->data, t->size, false, nu, have, kRefMax, hit_charge)) {
            S->c.rmw_miss++;
            return;
        }
        if (have) free_ids(S, S->ids, nu);
    }
}

bool SlotStore::Delete(uint64_t key) {
    bool r = delete_op(key);
    maybe_compact_ov();  // outside the op: compaction quiesces every session
    return r;
}

bool SlotStore::delete_op(uint64_t key) {
    Session* S = sess();
    OpGuard g(this, S);
    S->c.deletes++;
    sample_touch(key);
    const Cls c = classify(key);
    std::atomic<uint64_t>* ep = find_entry(c);
    if (!ep) return false;
    for (;;) {
        uint64_t e = ep->load(std::memory_order_acquire);
        uint64_t p;
        if (__builtin_expect(stale_disk(ep, e, c), 0)) {  // entry reclaimed since we found it
            ep = find_entry(c);
            if (!ep) return false;  // the class is empty now
            continue;
        }
        if (e & kIsSlot) {
            uint32_t si = res_slot(e);
            Slot* s = slot(si);
            prefetch_w(s);
            uint32_t v = lock_slot(s);
            uint32_t w = ctl(s);
            if (!(w & kLive) || ep->load(std::memory_order_acquire) != e) {
                unlock_nomod(s, v);
                continue;
            }
            if (slot_key(s) == key) {
                if (is_tomb(w)) {
                    unlock_nomod(s, v);
                    return false;
                }
                // Tombstone in place; it reaches the log on eviction or checkpoint.
                note_clean_stale(S, s, w);
                uint32_t k = chain_conts(s, S->oids);
                set_ctl(s, kKind | kDirty, kTomb | kDirty);
                unlock_mod(s, v);
                mark_dirty(si);
                free_ids(S, S->oids, k);
                // No chain: the class now holds only this deleted key. Queue it; a batch writes the
                // tombstones and frees their entries and slots, so key churn keeps the index bounded.
                if (chain_start(e) == 0) {
                    S->tpend.push_back(si);
                    if (S->tpend.size() >= kTombBatch) reclaim_tombs(S);
                }
                return true;
            }
            uint64_t prev = chain_start(e);
            unlock_nomod(s, v);
            // Another key of our class is resident: only if ours is on its chain do we have to
            // move that slot out of the way to put our tombstone at the head.
            if (!chain_lookup_sync(S, prev, key, nullptr)) {
                if (ep->load(std::memory_order_acquire) == e) return false;
                continue;
            }
            flush_slot_sync(S, si, ep, e);
            continue;
        }
        p = e & kPay;
        if (!p) return false;
        uint64_t hit_charge = 0;  // the key's record this tombstone makes non-current
        if (!chain_lookup_sync(S, p, key, nullptr, nullptr, &hit_charge)) {
            if (ep->load(std::memory_order_acquire) == e) return false;
            continue;
        }
        bool have = false;
        if (push_head(S, ep, e, key, nullptr, 0, true, 1, have, 0, hit_charge)) return true;
        if (have) free_ids(S, S->ids, 1);
    }
}

// ───────────────────────────── evictor ─────────────────────────────
// One CLOCK pass over claimed chunks of the arena. A slot's counter is decremented on each visit;
// at 0 it is a victim. A clean victim (its record D is current) is dropped at once: the index goes
// back to D, no I/O. A dirty victim is encoded into `buf` (allow_write only), appended in one write,
// and only then unlinked (index -> new record). Workers call it with allow_write = false (CPU only).
size_t SlotStore::evict_batch(uint8_t* buf, size_t cap, size_t want, std::vector<Victim>& vic,
                              std::vector<uint32_t>& freed, bool allow_write, uint64_t scan_limit) {
    vic.clear();
    freed.clear();
    alignas(8) uint8_t tmp[GenValue::kMaxSize + 8];
    uint32_t conts[kMaxUnits];
    uint64_t pos = 0, scanned = 0, ndrop = 0;
    bool full = !allow_write;
    while (ndrop + vic.size() < want && scanned < scan_limit && !(allow_write && full)) {
        uint64_t start = hand_.fetch_add(kHandChunk, std::memory_order_relaxed) % nslots_;
        uint64_t end = std::min<uint64_t>(start + kHandChunk, nslots_);
        for (uint64_t si = start; si < end; si++) {
            scanned++;
            Slot* s = slot(si);
            uint8_t w8 = flg((uint32_t)si).load(std::memory_order_relaxed);
            uint32_t w = w8;
            if (!(w & kLive)) continue;
            if (w & kRefMask) {
                // CAS on the side byte: lock holders change its other bits by CAS too, so neither is lost.
                flg((uint32_t)si).compare_exchange_weak(w8, (uint8_t)(w - kRef1), std::memory_order_relaxed);
                continue;
            }
            if (ndrop + vic.size() >= want) continue;  // finish the chunk's sweep only
            if (full && !is_clean(w)) continue;
            uint32_t v;
            if (!try_lock_slot(s, v)) continue;
            w = ctl(s);
            if (!(w & kLive) || (w & kRefMask) || (full && !is_clean(w))) {
                unlock_nomod(s, v);
                continue;
            }
            uint64_t key = slot_key(s);
            Cls c;
            std::atomic<uint64_t>* ep = find_own_entry(key, (uint32_t)si, &c);
            if (!ep) {
                unlock_nomod(s, v);
                continue;
            }
            uint64_t e = ep->load(std::memory_order_acquire);
            const uint64_t hdr = disk_hdr(ep, c);
            if (is_clean(w)) {
                if (!res_has_d(e)) die("clean slot without a disk record", (long long)si);
                uint32_t k = chain_conts(s, conts);
                // A clean tombstone with no chain: its class is empty, the tombstone is in the log.
                bool reclaim = is_tomb(w) && res_mode(e) == kMDSolo;
                ep->store(reclaim ? 0 : (hdr | clean_disk_ptr(e, s, w)), std::memory_order_release);
                if (reclaim) note_reclaim(ep);
                set_ctl(s, kFlagMask, 0);
                unlock_mod(s, v);
                freed.push_back((uint32_t)si);
                freed.insert(freed.end(), conts, conts + k);
                ndrop++;
                continue;
            }
            bool tomb = is_tomb(w);
            uint32_t sz = tomb ? 0 : value_size(s, w);
            uint32_t bytes = kHdr + sz;
            uint64_t at = place(pos, bytes);
            if (at + round_up(bytes, kUnit) > cap) {
                unlock_nomod(s, v);
                full = true;
                continue;
            }
            if (!wb_begin(s)) {  // another write-behind of this slot is in flight
                unlock_nomod(s, v);
                continue;
            }
            if (!tomb) copy_value(s, sz, tmp);
            uint64_t chain = chain_start(e);
            enc_record(buf + at, key, next_seq(), tomb, chain, tmp, sz);
            unlock_nomod(s, v);
            vic.push_back({(uint32_t)si, (uint32_t)at, bytes, ep, e, chain, hdr, v & kVerMask});
            pos = at + round_up(bytes, kUnit);
        }
    }
    if (!vic.empty()) {
        uint64_t off = write_victims(buf, pos, batch_live(buf, vic.data(), vic.size()));
        commit_stall();
        uint64_t nd = 0, dead = 0;
        for (const Victim& x : vic) {
            Slot* s = slot(x.si);
            uint32_t v = lock_slot(s);
            uint64_t cur = x.ep->load(std::memory_order_acquire);
            // Evict only if the slot was not modified since it was encoded (version) and still holds
            // exactly what was written (content: the version wraps). Content alone is not enough: a
            // modify + revert in between may have had a newer record of the slot written by a
            // cleaner, which recovery (highest seq wins) would prefer to this one.
            const bool ok = wb_unmodified(s) && cur == x.e && slot_matches(s, buf + x.boff, tmp);
            wb_end(s);
            if (ok) {
                uint32_t k = chain_conts(s, conts);
                const bool tomb = is_tomb(ctl(s));
                bool reclaim = tomb && x.chain == 0;  // written tombstone, no chain
                x.ep->store(reclaim ? 0 : (x.hdr | mk_ptr(off + x.boff, x.bytes, tomb)), std::memory_order_release);
                if (reclaim) note_reclaim(x.ep);
                set_ctl(s, kFlagMask, 0);
                unlock_mod(s, v);
                freed.push_back(x.si);
                freed.insert(freed.end(), conts, conts + k);
                nd++;
            } else {
                unlock_nomod(s, v);
                st_evict_abort_.fetch_add(1, std::memory_order_relaxed);
                dead += live_charge(buf + x.boff, x.bytes);  // written, never referenced
            }
        }
        if (dead) dead_bytes_.fetch_add(dead, std::memory_order_relaxed);
        st_evict_dirty_.fetch_add(nd, std::memory_order_relaxed);
        // Gaps between records must read as zero for recovery; clear what we used.
        std::memset(buf, 0, pos);
    }
    if (ndrop) st_drop_clean_.fetch_add(ndrop, std::memory_order_relaxed);
    st_evict_.fetch_add(freed.size(), std::memory_order_relaxed);
    return freed.size();
}

// Reclaim the queued tombstone heads of S (called inside an op). Each is re-validated under its slot
// lock: still a live tombstone head of a bucket entry with no chain. A dirty one gets its tombstone
// record (seq taken under the lock, so it is newer than any record of the key already encoded by a
// cleaner or evictor) appended in one write; then, if the slot and entry are unchanged, the entry is
// freed (0) and the slot returned. A clean one (its tombstone already in the log) is freed at once.
void SlotStore::reclaim_tombs(Session* S) {
    if (!S->tbuf) S->tbuf = aligned_buf(kTombBatch * kUnit);  // zeroed; O_DIRECT aligned
    uint8_t* buf = S->tbuf;
    std::vector<Victim> vic(S->tpend.size());
    size_t nv = 0;
    uint64_t pos = 0;
    for (uint32_t si : S->tpend) {
        Slot* s = slot(si);
        uint32_t v = lock_slot(s);
        uint32_t w = ctl(s);
        if (!(w & kLive) || !is_tomb(w)) {
            unlock_nomod(s, v);
            continue;
        }
        uint64_t key = slot_key(s);
        Cls c;
        std::atomic<uint64_t>* ep = find_own_entry(key, si, &c);
        uint64_t e = ep ? ep->load(std::memory_order_acquire) : 0;
        if (!ep || chain_start(e) != 0) {
            unlock_nomod(s, v);
            continue;
        }
        if (is_clean(w)) {
            if (!res_has_d(e)) die("clean slot without a disk record", (long long)si);
            ep->store(0, std::memory_order_release);
            note_reclaim(ep);
            set_ctl(s, kFlagMask, 0);
            unlock_mod(s, v);
            free_ids(S, &si, 1);
            continue;
        }
        if (!wb_begin(s)) {  // a cleaner or evictor is writing it behind; it reclaims it later
            unlock_nomod(s, v);
            continue;
        }
        uint64_t at = place(pos, kHdr);
        enc_record(buf + at, key, next_seq(), true, 0, nullptr, 0);
        unlock_nomod(s, v);
        vic[nv++] = {si, (uint32_t)at, (uint32_t)kHdr, ep, e, 0, 0, v & kVerMask};
        pos = at + kUnit;
    }
    S->tpend.clear();
    if (!nv) return;
    uint64_t len = round_up(pos, align_);
    uint64_t off = write_victims(buf, pos, 0);  // tombstones only: all dead once durable
    (void)off;
    for (size_t i = 0; i < nv; i++) {
        const Victim& x = vic[i];
        Slot* s = slot(x.si);
        uint32_t v = lock_slot(s);
        // Version and content unchanged: no modification since the tombstone was encoded (a
        // re-upsert + re-delete in between could have had a newer record of the key written).
        const bool ok = wb_unmodified(s) && x.ep->load(std::memory_order_acquire) == x.e &&
                        slot_matches(s, buf + x.boff, S->gbuf);
        wb_end(s);
        if (ok) {
            x.ep->store(0, std::memory_order_release);
            note_reclaim(x.ep);
            set_ctl(s, kFlagMask, 0);
            unlock_mod(s, v);
            free_ids(S, &x.si, 1);
        } else {
            unlock_nomod(s, v);
        }
    }
    std::memset(buf, 0, len);
}

// Append buf[0, pos) (zero-padded to the O_DIRECT unit) to the log; returns its offset.
uint64_t SlotStore::write_victims(uint8_t* buf, uint64_t pos, uint64_t live) {
    uint64_t len = round_up(pos, align_);
    if (len > live) dead_bytes_.fetch_add(len - live, std::memory_order_relaxed);
    std::memset(buf + pos, 0, len - pos);
    uint64_t off = log_alloc(len);
    log_write(off, buf, len);
    return off;
}

void SlotStore::push_chains(const std::vector<uint32_t>& freed, size_t from) {
    size_t i = from;
    while (i < freed.size()) {
        size_t m = std::min<size_t>(256, freed.size() - i);
        link_chain(&freed[i], m);
        {
            std::lock_guard<std::mutex> g(pool_mu_);
            chains_.push_back({freed[i], (uint32_t)m});
        }
        recycled_.fetch_add(m, std::memory_order_relaxed);
        i += m;
    }
}

// A worker with no free slot: drop clean victims itself (CPU only, never I/O), a few chunks at most.
bool SlotStore::help_evict(Session* S) {
    size_t n = evict_batch(nullptr, 0, 256, S->evic, S->efreed, false, 16 * kHandChunk);
    if (!n) return false;
    size_t keep = std::min<size_t>(n, 256);
    for (size_t i = 0; i < keep; i++) S->fcache.push_back(S->efreed[i]);
    push_chains(S->efreed, keep);
    S->c.help_evicts += n;
    return true;
}

void SlotStore::evictor_main() {
    std::vector<Victim> vic;
    vic.reserve(kBatch / kUnit);
    std::vector<uint32_t> freed;
    freed.reserve(kBatch / kUnit);
    uint8_t* buf = aligned_buf(kBatch);
    while (!ev_stop_.load(std::memory_order_relaxed)) {
        if (ev_pause_.load(std::memory_order_acquire)) {
            ev_paused_.fetch_add(1);
            while (ev_pause_.load(std::memory_order_acquire) && !ev_stop_.load())
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            ev_paused_.fetch_sub(1);
            continue;
        }
        uint64_t fr = free_slots();
        if (fr >= high_wm_) {
            std::unique_lock<std::mutex> g(ev_mu_);
            ev_sleeping_.fetch_add(1);
            // system_clock wait = pthread_cond_timedwait (steady_clock's clockwait is not seen by TSan).
            ev_cv_.wait_until(g, std::chrono::system_clock::now() + std::chrono::milliseconds(1));
            ev_sleeping_.fetch_sub(1);
            continue;
        }
        // Small batches: freed slots reach the pool quickly while workers wait on it.
        size_t want = std::min<size_t>((high_wm_ - fr) + 64, 8192);
        // Once reads have started, writes compete with them for the device: drop clean victims
        // first (a bounded sweep), and write dirty ones only if that sweep found none or the pool
        // is still below its low watermark.
        size_t got = 0;
        if (run_started_.load(std::memory_order_relaxed)) {
            got = evict_batch(nullptr, 0, want, vic, freed, false, 16 * (uint64_t)want + 2 * kHandChunk);
            if (got) push_chains(freed, 0);
            if (got >= want || (got && free_slots() >= low_wm_)) continue;
        }
        size_t more = want - got;
        if (!evict_batch(buf, kBatch, more, vic, freed, true, 5 * nslots_ + 2 * kHandChunk)) {
            if (!got) std::this_thread::yield();
            continue;
        }
        push_chains(freed, 0);
    }
    free(buf);
}

// ── write-behind cleaner
int64_t SlotStore::sample_slot(uint64_t key) {
    uint64_t h = mix64(key ^ 0xD6E8FEB86659FD93ULL);
    if ((h & kSampMask) || key == ~0ULL || !run_started_.load(std::memory_order_relaxed)) return -1;
    const uint64_t mask = samp_cap_ - 1;
    uint64_t i = (h >> 11) & mask;
    for (int n = 0; n < 64; n++, i = (i + 1) & mask) {
        uint64_t k = samp_key_[i].load(std::memory_order_relaxed);
        if (k == key + 1) return (int64_t)i;
        if (k == 0) {
            uint64_t z = 0;
            if (samp_key_[i].compare_exchange_strong(z, key + 1, std::memory_order_relaxed) || z == key + 1)
                return (int64_t)i;
        }
    }
    return -1;
}

void SlotStore::sample_touch(uint64_t key) {
    int64_t i = sample_slot(key);
    if (i >= 0 && !(samp_st_[i].load(std::memory_order_relaxed) & 1)) samp_st_[i].fetch_or(1, std::memory_order_relaxed);
}

int64_t SlotStore::sample_cold(uint64_t key) {
    int64_t i = sample_slot(key);
    if (i < 0) return -1;
    uint8_t old = samp_st_[i].fetch_or(3, std::memory_order_relaxed);
    st_samp_cold_.fetch_add(1, std::memory_order_relaxed);
    if (old & 2) st_dup_inflight_.fetch_add(1, std::memory_order_relaxed);
    else if (old & 1) st_reread_.fetch_add(1, std::memory_order_relaxed);
    return i;
}

// Cleaner 0 watches the cold-read counter: while reads flow, write-behind stays off the device.
void SlotStore::monitor_reads() {
    static thread_local uint64_t last_ios = 0;
    static thread_local int64_t last_check = 0;
    int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - last_check < 1000000) return;
    last_check = now;
    Counters t = sum_counters();
    uint64_t ios = t.ios;
    if (ios != last_ios || t.reaped != ios) {  // reads submitted since last look, or still in flight
        last_ios = ios;
        last_io_ns_.store(now, std::memory_order_relaxed);
        if (!reads_busy_.exchange(true, std::memory_order_relaxed) && std::getenv("KVSTORE_DEBUG"))
            std::fprintf(stderr, "kvstore: reads busy (cleaned=%llu)\n", (unsigned long long)st_cleaned_.load());
    } else if (now - last_io_ns_.load(std::memory_order_relaxed) > 20000000) {
        if (reads_busy_.exchange(false, std::memory_order_relaxed) && std::getenv("KVSTORE_DEBUG"))
            std::fprintf(stderr, "kvstore: reads idle (cleaned=%llu)\n", (unsigned long long)st_cleaned_.load());
    }
}

// Blocks the last stopping session until every slot dirtied since the last drain is written behind.
// Only the chunks marked dirty are swept, so the cost tracks the dirty work, not the arena: a few
// chunks are cleaned right here; a large backlog (the end of a load) is spread over the cleaners.
// Abandoned at once if a session starts, an async read is issued, or the store stops, and after a
// hard 300 s cap. Skipped when no write happened since the last complete drain.
void SlotStore::drain_dirty() {
    std::lock_guard<std::mutex> dg(drain_mu_);
    if (run_started_.load(std::memory_order_acquire) || ev_stop_.load() || ncleaners_ == 0) return;
    Counters t = sum_counters();
    const uint64_t w = t.ups_inplace + t.ups_push + t.slow_writes + t.rmw_hit + t.rmw_miss + t.deletes;
    if (w == drained_writes_) return;
    auto t0 = std::chrono::steady_clock::now();
    const uint64_t c0 = st_cleaned_.load();
    const uint64_t used = (std::min<uint64_t>(fresh_.load(), nslots_) + kHandChunk - 1) / kHandChunk;
    uint64_t n = 0;
    for (uint64_t i = 0; i < used; i++) {
        if (!dmark_[i].load(std::memory_order_relaxed)) continue;
        dmark_[i].store(0, std::memory_order_relaxed);  // cleared before the sweep: a later dirtying re-marks
        dlist_[n++].store((uint32_t)i, std::memory_order_relaxed);
    }
    bool ok = false;
    if (n <= 64) {
        // This thread sweeps outside any op guard, holding overflow-entry pointers (Victim.ep) across
        // the batch write: hold ckpt_mu_ so no quiescer (compact_ov unmapping the overflow tables, or
        // a Checkpoint publishing a tail over this unwritten batch) runs meanwhile. Lock order
        // drain_mu_ -> ckpt_mu_; ckpt_mu_ holders only try_lock drain_mu_ (maybe_gc).
        std::lock_guard<std::mutex> ck(ckpt_mu_);
        if (!drain_ctx_) {
            drain_ctx_ = new CleanCtx();
            drain_ctx_->buf = aligned_buf(kBatch);
        }
        for (uint64_t j = 0; j < n; j++) {
            uint64_t ch = dlist_[j].load(std::memory_order_relaxed);
            if (clean_chunk(*drain_ctx_, ch * kHandChunk, std::min<uint64_t>((ch + 1) * kHandChunk, nslots_)))
                dmark_[ch].store(1, std::memory_order_relaxed);
        }
        clean_flush(*drain_ctx_);
        ok = true;
    } else {
        const uint64_t g = ++dgen_ & ((1ULL << (64 - kDGenShift)) - 1);
        ddone_.store(g << kDGenShift, std::memory_order_release);
        dend_.store(n, std::memory_order_release);
        dcur_.store(g << kDGenShift, std::memory_order_release);
        for (;;) {
            if ((ddone_.load(std::memory_order_acquire) & kDPosMask) >= n) {
                ok = true;
                break;
            }
            if (run_started_.load(std::memory_order_acquire) || ev_stop_.load() ||
                std::chrono::steady_clock::now() - t0 > std::chrono::seconds(300))
                break;
            {
                std::lock_guard<std::mutex> g2(sess_mu_);
                if (!by_thread_.empty()) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        dend_.store(0, std::memory_order_release);  // stop claims (abandon, or nothing left)
        if (!ok)  // re-mark what may not have been swept
            for (uint64_t j = 0; j < n; j++) dmark_[dlist_[j].load(std::memory_order_relaxed)].store(1, std::memory_order_relaxed);
    }
    if (ok) drained_writes_ = w;
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (secs > 0.05 || std::getenv("KVSTORE_DEBUG"))
        std::fprintf(stderr, "kvstore: pre-run drain %s: %llu of %llu chunks dirty, %llu slots written behind, %.2f s\n",
                     ok ? "done" : "abandoned", (unsigned long long)n, (unsigned long long)used,
                     (unsigned long long)(st_cleaned_.load() - c0), secs);
}

// Write the encoded batch and turn each still-unchanged slot clean: its entry takes the new record
// as D (if the record fits an entry; else the slot simply stays dirty).
void SlotStore::clean_flush(CleanCtx& C) {
    if (C.vic.empty()) return;
    uint64_t off = write_victims(C.buf, C.pos, batch_live(C.buf, C.vic.data(), C.vic.size()));
    commit_stall();
    uint64_t ok = 0, dead = 0;
    for (const Victim& x : C.vic) {
        Slot* s = slot(x.si);
        uint32_t v = lock_slot(s);
        uint64_t rp = mk_ptr(off + x.boff, x.bytes);
        // Same rule as the evictor's commit: unmodified since encode (version) and same content.
        const bool good = wb_unmodified(s) && ptr_fits(rp) && x.ep->load(std::memory_order_acquire) == x.e &&
                          slot_matches(s, C.buf + x.boff, C.tmp);
        wb_end(s);
        if (good) {
            x.ep->store(mk_res(x.chain ? kMD : kMDSolo, x.si, rp & kAddrMask, res_tag5(x.e)), std::memory_order_release);
            set_ctl(s, kDirty, 0);
            unlock_mod(s, v);
            ok++;
        } else {
            unlock_nomod(s, v);
            dead += live_charge(C.buf + x.boff, x.bytes);  // the slot stays dirty: rewritten later
        }
    }
    if (dead) dead_bytes_.fetch_add(dead, std::memory_order_relaxed);
    st_cleaned_.fetch_add(ok, std::memory_order_relaxed);
    if (ok < C.vic.size()) st_clean_abort_.fetch_add(C.vic.size() - ok, std::memory_order_relaxed);
    std::memset(C.buf, 0, C.pos);
    C.pos = 0;
    C.vic.clear();
}

// Encodes every dirty live head of [start, end) into C.buf (flushing when full). True if a dirty
// slot was skipped because it was locked.
bool SlotStore::clean_chunk(CleanCtx& C, uint64_t start, uint64_t end) {
    bool skipped = false;
    for (uint64_t si = start; si < end; si++) {
        Slot* s = slot(si);
        uint32_t w = ctl(s);
        if ((w & (kLive | kDirty)) != (kLive | kDirty)) continue;
        uint32_t v;
        if (!try_lock_slot(s, v)) {
            skipped = true;
            continue;
        }
        w = ctl(s);
        if ((w & (kLive | kDirty)) != (kLive | kDirty)) {
            unlock_nomod(s, v);
            continue;
        }
        bool tomb = is_tomb(w);
        uint32_t sz = tomb ? 0 : value_size(s, w);
        uint32_t bytes = kHdr + sz;
        uint64_t at = place(C.pos, bytes);
        if (at + round_up(bytes, kUnit) > kBatch) {
            unlock_nomod(s, v);
            clean_flush(C);
            si--;  // retry this slot into the empty buffer
            continue;
        }
        uint64_t key = slot_key(s);
        std::atomic<uint64_t>* ep = find_own_entry(key, (uint32_t)si, nullptr);
        if (!ep) {
            unlock_nomod(s, v);
            continue;
        }
        if (!wb_begin(s)) {  // another write-behind of this slot is in flight: revisit the chunk
            unlock_nomod(s, v);
            skipped = true;
            continue;
        }
        uint64_t e = ep->load(std::memory_order_acquire);
        if (!tomb) copy_value(s, sz, C.tmp);
        uint64_t chain = chain_start(e);
        enc_record(C.buf + at, key, next_seq(), tomb, chain, C.tmp, sz);
        unlock_nomod(s, v);
        C.vic.push_back({(uint32_t)si, (uint32_t)at, bytes, ep, e, chain, 0, v & kVerMask});
        C.pos = at + round_up(bytes, kUnit);
    }
    return skipped;
}

// Writes dirty resident records behind, in CLOCK order just ahead of the CLOCK hand (never more than
// one revolution ahead, so each slot is written at most once per revolution). A cleaned slot's entry
// keeps its new record address (D mode) and the slot is later evicted with no write.
void SlotStore::cleaner_main(int idx) {
    CleanCtx C;
    C.vic.reserve(kBatch / kUnit);
    C.buf = aligned_buf(kBatch);
    while (!ev_stop_.load(std::memory_order_relaxed)) {
        if (ev_pause_.load(std::memory_order_acquire)) {
            ev_paused_.fetch_add(1);
            while (ev_pause_.load(std::memory_order_acquire) && !ev_stop_.load())
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            ev_paused_.fetch_sub(1);
            continue;
        }
        if (idx == 0) monitor_reads();
        // Pre-run drain: claim entries of dlist_[0, dend_) until the sweep is done (ignores
        // reads_busy_ and the CLOCK-relative bound; no reads have been issued and no session is active).
        if ((dcur_.load(std::memory_order_acquire) & kDPosMask) < dend_.load(std::memory_order_acquire)) {
            uint64_t gen = ~0ULL, n = 0;
            // A chunk counts as done only once its records are written (after flush).
            auto credit = [&] {
                clean_flush(C);
                uint64_t d = ddone_.load(std::memory_order_acquire);
                while (n && (d >> kDGenShift) == gen &&
                       !ddone_.compare_exchange_weak(d, d + n, std::memory_order_acq_rel)) {
                }
                n = 0;
            };
            while (!ev_stop_.load(std::memory_order_relaxed) && !ev_pause_.load(std::memory_order_relaxed)) {
                uint64_t v = dcur_.fetch_add(1, std::memory_order_acq_rel);
                uint64_t g = v >> kDGenShift, j = v & kDPosMask;
                uint64_t de = dend_.load(std::memory_order_acquire);
                if (g != gen) {  // a new drain began: credit the old one first
                    credit();
                    gen = g;
                }
                if (j >= de) break;
                uint64_t ch = dlist_[j].load(std::memory_order_relaxed);
                if (ch < nchunks_ &&
                    clean_chunk(C, ch * kHandChunk, std::min<uint64_t>((ch + 1) * kHandChunk, nslots_)))
                    dmark_[ch].store(1, std::memory_order_relaxed);
                n++;
            }
            credit();
            continue;
        }
        if (reads_busy_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        // Bound: [hand, hand + one revolution - gap), and never past the slots handed out so far.
        const uint64_t gap = std::min<uint64_t>(nslots_ / 4, 2 * high_wm_ + kHandChunk);
        uint64_t h = hand_.load(std::memory_order_relaxed);
        uint64_t c = chand_.load(std::memory_order_relaxed);
        if (c < h) {
            chand_.compare_exchange_strong(c, h, std::memory_order_relaxed);
            continue;
        }
        uint64_t lim = h + nslots_ - gap;
        uint64_t f = fresh_.load(std::memory_order_relaxed);
        if (f < nslots_) lim = std::min(lim, f);
        if (c + kHandChunk > lim) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        for (int nchunk = 0; nchunk < 64; nchunk++) {
            if (ev_stop_.load(std::memory_order_relaxed) || ev_pause_.load(std::memory_order_relaxed) ||
                reads_busy_.load(std::memory_order_relaxed))
                break;
            uint64_t c0 = chand_.fetch_add(kHandChunk, std::memory_order_relaxed);
            uint64_t start = c0 % nslots_;
            uint64_t end = std::min<uint64_t>(start + kHandChunk, nslots_);
            clean_chunk(C, start, end);
            if (chand_.load(std::memory_order_relaxed) + kHandChunk > lim) break;
        }
        clean_flush(C);
    }
    free(C.buf);
}

// ───────────────────────────── checkpoint / recovery ─────────────────────────────
// Version 3: the V2 header (version 3), then the digest trailer: n, n x ManifestSeg (one per live
// segment, ascending), crc32c of the trailer. The caller is quiesced (or alone at open), so every
// record below the tail is written and counted.
void SlotStore::write_manifest() {
    std::vector<uint8_t> blob(sizeof(ManifestV2));
    ManifestV2& m = *reinterpret_cast<ManifestV2*>(blob.data());
    std::memset(&m, 0, sizeof m);
    m.magic = kManifestMagic;
    m.version = 3;
    std::vector<ManifestSeg> ents;
    {
        std::lock_guard<std::mutex> g(alloc_mu_);
        uint64_t t = tail_.load();
        m.active_seg = t >> kSegShift;
        m.active_off = t & (kSegSize - 1);
        std::memcpy(m.live, seg_used_, sizeof m.live);
        for (uint32_t w = 0; w < kSegWords; w++)
            for (uint64_t b = seg_used_[w]; b; b &= b - 1) {
                uint32_t sg = w * 64 + (uint32_t)__builtin_ctzll(b);
                ents.push_back({sg, seg_cnt_[sg].load(std::memory_order_acquire), seg_dig_[sg].load(std::memory_order_acquire)});
            }
    }
    m.next_seq = seq_.load();
    m.nb = nb_;
    m.align = align_;
    m.crc = crc32c(0xFFFFFFFFu, (const uint8_t*)&m, offsetof(ManifestV2, crc));
    const uint64_t n = ents.size();
    const size_t t0 = blob.size();
    blob.resize(t0 + 8 + n * sizeof(ManifestSeg) + 4);
    std::memcpy(blob.data() + t0, &n, 8);
    if (n) std::memcpy(blob.data() + t0 + 8, ents.data(), n * sizeof(ManifestSeg));
    const uint32_t tc = crc32c(0xFFFFFFFFu, blob.data() + t0, 8 + n * sizeof(ManifestSeg));
    std::memcpy(blob.data() + blob.size() - 4, &tc, 4);
    ::unlinkat(dirfd_, "MANIFEST.tmp", 0);  // never write through a planted name
    int fd = ::openat(dirfd_, "MANIFEST.tmp", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0) die("open MANIFEST.tmp");
    for (size_t done = 0; done < blob.size();) {
        ssize_t r = ::write(fd, blob.data() + done, blob.size() - done);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) die("write MANIFEST.tmp");
        done += (size_t)r;
    }
    if (::fsync(fd) != 0) die("fsync MANIFEST.tmp");
    ::close(fd);
    if (::renameat(dirfd_, "MANIFEST.tmp", dirfd_, "MANIFEST") != 0) die("rename MANIFEST");
    if (::fsync(dirfd_) != 0) die("fsync dir");
}

// Stop the world (caller holds ckpt_mu_ and is not inside an op): no session op in flight, the
// background threads paused at their loop tops (their victim batches flushed).
void SlotStore::quiesce_all() {
    ckpt_flag_.store(1, std::memory_order_seq_cst);
    {
        std::lock_guard<std::mutex> g(sess_mu_);
        for (Session* s : sessions_)
            while (s->in_op.load(std::memory_order_seq_cst)) std::this_thread::yield();
    }
    int ne = (int)evth_.size();
    ev_pause_.store(true, std::memory_order_release);
    kick_evictor();
    {
        std::lock_guard<std::mutex> g(ev_mu_);
        ev_cv_.notify_all();
    }
    while (ev_paused_.load() < ne) std::this_thread::sleep_for(std::chrono::microseconds(100));
}

void SlotStore::resume_all() {
    ev_pause_.store(false, std::memory_order_release);
    ckpt_flag_.store(0, std::memory_order_seq_cst);
}

// Rebuild the overflow index from its live (non-zero) entries, with the store quiesced: nothing
// holds an overflow entry pointer across this except pending async reads, whose (optional) cache
// install is dropped. Dead classes' slots vanish; the overflow bits are recomputed.
void SlotStore::compact_ov() {
    std::unique_lock<std::mutex> ck(ckpt_mu_, std::try_to_lock);
    if (!ck.owns_lock()) return;
    uint64_t d = ov_dead_.load(std::memory_order_relaxed);
    if (d < kOvCompactMin || 2 * d < ov_total_.load(std::memory_order_relaxed)) return;  // another thread did it
    quiesce_all();
    std::lock_guard<std::mutex> g(ov_mu_);
    const int n = nov_.load(std::memory_order_relaxed);
    uint64_t live = 0;
    for (int k = 0; k < n; k++)
        for (uint64_t i = 0; i <= ov_[k].mask; i++)
            live += ov_[k].cls[i].load(std::memory_order_relaxed) != 0 && ov_[k].ent[i].load(std::memory_order_relaxed) != 0;
    uint64_t cap = ovcap0_;
    while (cap < 2 * live + kOvCompactMin) cap <<= 1;
    OvTable nt;
    nt.cls = (std::atomic<uint64_t>*)map_anon(cap * 8, true);
    nt.ent = (std::atomic<uint64_t>*)map_anon(cap * 8, true);
    nt.mask = cap - 1;
    nt.count = 0;
    for (uint64_t w = 0; w < (nb_ + 63) / 64; w++) ovbits_[w].store(0, std::memory_order_relaxed);
    for (int k = 0; k < n; k++) {
        const OvTable& t = ov_[k];
        for (uint64_t i = 0; i <= t.mask; i++) {
            uint64_t c = t.cls[i].load(std::memory_order_relaxed), e = t.ent[i].load(std::memory_order_relaxed);
            if (!c || !e) continue;
            uint64_t id = c - 1, j = mix64(id) & nt.mask;
            while (nt.cls[j].load(std::memory_order_relaxed) != 0) j = (j + 1) & nt.mask;
            nt.ent[j].store(e, std::memory_order_relaxed);
            nt.cls[j].store(c, std::memory_order_relaxed);
            nt.count++;
            uint64_t b1 = id >> 21;
            ovbits_[b1 >> 6].fetch_or(1ULL << (b1 & 63), std::memory_order_relaxed);
        }
    }
    {
        std::lock_guard<std::mutex> sg(sess_mu_);
        for (Session* s : sessions_)
            for (Pend& pd : s->pend)
                for (int k = 0; k < n; k++)
                    if (pd.ep >= ov_[k].ent && pd.ep <= ov_[k].ent + ov_[k].mask) {
                        pd.ep = nullptr;
                        pd.head_e = 0;
                        break;
                    }
    }
    for (int k = 0; k < n; k++) {
        munmap(ov_[k].cls, (ov_[k].mask + 1) * 8);
        munmap(ov_[k].ent, (ov_[k].mask + 1) * 8);
        ov_[k] = OvTable();
    }
    ov_[0] = nt;
    nov_.store(1, std::memory_order_release);
    ov_total_.store(nt.count, std::memory_order_relaxed);
    ov_dead_.store(0, std::memory_order_relaxed);
    st_ov_compact_.fetch_add(1, std::memory_order_relaxed);
    resume_all();
}

void SlotStore::Checkpoint() {
    std::lock_guard<std::mutex> ck(ckpt_mu_);
    auto t0 = std::chrono::steady_clock::now();
    quiesce_all();

    if (run_ready_.load(std::memory_order_acquire) && !run_printed_.exchange(true)) print_run_stats("before checkpoint");

    // Append every dirty resident head; each then remembers its record in its entry (clean, D mode)
    // when the record fits an entry. Sessions are quiesced and the background threads paused.
    uint8_t* buf = aligned_buf(kBatch);
    alignas(8) uint8_t tmp[GenValue::kMaxSize + 8];
    uint64_t pos = 0, nrec = 0;
    std::vector<Victim> done;
    auto flush = [&] {
        if (!pos) return;
        uint64_t off = write_victims(buf, pos, batch_live(buf, done.data(), done.size()));
        uint64_t dead = 0;
        for (const Victim& x : done) {
            Slot* s = slot(x.si);
            uint32_t v = lock_slot(s);
            uint64_t rp = mk_ptr(off + x.boff, x.bytes);
            if (ptr_fits(rp) && x.ep->load(std::memory_order_acquire) == x.e) {
                x.ep->store(mk_res(x.chain ? kMD : kMDSolo, x.si, rp & kAddrMask, res_tag5(x.e)),
                            std::memory_order_release);
                set_ctl(s, kDirty, 0);
                unlock_mod(s, v);
            } else {
                unlock_nomod(s, v);
                dead += live_charge(buf + x.boff, x.bytes);  // the slot stays dirty: rewritten later
            }
        }
        if (dead) dead_bytes_.fetch_add(dead, std::memory_order_relaxed);
        done.clear();
        std::memset(buf, 0, kBatch);
        pos = 0;
    };
    uint64_t lim = std::min<uint64_t>(fresh_.load(), nslots_);
    for (uint64_t i = 0; i < lim; i++) {
        Slot* s = slot(i);
        uint32_t w = ctl(s);
        if ((w & (kLive | kDirty)) != (kLive | kDirty)) continue;
        uint32_t v = lock_slot(s);
        w = ctl(s);
        std::atomic<uint64_t>* ep = nullptr;
        if ((w & (kLive | kDirty)) == (kLive | kDirty) && (ep = find_own_entry(slot_key(s), (uint32_t)i, nullptr))) {
            uint64_t e = ep->load(std::memory_order_acquire);
            bool tomb = is_tomb(w);
            uint32_t sz = tomb ? 0 : value_size(s, w);
            uint32_t bytes = kHdr + sz;
            uint64_t at = place(pos, bytes);
            if (at + round_up(bytes, kUnit) > kBatch) {
                // flush() locks the batch's slots, which may share this slot's stripe: release it first
                // and redo this slot into the empty buffer (sessions are quiesced).
                unlock_nomod(s, v);
                flush();
                i--;
                continue;
            }
            if (!tomb) copy_value(s, sz, tmp);
            uint64_t chain = chain_start(e);
            enc_record(buf + at, slot_key(s), next_seq(), tomb, chain, tmp, sz);
            done.push_back({(uint32_t)i, (uint32_t)at, bytes, ep, e, chain, 0, v & kVerMask});
            pos = at + round_up(bytes, kUnit);
            nrec++;
        }
        unlock_nomod(s, v);
    }
    flush();
    free(buf);
    st_ckpt_recs_.fetch_add(nrec);

    sync_segments();
    write_manifest();
    uint64_t gc_segs = maybe_gc();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "kvstore: checkpoint %llu dirty records, log %llu MiB (live est %llu MiB), gc %llu segments, %.2f s\n",
                 (unsigned long long)nrec, (unsigned long long)(log_bytes_.load() >> 20),
                 (unsigned long long)(live_estimate() >> 20), (unsigned long long)gc_segs, secs);
    print_stats("checkpoint");
    resume_all();
}

// fsync every dirty segment in use, then the directory.
void SlotStore::sync_segments() {
    uint64_t bits[kSegWords];
    {
        std::lock_guard<std::mutex> g(alloc_mu_);
        std::memcpy(bits, seg_used_, sizeof bits);
    }
    for (uint32_t w = 0; w < kSegWords; w++) {
        for (uint64_t b = bits[w]; b; b &= b - 1) {
            uint32_t sg = w * 64 + (uint32_t)__builtin_ctzll(b);
            if (segdirty_[sg].load(std::memory_order_relaxed)) {
                int fd = fds_[sg].load();
                if (fd >= 0 && ::fsync(fd) != 0) die("fsync segment", (long long)sg);
                segdirty_[sg].store(0, std::memory_order_relaxed);
            }
        }
    }
    if (::fsync(dirfd_) != 0) die("fsync dir");
}

// Parse the records of segment sg in [seg_start(sg), seg_end): f(record, address, bytes) -> false
// stops the scan. Inside the checkpointed range only zero padding may fail to parse (records are
// placed by sector and every gap is written as zeros); anything else is a damaged record, and
// serving an older version or resurrecting a deleted key instead is not allowed: fail loudly.
// A scan that runs to the end re-derives the segment's record count and digest: with sr.chk they
// must equal the checkpoint's (a record lost inside the range, e.g. zeroed, is damage); either way
// they become the segment's running totals (recovery adopts them, appends then add to them).
template <class F>
void SlotStore::scan_seg(const SegRange& sr, uint8_t* buf, size_t chunk, F&& f) {
    const uint32_t sg = sr.seg;
    const uint64_t seg_end = sr.end;
    const uint64_t seg_lo = (uint64_t)sg << kSegShift;
    uint64_t cnt = 0, dig = 0;
    const int fd = fds_[sg].load();
    uint64_t base = 0, resume = seg_start(sg);
    while (base < seg_end) {
        uint64_t want = std::min<uint64_t>(chunk, round_up(seg_end - base, align_));
        ssize_t got = ::pread(fd, buf, want, (off_t)base);
        if (got < 0) die("log scan read", (long long)sg);
        if ((uint64_t)got < std::min<uint64_t>(want, seg_end - base)) {
            std::string w = "short read inside the checkpointed log, segment " + std::to_string((unsigned long long)sg) +
                            " offset " + std::to_string((unsigned long long)base) + " (truncated/damaged)";
            die_state(dir_, w.c_str());
        }
        uint64_t avail = std::min<uint64_t>((uint64_t)got, seg_end - base);
        uint64_t u = resume;
        resume = 0;
        uint64_t next_base = base + avail;
        while (u + kHdr <= avail) {
            RecView r;
            if (!parse_record(buf + u, avail - u, r)) {
                // A large record cut by the chunk end: re-read from here.
                uint64_t sf;
                std::memcpy(&sf, buf + u + 8, 8);
                uint16_t sz;
                std::memcpy(&sz, buf + u + 20, 2);
                if (((sf >> 56) & 0xFE) == 0xA0 && sz <= GenValue::kMaxSize && u + kHdr + sz > avail &&
                    base + avail < seg_end && (u % kSector) == 0 && u > 0) {
                    next_base = round_dn(base + u, align_);
                    resume = base + u - next_base;
                    break;
                }
                const uint8_t* q = buf + u;
                uint64_t n = std::min<uint64_t>(kUnit, avail - u);
                bool zero = true;
                for (uint64_t j = 0; j < n && zero; j++) zero = q[j] == 0;
                if (!zero) {
                    std::string w = "corrupt record (bad checksum or format) inside the checkpointed log, segment " +
                                    std::to_string((unsigned long long)sg) + " offset " +
                                    std::to_string((unsigned long long)(base + u));
                    die_state(dir_, w.c_str());
                }
                u += kUnit;
                continue;
            }
            uint32_t bytes = kHdr + r.size;
            uint64_t off = seg_lo + base + u;
            uint32_t stored;
            std::memcpy(&stored, buf + u + 16, 4);
            cnt++;
            dig += rec_dig(stored, base + u);
            u += round_up(bytes, kUnit);
            if (!f(r, off, bytes)) return;
        }
        base = next_base;
    }
    if (sr.chk && (cnt != sr.cnt || dig != sr.dig)) {
        std::string w = "checkpointed log segment " + std::to_string((unsigned long long)sg) + " holds " +
                        std::to_string((unsigned long long)cnt) + " records but the checkpoint recorded " +
                        std::to_string((unsigned long long)sr.cnt) +
                        (cnt == sr.cnt ? " with a different digest" : "") +
                        " (a record is missing, zeroed or damaged)";
        die_state(dir_, w.c_str());
    }
    seg_cnt_[sg].store(cnt, std::memory_order_relaxed);
    seg_dig_[sg].store(dig, std::memory_order_relaxed);
}

// Open-time recovery from a manifest: the checkpointed log is `segs` (the active one read up to
// active_off). Each segment must exist and hold its checkpointed bytes: a passed segment was
// extended to kSegSize when the log moved past it (log_alloc), the active one holds at least
// active_off. Anything short or missing is damage: fail before changing a file. Then segment files
// outside the set (post-checkpoint segments, or ones a GC freed) are removed and bytes past
// active_off (post-checkpoint writes, possibly torn) are dropped.
void SlotStore::recover(const std::vector<SegRange>& segs, uint64_t act, uint64_t act_off, uint64_t next_seq) {
    std::vector<uint8_t> inset(kMaxSeg, 0);
    for (const SegRange& r : segs) {
        inset[r.seg] = 1;
        char name[64];
        std::snprintf(name, sizeof name, "seg_%06u.log", (unsigned)r.seg);
        struct stat st;
        if (::fstatat(dirfd_, name, &st, 0) != 0) {
            if (r.seg == act && r.end <= seg_start(r.seg)) continue;  // nothing checkpointed in it yet
            std::string w = std::string("checkpointed log segment ") + name + " is missing (damaged)";
            die_state(dir_, w.c_str());
        }
        if ((uint64_t)st.st_size < r.end) {
            std::string w = std::string("checkpointed log segment ") + name + " is shorter than the checkpoint (" +
                            std::to_string((unsigned long long)st.st_size) + " < " +
                            std::to_string((unsigned long long)r.end) + " bytes; truncated/damaged)";
            die_state(dir_, w.c_str());
        }
    }
    if (DIR* d = ::opendir(dir_.c_str())) {
        while (dirent* de = ::readdir(d)) {
            unsigned sidx;
            if (parse_seg_name(de->d_name, &sidx) && (sidx >= kMaxSeg || !inset[sidx])) ::unlinkat(dirfd_, de->d_name, 0);
        }
        ::closedir(d);
    }
    for (const SegRange& r : segs) open_seg(r.seg, r.seg == act);
    {
        int fd = fds_[act].load();
        struct stat st;
        if (::fstat(fd, &st) == 0 && (uint64_t)st.st_size != act_off) {
            if (::ftruncate(fd, (off_t)act_off) != 0) die("ftruncate segment");
            ::fsync(fd);
        }
    }
    ::fsync(dirfd_);
    uint64_t lb = 0, units = 0;
    {
        std::lock_guard<std::mutex> g(alloc_mu_);
        for (const SegRange& r : segs) {
            seg_used_[r.seg / 64] |= 1ULL << (r.seg % 64);
            struct stat st;
            if (::fstat(fds_[r.seg].load(), &st) == 0) lb += (uint64_t)st.st_blocks * 512;
            units += r.end / kUnit;
        }
    }
    tail_.store((act << kSegShift) + act_off);
    log_bytes_.store(lb);
    uint64_t live = rebuild_index(segs, units + units / 4 + 16, next_seq);
    rebase_live(live + (log_bytes_.load() - lb));  // + any chain records the rebuild wrote
}

// Build the index from a log scan: the latest record per key (highest seq) is current. The map
// lives in scratch carved from the slot arena, which is budgeted and empty here (open, or GC after
// reset_mem), so this maps no memory that grows with the log. The map is the largest power of two
// that fits the arena; when the log has more keys than it holds (`need` plans for it), the key
// space is split into hash ranges (a hash independent of the map slot) and the log is re-read once
// per range. A range whose map fills anyway is split in two and re-read (nothing from it reached
// the index yet). The arena is released (zero pages) before the slot code runs. Returns the live
// keys' record bytes (unit-rounded).
uint64_t SlotStore::rebuild_index(const std::vector<SegRange>& segs, uint64_t need, uint64_t next_seq) {
    auto t0 = std::chrono::steady_clock::now();
    struct RE {
        uint64_t key, seq, ptr;  // seq: bit63 occupied, bit62 tomb; ptr: record pointer | units << 48
    };
    const uint64_t scratch = nslots_ * sizeof(Slot);
    RE* const map = reinterpret_cast<RE*>(arena_);
    uint64_t capmax = 1024;
    while (capmax * 2 * sizeof(RE) <= scratch) capmax <<= 1;
    uint64_t npass = (need + capmax - 1) / capmax;
    if (npass == 0) npass = 1;
    uint64_t cap1 = 1024;
    while (cap1 < need && cap1 < capmax) cap1 <<= 1;
    auto part_hash = [](uint64_t key) { return mix64(key ^ 0x7A3C5E1F94D2B680ULL); };
    std::vector<std::pair<uint64_t, uint64_t>> ranges;  // inclusive [lo, hi] of part_hash
    for (uint64_t p = npass; p-- > 0;) {
        unsigned __int128 lo = ((unsigned __int128)p << 64) / npass;
        unsigned __int128 hi = (((unsigned __int128)(p + 1) << 64) / npass) - 1;
        ranges.push_back({(uint64_t)lo, (uint64_t)hi});
    }
    uint64_t maxseq = 0, nrec = 0, nkeys = 0, passes = 0, splits = 0, prevcap = 0, logb = 0;
    const size_t kChunk = 4u << 20;
    uint8_t* buf = aligned_buf(kChunk);
    std::vector<std::pair<std::atomic<uint64_t>*, uint64_t>> extra;
    uint64_t live = 0, live_bytes = 0;
    while (!ranges.empty()) {
        const uint64_t rlo = ranges.back().first, rhi = ranges.back().second;
        ranges.pop_back();
        const bool all = rlo == 0 && rhi == ~0ULL;
        const uint64_t cap = all ? cap1 : capmax;
        const uint64_t mmask = cap - 1, full = cap - cap / 16;
        if (passes) std::memset((void*)map, 0, prevcap * sizeof(RE));
        prevcap = cap;
        passes++;
        uint64_t pkeys = 0, prec = 0;
        bool overflow = false;
        for (const SegRange& sr : segs) {
            if (overflow) break;
            logb += sr.end;
            scan_seg(sr, buf, kChunk, [&](const RecView& r, uint64_t off, uint32_t bytes) {
                prec++;
                if (r.seq > maxseq) maxseq = r.seq;
                if (!all) {
                    uint64_t ph = part_hash(r.key);
                    if (ph < rlo || ph > rhi) return true;
                }
                const uint64_t ptr = mk_ptr(off, bytes) | ((rec_charge(bytes) / kUnit) << 48);
                uint64_t i = mix64(r.key) & mmask;
                for (;;) {
                    RE& x = map[i];
                    if (!(x.seq >> 63)) {
                        if (pkeys >= full) {
                            overflow = true;
                            return false;
                        }
                        x.key = r.key;
                        x.seq = (1ULL << 63) | (r.tomb ? (1ULL << 62) : 0) | r.seq;
                        x.ptr = ptr;
                        pkeys++;
                        return true;
                    }
                    if (x.key == r.key) {
                        if (r.seq > (x.seq & ((1ULL << 62) - 1))) {
                            x.seq = (1ULL << 63) | (r.tomb ? (1ULL << 62) : 0) | r.seq;
                            x.ptr = ptr;
                        }
                        return true;
                    }
                    i = (i + 1) & mmask;
                }
            });
        }
        if (overflow) {
            // This range has more keys than the map holds: split it and re-read each half.
            if (rlo == rhi) die("recovery map overflow on a single hash value", (long long)pkeys);
            uint64_t mid = rlo + (rhi - rlo) / 2;
            ranges.push_back({mid + 1, rhi});
            ranges.push_back({rlo, mid});
            splits++;
            continue;
        }
        nrec = prec;
        nkeys += pkeys;
        // Build this range's part of the index: one head per class; a class with several live keys
        // gets a fresh chain (after the last range, below).
        for (uint64_t i = 0; i < cap; i++) {
            const RE& x = map[i];
            if (!(x.seq >> 63) || (x.seq >> 62 & 1)) continue;
            live++;
            live_bytes += (x.ptr >> 48) * kUnit;
            const uint64_t ptr = x.ptr & kPtrMask;
            Cls c = classify(x.key);
            std::atomic<uint64_t>* ep = find_entry(c);
            if (!ep) {
                ep = create_entry(c);
                uint64_t e = ep->load();
                ep->store((e & kHdrBits) | ptr | kPtrT);
            } else {
                extra.push_back({ep, ptr});
            }
        }
    }
    seq_.store(std::max<uint64_t>(std::max<uint64_t>(next_seq, seq_.load()), maxseq + 1));
    // Release the scratch: the slot code sees the zeroed arena of a fresh open.
    if (::madvise(arena_, scratch, MADV_DONTNEED) != 0) die("madvise arena");
    if (!extra.empty()) {
        Session tmp;
        tmp.sbuf = aligned_buf(16384);
        tmp.wbuf = aligned_buf(8192);
        std::sort(extra.begin(), extra.end());
        size_t i = 0;
        while (i < extra.size()) {
            std::atomic<uint64_t>* ep = extra[i].first;
            std::vector<uint64_t> members;
            uint64_t e = ep->load();
            members.push_back((e & kPay) & ~kPtrT);
            while (i < extra.size() && extra[i].first == ep) members.push_back(extra[i++].second);
            uint64_t prevp = 0;
            for (uint64_t p : members) {
                RecView r;
                read_record_sync(tmp.sbuf, p, r);
                if (std::getenv("KVSTORE_DEBUG"))
                    std::fprintf(stderr, "kvstore: chained class member key %llu (ptr %llx)\n", (unsigned long long)r.key,
                                 (unsigned long long)p);
                std::vector<uint8_t> v(r.val, r.val + r.size);
                prevp = write_record_sync(&tmp, r.key, false, prevp, v.data(), r.size);
            }
            ep->store((e & kHdrBits) | prevp);
        }
        free(tmp.sbuf);
        free(tmp.wbuf);
    }
    free(buf);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "kvstore: recovered %llu live keys (%llu distinct) from %llu records (%llu MiB log, %zu chained) in %.2f s; "
                 "map in arena scratch %llu MiB, %llu range(s) planned, %llu pass(es), %llu split(s)\n",
                 (unsigned long long)live, (unsigned long long)nkeys, (unsigned long long)nrec,
                 (unsigned long long)((passes ? logb / passes : 0) >> 20), extra.size(), secs,
                 (unsigned long long)(scratch >> 20), (unsigned long long)npass, (unsigned long long)passes,
                 (unsigned long long)splits);
    return live_bytes;
}

uint64_t SlotStore::live_estimate() const {
    uint64_t sub = (sum_counters().live_sub - sub_base_) + (dead_bytes_.load() - dead_base_);
    uint64_t v = live_base_ + (log_bytes_.load() - log_base_);
    return v > sub ? v - sub : 0;
}

// Live log bytes of the class heads (quiesced, right after a checkpoint wrote every dirty head): each
// disk head's record by its size code (tombstones 0), each clean resident non-tombstone head's record D
// by its slot's value size. A dirty resident head counts 0, as in the estimate (its checkpointed copy
// was counted dead: the slot is rewritten later). A lower bound on the live bytes: non-head members
// of multi-key classes are left out, and every head's charge is exact (mk_ptr codes every pointer that
// becomes an entry head). One pass over the index, split across threads.
uint64_t SlotStore::live_heads() const {
    auto entry_live = [this](uint64_t e) -> uint64_t {
        if (!e) return 0;
        if (e & kIsSlot) {
            const Slot* s = slot(res_slot(e));
            const uint32_t w = ctl(s);
            if (!(w & kLive) || is_tomb(w) || !is_clean(w)) return 0;
            return rec_charge(kHdr + value_size(s, w));
        }
        const uint64_t p = e & kPay;
        return p ? ptr_live_charge(p) : 0;
    };
    const uint64_t n = nb_ * 8;
    unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    if (n < (1u << 22)) nt = 1;
    std::vector<uint64_t> part(nt, 0);
    auto run = [&](unsigned k) {
        const uint64_t lo = n * k / nt, hi = n * (k + 1) / nt;
        uint64_t sum = 0;
        for (uint64_t i = lo; i < hi; i++) sum += entry_live(bk_[i].load(std::memory_order_relaxed));
        part[k] = sum;
    };
    std::vector<std::thread> th;
    for (unsigned k = 1; k < nt; k++) th.emplace_back(run, k);
    run(0);
    for (auto& t : th) t.join();
    uint64_t live = 0;
    for (uint64_t v : part) live += v;
    for (int k = 0; k < nov_.load(); k++)
        for (uint64_t i = 0; i <= ov_[k].mask; i++) live += entry_live(ov_[k].ent[i].load(std::memory_order_relaxed));
    return live;
}

// Log garbage collection, at Checkpoint (quiesced, the checkpoint just published) when the
// allocated log exceeds 1.5 x the live bytes + 128 MiB. Never during operations: the scored
// run writes nothing and its checkpoint stays far below the trigger. Returns segments freed.
// The live bytes come from the estimate E (live_estimate). Every stale charge it subtracts is exact
// or an over-charge, except a blind Upsert's push over a disk head in a multi-key class (the key's
// old record may sit deeper in the chain), which under-charges by at most kMaxRecCharge. So with n
// blind pushes since the last exact base, true live L >= E - n * kMaxRecCharge. When that lower bound
// already clears the trigger, no GC is due; when E itself trips it, GC; in between the error could
// decide it, so the heads are summed exactly (live_heads <= L) and the estimate re-based on it.
// Either way a checkpoint that does not collect leaves allocated <= 1.5 x L + 128 MiB.
uint64_t SlotStore::maybe_gc() {
    static const bool force = std::getenv("KVSTORE_TEST_GC_ALWAYS") != nullptr;  // test seam
    const uint64_t lb = log_bytes_.load();
    auto trips = [lb](uint64_t live) { return lb > live + live / 2 + (128ULL << 20); };
    uint64_t live = live_estimate();
    if (!force && trips(live)) {
        // fall through to the GC
    } else if (!force) {
        const uint64_t n = sum_counters().ups_push - push_base_;
        const uint64_t err = n > (~0ULL / kMaxRecCharge) ? ~0ULL : n * kMaxRecCharge;
        if (!trips(live > err ? live - err : 0)) return 0;
        auto t0 = std::chrono::steady_clock::now();
        const uint64_t heads = live_heads();
        st_live_walks_++;
        std::fprintf(stderr, "kvstore: live walk: heads %llu MiB vs estimate %llu MiB (%llu blind pushes), %.2f s\n",
                     (unsigned long long)(heads >> 20), (unsigned long long)(live >> 20), (unsigned long long)n,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        rebase_live(heads);
        if (!trips(heads)) return 0;
    }
    // A pre-run drain (StopSession) sweeps slots outside the op guard: never reset under it.
    std::unique_lock<std::mutex> dg(drain_mu_, std::try_to_lock);
    if (!dg.owns_lock()) return 0;
    return gc_run();
}

// Drop every in-memory reference to slots and log addresses (quiesced; every slot's contents are
// in the checkpointed log): empty index, empty arena, empty free pools, session slot caches cleared,
// in-flight async reads told to re-resolve (gc_gen_). The state of a fresh open before recovery.
void SlotStore::reset_mem() {
    std::memset((void*)bk_, 0, nb_ * 64);
    std::memset((void*)ovbits_, 0, (nb_ + 63) / 64 * 8);
    {
        std::lock_guard<std::mutex> g(ov_mu_);
        const int n = nov_.load();
        for (int k = 0; k < n; k++) {
            munmap(ov_[k].cls, (ov_[k].mask + 1) * 8);
            munmap(ov_[k].ent, (ov_[k].mask + 1) * 8);
            ov_[k] = OvTable();
        }
        ov_[0].cls = (std::atomic<uint64_t>*)map_anon(ovcap0_ * 8, true);
        ov_[0].ent = (std::atomic<uint64_t>*)map_anon(ovcap0_ * 8, true);
        ov_[0].mask = ovcap0_ - 1;
        ov_[0].count = 0;
        nov_.store(1, std::memory_order_release);
        ov_total_.store(0);
        ov_dead_.store(0);
    }
    if (::madvise(arena_, nslots_ * sizeof(Slot), MADV_DONTNEED) != 0) die("madvise arena");
    // Every slot free: flags 0, stripes unlocked (no write-behind is in flight here).
    if (::madvise(side_, nside_ * sizeof(SideLine), MADV_DONTNEED) != 0) die("madvise side lines");
    fresh_.store(0);
    hand_.store(0);
    chand_.store(0);
    {
        std::lock_guard<std::mutex> g(pool_mu_);
        chains_.clear();
        recycled_.store(0);
    }
    std::memset((void*)dmark_, 0, nchunks_);
    std::memset((void*)dlist_, 0, nchunks_ * 4);
    {
        std::lock_guard<std::mutex> g(sess_mu_);
        for (Session* s : sessions_) {
            s->fcache.clear();
            s->tpend.clear();
            for (Pend& pd : s->pend) {
                pd.ep = nullptr;
                pd.head_e = 0;
            }
        }
    }
    gc_gen_.fetch_add(1, std::memory_order_relaxed);
}

// Full compaction of the checkpointed log into fresh segments (quiesced, drain_mu_ held, right after
// a checkpoint published: the log alone holds the whole state). 1. Drop the in-memory state and
// rebuild the index from the old segments, as recovery does (highest seq per key; multi-key classes
// get fresh chains, written into the new segments). 2. One sequential pass over the old segments
// copies each solo class's current record (same key, seq and value; prev 0) into the new segments
// and repoints its entry. Tombstones and stale versions are dropped with the old segments. 3. Durable
// order: new data + fsync, a manifest without the old segments (atomic replace + dir fsync), then
// unlink the old segments. A crash before the manifest recovers the old checkpoint (the new
// segments are outside its set and removed); after it, the same state from the new segments.
uint64_t SlotStore::gc_run() {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<SegRange> old;
    uint64_t old_bits[kSegWords];
    {
        std::lock_guard<std::mutex> g(alloc_mu_);
        const uint64_t t = tail_.load(), act = t >> kSegShift;
        std::memcpy(old_bits, seg_used_, sizeof old_bits);
        for (uint32_t sg = 0; sg < kMaxSeg; sg++)
            if ((old_bits[sg / 64] >> (sg % 64)) & 1) {
                SegRange r{sg, sg == act ? (t & (kSegSize - 1)) : kSegSize};
                r.cnt = seg_cnt_[sg].load(std::memory_order_acquire);
                r.dig = seg_dig_[sg].load(std::memory_order_acquire);
                r.chk = true;  // quiesced right after the manifest: the totals are exactly the range's
                old.push_back(r);
            }
    }
    uint64_t keys = 0;
    for (uint64_t i = 0; i < nb_ * 8; i++) keys += bk_[i].load(std::memory_order_relaxed) != 0;
    for (int k = 0; k < nov_.load(); k++)
        for (uint64_t i = 0; i <= ov_[k].mask; i++) keys += ov_[k].ent[i].load(std::memory_order_relaxed) != 0;
    const uint64_t log0 = log_bytes_.load();

    reset_mem();
    log_switch();
    rebuild_index(old, keys + keys / 16 + 1024, seq_.load());

    // Copy pass.
    struct Cp {
        std::atomic<uint64_t>* ep;
        uint32_t boff, bytes;
    };
    std::vector<Cp> cps;
    const size_t kScan = 1u << 20;
    uint8_t* sbuf = aligned_buf(kScan);
    uint8_t* buf = aligned_buf(kBatch);
    uint64_t pos = 0, ncopy = 0;
    auto flush = [&] {
        if (!pos) return;
        uint64_t off = write_victims(buf, pos, 0);  // accounting re-based after the GC
        for (const Cp& c : cps) {
            uint64_t e = c.ep->load(std::memory_order_relaxed);
            c.ep->store((e & kHdrBits) | mk_ptr(off + c.boff, c.bytes) | kPtrT, std::memory_order_relaxed);
        }
        cps.clear();
        std::memset(buf, 0, pos);
        pos = 0;
    };
    for (const SegRange& sr : old) {
        scan_seg(sr, sbuf, kScan, [&](const RecView& r, uint64_t off, uint32_t bytes) {
            if (r.tomb) return true;
            std::atomic<uint64_t>* ep = find_entry(classify(r.key));
            if (!ep) return true;
            const uint64_t e = ep->load(std::memory_order_relaxed);
            if ((e & kIsSlot) || !(e & kPtrT) || ((e & kPay) & ~kPtrT) != mk_ptr(off, bytes)) return true;
            uint64_t at = place(pos, bytes);
            if (at + round_up(bytes, kUnit) > kBatch) {
                flush();
                at = 0;
            }
            enc_record(buf + at, r.key, r.seq, false, 0, r.val, r.size);
            cps.push_back({ep, (uint32_t)at, bytes});
            pos = at + round_up(bytes, kUnit);
            ncopy++;
            return true;
        });
    }
    flush();
    free(buf);
    free(sbuf);
    if (std::getenv("KVSTORE_DEBUG")) {  // no entry may point into a freed segment
        auto check = [&](uint64_t e) {
            if (!e || (e & kIsSlot)) return;
            uint64_t p = e & kPay & ~kPtrT;
            uint64_t sg = ptr_byte(p) >> kSegShift;
            if (p && ((old_bits[sg / 64] >> (sg % 64)) & 1)) die("gc: entry still points into a freed segment", (long long)sg);
        };
        for (uint64_t i = 0; i < nb_ * 8; i++) check(bk_[i].load());
        for (int k = 0; k < nov_.load(); k++)
            for (uint64_t i = 0; i <= ov_[k].mask; i++) check(ov_[k].ent[i].load());
    }

    sync_segments();
    {
        std::lock_guard<std::mutex> g(alloc_mu_);
        for (uint32_t w = 0; w < kSegWords; w++) seg_used_[w] &= ~old_bits[w];
    }
    write_manifest();
    for (const SegRange& sr : old) {
        int fd;
        {
            std::lock_guard<std::mutex> g(seg_mu_);
            fd = fds_[sr.seg].exchange(-1, std::memory_order_acq_rel);
        }
        if (fd >= 0) ::close(fd);
        segdirty_[sr.seg].store(0, std::memory_order_relaxed);
        char name[64];
        std::snprintf(name, sizeof name, "seg_%06u.log", (unsigned)sr.seg);
        if (::unlinkat(dirfd_, name, 0) != 0 && errno != ENOENT) die("unlink freed segment", (long long)sr.seg);
    }
    if (::fsync(dirfd_) != 0) die("fsync dir");
    log_bytes_.fetch_sub(log0);
    rebase_live(log_bytes_.load());
    st_gc_runs_++;
    st_gc_segs_ += old.size();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "kvstore: gc freed %zu segments (%llu MiB), copied %llu records (%llu MiB live), %.2f s\n",
                 old.size(), (unsigned long long)(log0 >> 20), (unsigned long long)ncopy,
                 (unsigned long long)(live_base_ >> 20), secs);
    return old.size();
}

// ───────────────────────────── stats ─────────────────────────────
Counters SlotStore::sum_counters() const {
    Counters t;
    std::lock_guard<std::mutex> g(sess_mu_);
    for (const Session* s : sessions_) {
        const Counters& c = s->c;
        t.read_mem += c.read_mem;
        t.read_disk_found += c.read_disk_found;
        t.read_notfound += c.read_notfound;
        t.ios += c.ios;
        t.hops += c.hops;
        t.installs += c.installs;
        t.ups_inplace += c.ups_inplace;
        t.ups_push += c.ups_push;
        t.slow_writes += c.slow_writes;
        t.rmw_hit += c.rmw_hit;
        t.rmw_miss += c.rmw_miss;
        t.slot_waits += c.slot_waits;
        t.help_evicts += c.help_evicts;
        t.depth_sum += c.depth_sum;
        t.enters += c.enters;
        t.reaped += c.reaped;
        t.deletes += c.deletes;
        t.live_sub += c.live_sub;
        t.lat_n += c.lat_n;
        t.lat_sum_ns += c.lat_sum_ns;
        t.lag_sum_ns += c.lag_sum_ns;
        for (int i = 0; i < Counters::kHB; i++) {
            t.lat_h[i] += c.lat_h[i];
            t.lag_h[i] += c.lag_h[i];
        }
    }
    return t;
}

void SlotStore::print_stats(const char* tag) {
    Counters t = sum_counters();
    uint64_t resident = std::min<uint64_t>(fresh_.load(), nslots_) - recycled_.load();
    std::fprintf(stderr,
                 "kvstore[%s]: slots=%llu resident~%llu read_mem=%llu read_disk=%llu read_nf=%llu ios=%llu hops=%llu "
                 "installs=%llu ups_inplace=%llu ups_push=%llu slow_writes=%llu slot_waits=%llu help_evicts=%llu evictions=%llu "
                 "evict_aborts=%llu flush_sync=%llu written=%llu MiB\n",
                 tag, (unsigned long long)nslots_, (unsigned long long)resident, (unsigned long long)t.read_mem,
                 (unsigned long long)t.read_disk_found, (unsigned long long)t.read_notfound,
                 (unsigned long long)t.ios, (unsigned long long)t.hops, (unsigned long long)t.installs,
                 (unsigned long long)t.ups_inplace, (unsigned long long)t.ups_push,
                 (unsigned long long)t.slow_writes, (unsigned long long)t.slot_waits, (unsigned long long)t.help_evicts,
                 (unsigned long long)st_evict_.load(), (unsigned long long)st_evict_abort_.load(),
                 (unsigned long long)st_flush_sync_.load(), (unsigned long long)(st_bytes_written_.load() >> 20));
    std::fprintf(stderr,
                 "kvstore[%s]: clean_drops=%llu dirty_evictions=%llu cleaned=%llu clean_aborts=%llu depth_sum=%llu "
                 "enters=%llu ov_compactions=%llu\n",
                 tag, (unsigned long long)st_drop_clean_.load(), (unsigned long long)st_evict_dirty_.load(),
                 (unsigned long long)st_cleaned_.load(), (unsigned long long)st_clean_abort_.load(),
                 (unsigned long long)t.depth_sum, (unsigned long long)t.enters,
                 (unsigned long long)st_ov_compact_.load());
}

// Run-window counters, printed once at the first Checkpoint after the run (before it writes), so
// the bytes exclude the checkpoint. IOPS are over the active window: run start to the last moment
// the cold-read counter moved (as seen by the cleaner's 1-3 ms monitor). Re-reads are estimated
// from a 1/2048 key sample: a cold read of a key already accessed in the run (evicted and read
// again) or with a cold read already in flight (duplicate).
void SlotStore::print_run_stats(const char* when) {
    Counters t = sum_counters();
    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(now - run_t0_).count();
    int64_t t0ns = std::chrono::duration_cast<std::chrono::nanoseconds>(run_t0_.time_since_epoch()).count();
    int64_t lastns = last_io_ns_.load();
    double active = lastns > t0ns ? (lastns - t0ns) / 1e9 : secs;
    if (active <= 0 || active > secs) active = secs;
    uint64_t ios = t.ios - run_ios_;
    uint64_t ops = t.read_mem + t.read_disk_found + t.read_notfound + t.ups_inplace + t.ups_push + t.rmw_hit + t.rmw_miss -
                   run_ops_;
    double scale = (double)(kSampMask + 1);
    std::fprintf(stderr,
                 "kvstore[run %s]: resident_at_run_start=%llu (%.3f of %llu slots) ops=%llu cold_ios=%llu "
                 "active=%.2f s (wall %.2f s) IOPS_active=%.0f written_in_run=%llu MiB dirty_evictions=%llu "
                 "clean_drops=%llu cleaned_in_run=%llu rereads_est=%.0f dup_inflight_est=%.0f (sampled cold=%llu "
                 "reread=%llu dup=%llu) avg_inflight_at_submit=%.1f reads_per_enter=%.1f slot_waits=%llu "
                 "help_evicts=%llu\n",
                 when, (unsigned long long)run_resident_, nslots_ ? (double)run_resident_ / nslots_ : 0.0,
                 (unsigned long long)nslots_, (unsigned long long)ops, (unsigned long long)ios, active, secs,
                 ios / active, (unsigned long long)((st_bytes_written_.load() - run_bytes_) >> 20),
                 (unsigned long long)(st_evict_dirty_.load() - run_dirty_),
                 (unsigned long long)(st_drop_clean_.load() - run_drop_),
                 (unsigned long long)(st_cleaned_.load() - run_clean0_), st_reread_.load() * scale,
                 st_dup_inflight_.load() * scale, (unsigned long long)st_samp_cold_.load(),
                 (unsigned long long)st_reread_.load(), (unsigned long long)st_dup_inflight_.load(),
                 ios ? (double)(t.depth_sum - run_depth_) / ios : 0.0,
                 (t.enters - run_enters_) ? (double)ios / (t.enters - run_enters_) : 0.0,
                 (unsigned long long)(t.slot_waits - run_waits_), (unsigned long long)(t.help_evicts - run_helps_));
    // Cold-read latency (bucket b holds [2^b, 2^(b+1)) ns; percentiles are bucket upper bounds).
    if (t.lat_n) {
        auto pct = [&](const Ctr* h, double q) {
            uint64_t want = (uint64_t)(q * (double)t.lat_n), acc = 0;
            for (int i = 0; i < Counters::kHB; i++) {
                acc += h[i];
                if (acc > want) return (double)(2ULL << i) / 1000.0;
            }
            return 0.0;
        };
        double n = (double)t.lat_n;
        std::fprintf(stderr,
                     "kvstore[run %s]: cold reads timed=%llu submit->reap mean=%.1f us p50<=%.0f p90<=%.0f p99<=%.0f us; "
                     "reap lag (CQ last drained->reap, upper bound on CQE->done) mean=%.1f us p50<=%.0f p90<=%.0f "
                     "p99<=%.0f us; lag share of latency=%.1f%%\n",
                     when, (unsigned long long)t.lat_n, t.lat_sum_ns / n / 1000.0, pct(t.lat_h, 0.5), pct(t.lat_h, 0.9),
                     pct(t.lat_h, 0.99), t.lag_sum_ns / n / 1000.0, pct(t.lag_h, 0.5), pct(t.lag_h, 0.9),
                     pct(t.lag_h, 0.99), 100.0 * (double)t.lag_sum_ns / (double)t.lat_sum_ns);
        std::fprintf(stderr, "kvstore[run %s]: lat_hist_log2ns=", when);
        for (int i = 0; i < Counters::kHB; i++)
            if (t.lat_h[i] || t.lag_h[i])
                std::fprintf(stderr, "%d:%llu/%llu ", i, (unsigned long long)t.lat_h[i], (unsigned long long)t.lag_h[i]);
        std::fprintf(stderr, "(bucket:submit->reap/lag)\n");
    }
}

CacheStats SlotStore::GetCacheStats() const {
    Counters t = sum_counters();
    CacheStats cs;
    cs.read_hits = t.read_mem;
    cs.read_misses = t.ios;
    cs.rmw_hits = t.rmw_hit;
    cs.rmw_misses = t.rmw_miss;
    uint64_t resident = std::min<uint64_t>(fresh_.load(), nslots_) - recycled_.load();
    cs.hot_bytes = resident * sizeof(Slot);
    cs.total_bytes = cs.hot_bytes + log_bytes_.load();
    cs.budget_bytes = budget_;
    cs.evictions = st_evict_.load();
    return cs;
}

}  // namespace

IKVStore* create_kvstore() { return new SlotStore(); }
