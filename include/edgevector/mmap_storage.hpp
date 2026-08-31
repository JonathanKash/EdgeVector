#ifndef EDGEVECTOR_MMAP_STORAGE_HPP
#define EDGEVECTOR_MMAP_STORAGE_HPP

// ============================================================================
// EdgeVector :: mmap_storage.hpp
//
// Zero-copy disk I/O for a quantized index: a build-time writer and a
// read-only memory-mapped reader.
//
// ENDIANNESS
// ----------
// The version-1 on-disk format is LITTLE-ENDIAN ONLY. Multi-byte header fields
// are stored in host byte order and read back with std::memcpy, so a file
// written on a little-endian machine (x86, ARM in its usual configuration) is
// only portable to another little-endian machine. No byte-swapping is
// performed. This matches the bit layout of quantize_math.hpp, which likewise
// assumes a little-endian target.
//
// ON-DISK FORMAT (version 1)
// --------------------------
//   Bytes  0.. 3  char          magic[4] = { 'E', 'V', 'E', 'C' }
//   Bytes  4.. 7  std::uint32_t version  = 1
//   Bytes  8..15  std::uint64_t dim      (> 0)
//   Bytes 16..23  std::uint64_t count    (may be 0)
//   Bytes 24..63  40 bytes of zeroed reserved padding
//   Bytes 64..    `count` contiguous records of padded_bytes(dim) bytes each,
//                 with no per-record framing.
//
//   Total file size MUST equal 64 + count * padded_bytes(dim); open()
//   validates this exactly and rejects anything else as corrupt_size.
//
// ALIGNMENT INVARIANT
// -------------------
// An OS mapping always begins on a page boundary (4 KiB or larger), so the
// mapping base is at least 8-byte aligned. The header is exactly 64 bytes and
// every record is padded_bytes(dim) bytes, which is always a multiple of 8.
// Therefore EVERY record returned by vector() is 8-byte aligned, which is the
// precondition hamming_distance() requires of its operands.
//
// CRITICAL-PATH CONTRACT (see CLAUDE.md section 3)
// ------------------------------------------------
// vector(), dim(), count(), record_bytes() and is_open() are noexcept, perform
// zero heap allocation and make no system calls: they are pure reads of fields
// cached at open() time. In particular the data-region base pointer and the
// record size are computed once during open(), so vector() is a single
// multiply-add with no division. open(), close() and write_storage_file() are
// load-time/build-time operations where allocation and buffered I/O are fine.
//
// This module throws no exceptions; every failure is reported as a
// StorageStatus value.
// ============================================================================

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "edgevector/quantize_math.hpp"

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace edgevector {

enum class StorageStatus : std::uint8_t {
    ok = 0,
    io_error,     // open/stat/map failed
    bad_magic,
    bad_version,
    corrupt_size  // file size != 64 + count * padded_bytes(dim), or dim == 0
};

namespace detail {

// --- Format constants -------------------------------------------------------

constexpr std::size_t   kHeaderBytes   = 64u;
constexpr std::uint32_t kFormatVersion = 1u;

constexpr std::size_t kOffsetMagic   = 0u;
constexpr std::size_t kOffsetVersion = 4u;
constexpr std::size_t kOffsetDim     = 8u;
constexpr std::size_t kOffsetCount   = 16u;

constexpr char kMagic0 = 'E';
constexpr char kMagic1 = 'V';
constexpr char kMagic2 = 'E';
constexpr char kMagic3 = 'C';

// --- Shared, platform-free arithmetic --------------------------------------

// True when `v` survives a round trip through std::size_t. Written as a round
// trip rather than a comparison against SIZE_MAX so that it does not become a
// tautological comparison (and a -Wtype-limits warning) on 64-bit targets.
inline bool fits_in_size_t(std::uint64_t v) noexcept {
    return static_cast<std::uint64_t>(static_cast<std::size_t>(v)) == v;
}

// Record size in bytes for `dim`, computed entirely in 64-bit arithmetic so a
// hostile header cannot overflow the intermediate. Mirrors padded_bytes().
// Returns false if dim == 0 or the rounding step would overflow.
inline bool record_bytes_checked(std::uint64_t dim, std::uint64_t& out) noexcept {
    if (dim == 0u) {
        return false;
    }
    if (dim > UINT64_MAX - 63u) {
        return false;
    }
    out = ((dim + 63u) / 64u) * 8u;
    return true;
}

// Total file size for (dim, count), guarding the multiplication against
// unsigned 64-bit overflow *before* it happens. Returns false on overflow.
inline bool total_bytes_checked(std::uint64_t record_bytes,
                                std::uint64_t count,
                                std::uint64_t& out) noexcept {
    if (record_bytes != 0u) {
        const std::uint64_t headroom =
            (UINT64_MAX - static_cast<std::uint64_t>(kHeaderBytes)) / record_bytes;
        if (count > headroom) {
            return false;
        }
    }
    out = static_cast<std::uint64_t>(kHeaderBytes) + count * record_bytes;
    return true;
}

// ---------------------------------------------------------------------------
// PLATFORM LAYER - the only platform-dependent code in this module.
//
// Primary backend is POSIX mmap. The _WIN32 branch is a shim so the suite can
// be built and run under MinGW-w64, which has no POSIX mmap. Everything above
// and below this block is platform-free.
// ---------------------------------------------------------------------------

struct MappedRegion {
    const std::uint8_t* base = nullptr;
    std::size_t         size = 0u;
#if defined(_WIN32)
    HANDLE file    = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int fd = -1;
#endif
};

#if defined(_WIN32)

inline bool map_file_readonly(const char* path, MappedRegion& out) noexcept {
    out = MappedRegion();

    HANDLE file = ::CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER file_size;
    if (::GetFileSizeEx(file, &file_size) == 0 || file_size.QuadPart <= 0) {
        ::CloseHandle(file);
        return false;
    }

    const std::uint64_t bytes = static_cast<std::uint64_t>(file_size.QuadPart);
    if (!fits_in_size_t(bytes)) {
        ::CloseHandle(file);
        return false; // >4GB file on a 32-bit target: cannot be mapped
    }

    HANDLE mapping =
        ::CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        ::CloseHandle(file);
        return false;
    }

    LPVOID view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        ::CloseHandle(mapping);
        ::CloseHandle(file);
        return false;
    }

    out.base    = static_cast<const std::uint8_t*>(view);
    out.size    = static_cast<std::size_t>(bytes);
    out.file    = file;
    out.mapping = mapping;
    return true;
}

inline void unmap_region(MappedRegion& region) noexcept {
    if (region.base != nullptr) {
        ::UnmapViewOfFile(static_cast<LPCVOID>(region.base));
    }
    if (region.mapping != nullptr) {
        ::CloseHandle(region.mapping);
    }
    if (region.file != INVALID_HANDLE_VALUE) {
        ::CloseHandle(region.file);
    }
    region = MappedRegion();
}

#else

inline bool map_file_readonly(const char* path, MappedRegion& out) noexcept {
    out = MappedRegion();

    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return false;
    }

    struct stat info;
    if (::fstat(fd, &info) != 0 || info.st_size <= 0) {
        ::close(fd);
        return false;
    }

    const std::uint64_t bytes = static_cast<std::uint64_t>(info.st_size);
    if (!fits_in_size_t(bytes)) {
        ::close(fd);
        return false; // >4GB file on a 32-bit target: cannot be mapped
    }

    void* view = ::mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ,
                        MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) {
        ::close(fd);
        return false;
    }

    out.base = static_cast<const std::uint8_t*>(view);
    out.size = static_cast<std::size_t>(bytes);
    out.fd   = fd;
    return true;
}

inline void unmap_region(MappedRegion& region) noexcept {
    if (region.base != nullptr) {
        ::munmap(const_cast<void*>(static_cast<const void*>(region.base)),
                 region.size);
    }
    if (region.fd >= 0) {
        ::close(region.fd);
    }
    region = MappedRegion();
}

#endif // platform layer

} // namespace detail

// ---------------------------------------------------------------------------
// Build-time writer. Allocation and buffered I/O are permitted here; this is
// never on the query path. `vectors` points at count * padded_bytes(dim) bytes
// laid out per Task 1. Any existing file at `path` is overwritten.
// ---------------------------------------------------------------------------
inline StorageStatus write_storage_file(const char* path,
                                        std::uint64_t dim,
                                        std::uint64_t count,
                                        const std::uint8_t* vectors) {
    if (path == nullptr) {
        return StorageStatus::io_error;
    }

    std::uint64_t record_bytes = 0u;
    if (!detail::record_bytes_checked(dim, record_bytes)) {
        return StorageStatus::corrupt_size; // dim == 0 or overflow
    }
    std::uint64_t total_bytes = 0u;
    if (!detail::total_bytes_checked(record_bytes, count, total_bytes)) {
        return StorageStatus::io_error; // would overflow a 64-bit size
    }
    if (!detail::fits_in_size_t(total_bytes)) {
        return StorageStatus::io_error; // unrepresentable on this target
    }
    if (count > 0u && vectors == nullptr) {
        return StorageStatus::io_error;
    }

    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        return StorageStatus::io_error;
    }

    // Header: zero-filled first, so the 40 reserved bytes are written as zero.
    std::uint8_t header[detail::kHeaderBytes];
    std::memset(header, 0, sizeof(header));

    const char magic[4] = { detail::kMagic0, detail::kMagic1,
                            detail::kMagic2, detail::kMagic3 };
    const std::uint32_t version = detail::kFormatVersion;
    std::memcpy(header + detail::kOffsetMagic, magic, sizeof(magic));
    std::memcpy(header + detail::kOffsetVersion, &version, sizeof(version));
    std::memcpy(header + detail::kOffsetDim, &dim, sizeof(dim));
    std::memcpy(header + detail::kOffsetCount, &count, sizeof(count));

    if (std::fwrite(header, 1u, sizeof(header), file) != sizeof(header)) {
        std::fclose(file);
        return StorageStatus::io_error;
    }

    // One fwrite per record keeps every length within std::size_t even when
    // the total payload does not fit.
    const std::size_t record_span = static_cast<std::size_t>(record_bytes);
    for (std::uint64_t i = 0u; i < count; ++i) {
        const std::uint8_t* src =
            vectors + static_cast<std::size_t>(i) * record_span;
        if (std::fwrite(src, 1u, record_span, file) != record_span) {
            std::fclose(file);
            return StorageStatus::io_error;
        }
    }

    if (std::fclose(file) != 0) {
        return StorageStatus::io_error;
    }
    return StorageStatus::ok;
}

// ---------------------------------------------------------------------------
// Zero-copy reader over a memory-mapped index file.
// ---------------------------------------------------------------------------
class MMapStorage {
public:
    MMapStorage() noexcept = default;

    ~MMapStorage() { close(); }

    MMapStorage(const MMapStorage&) = delete;
    MMapStorage& operator=(const MMapStorage&) = delete;

    MMapStorage(MMapStorage&& other) noexcept
        : region_(other.region_),
          data_base_(other.data_base_),
          dim_(other.dim_),
          count_(other.count_),
          record_bytes_(other.record_bytes_) {
        other.reset_fields();
    }

    MMapStorage& operator=(MMapStorage&& other) noexcept {
        if (this != &other) {
            close();
            region_       = other.region_;
            data_base_    = other.data_base_;
            dim_          = other.dim_;
            count_        = other.count_;
            record_bytes_ = other.record_bytes_;
            other.reset_fields();
        }
        return *this;
    }

    // Maps the file read-only and validates the header. On any failure the
    // object is left in the empty state. Allocation is permitted here: open()
    // is load-time, not query-time.
    StorageStatus open(const char* path) {
        close();

        if (path == nullptr) {
            return StorageStatus::io_error;
        }

        detail::MappedRegion region;
        if (!detail::map_file_readonly(path, region)) {
            return StorageStatus::io_error;
        }

        // Guard the header read itself before touching any field.
        if (region.size < detail::kHeaderBytes) {
            detail::unmap_region(region);
            return StorageStatus::corrupt_size;
        }

        // Header fields are memcpy'd out of the mapping into locals; the raw
        // bytes are never reinterpret_cast to a struct.
        char magic[4];
        std::memcpy(magic, region.base + detail::kOffsetMagic, sizeof(magic));
        if (magic[0] != detail::kMagic0 || magic[1] != detail::kMagic1 ||
            magic[2] != detail::kMagic2 || magic[3] != detail::kMagic3) {
            detail::unmap_region(region);
            return StorageStatus::bad_magic;
        }

        std::uint32_t version = 0u;
        std::memcpy(&version, region.base + detail::kOffsetVersion,
                    sizeof(version));
        if (version != detail::kFormatVersion) {
            detail::unmap_region(region);
            return StorageStatus::bad_version;
        }

        std::uint64_t dim = 0u;
        std::uint64_t count = 0u;
        std::memcpy(&dim, region.base + detail::kOffsetDim, sizeof(dim));
        std::memcpy(&count, region.base + detail::kOffsetCount, sizeof(count));

        std::uint64_t record_bytes = 0u;
        if (!detail::record_bytes_checked(dim, record_bytes)) {
            detail::unmap_region(region);
            return StorageStatus::corrupt_size; // dim == 0 or absurd
        }

        std::uint64_t expected_bytes = 0u;
        if (!detail::total_bytes_checked(record_bytes, count, expected_bytes)) {
            detail::unmap_region(region);
            return StorageStatus::io_error; // 64-bit overflow
        }
        if (!detail::fits_in_size_t(expected_bytes)) {
            detail::unmap_region(region);
            return StorageStatus::io_error; // unmappable on this target
        }
        if (expected_bytes != static_cast<std::uint64_t>(region.size)) {
            detail::unmap_region(region);
            return StorageStatus::corrupt_size;
        }

        // Cache everything the query path needs, so vector() never divides,
        // never allocates and never makes a system call.
        region_       = region;
        data_base_    = region.base + detail::kHeaderBytes;
        dim_          = dim;
        count_        = count;
        record_bytes_ = static_cast<std::size_t>(record_bytes);
        return StorageStatus::ok;
    }

    // Idempotent: safe to call on an already-closed object.
    void close() noexcept {
        if (region_.base != nullptr) {
            detail::unmap_region(region_);
        }
        reset_fields();
    }

    bool is_open() const noexcept { return data_base_ != nullptr; }

    std::uint64_t dim() const noexcept { return dim_; }
    std::uint64_t count() const noexcept { return count_; }
    std::size_t record_bytes() const noexcept { return record_bytes_; }

    // QUERY CRITICAL PATH: returns a pointer directly into the mapping.
    // Zero-copy, zero-allocation, no system calls.
    //
    // PRECONDITION: is_open() && i < count(). Violating it is undefined
    // behaviour; there is no bounds check beyond the debug assert below,
    // because this runs once per candidate during a search.
    const std::uint8_t* vector(std::uint64_t i) const noexcept {
        assert(data_base_ != nullptr && "vector() on a closed MMapStorage");
        assert(i < count_ && "vector() index out of range");
        return data_base_ + static_cast<std::size_t>(i) * record_bytes_;
    }

private:
    void reset_fields() noexcept {
        region_       = detail::MappedRegion();
        data_base_    = nullptr;
        dim_          = 0u;
        count_        = 0u;
        record_bytes_ = 0u;
    }

    detail::MappedRegion region_;
    const std::uint8_t*  data_base_    = nullptr; // region_.base + 64, cached
    std::uint64_t        dim_          = 0u;
    std::uint64_t        count_        = 0u;
    std::size_t          record_bytes_ = 0u;      // padded_bytes(dim), cached
};

} // namespace edgevector

#endif // EDGEVECTOR_MMAP_STORAGE_HPP
