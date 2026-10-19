#ifndef EMULATOR_INCLUDE_COMMON_DEBUGTRACERANGE_H_
#define EMULATOR_INCLUDE_COMMON_DEBUGTRACERANGE_H_

// Debug: KYTY_TRACE_RANGE=<base>,<size> (hex) logs host-side events that touch that guest range:
// shader specialization reads, written buffer bindings and indirect dispatch arguments.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Common::DebugTrace {

struct Range {
	uint64_t base = 0;
	uint64_t size = 0;
};

inline const Range& TraceRange() {
	static const Range range = [] {
		Range       result;
		const char* value = std::getenv("KYTY_TRACE_RANGE");
		if (value != nullptr) {
			char* end   = nullptr;
			result.base = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ',') result.size = std::strtoull(end + 1, nullptr, 16);
		}
		return result;
	}();
	return range;
}

inline bool Enabled() {
	return TraceRange().size != 0;
}

inline bool Overlaps(uint64_t address, uint64_t size) {
	const auto& range = TraceRange();
	return range.size != 0 && address < range.base + range.size && range.base < address + size;
}

// The shader whose program is being prepared on this thread.
inline uint64_t& CurrentShader() {
	static thread_local uint64_t hash = 0;
	return hash;
}

// Listed shaders (KYTY_TRACE_SHADERS, comma-separated hex hashes) also log their dispatches.
inline bool Listed(uint64_t hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_TRACE_SHADERS");
		while (value != nullptr && *value != '\0') {
			char*      end  = nullptr;
			const auto item = std::strtoull(value, &end, 16);
			if (end == value) break;
			result.push_back(item);
			value = *end == ',' ? end + 1 : end;
		}
		return result;
	}();
	for (const auto item: hashes)
		if (item == hash) return true;
	return false;
}

inline bool Budget() {
	static std::atomic<uint32_t> lines {0};
	return lines.fetch_add(1, std::memory_order_relaxed) < 3000000u;
}

} // namespace Common::DebugTrace

#endif
