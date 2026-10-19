#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/assert.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <bit>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

void EmitKillIfBoolFalse(EmitterState& state, uint32_t active) {
	const auto kill_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	const auto inactive    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), inactive, active);
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, inactive, kill_label, merge_label);
	EmitLabel(state, kill_label);
	state.builder.AddFunction(spv::OpKill);
	EmitLabel(state, merge_label);
}

void EmitKillIfPixelValidMaskInactive(EmitterState& state) {
	if (state.pixel_valid_mask_variable == 0) {
		return;
	}

	const auto mask_value = state.builder.AllocateId();
	const auto active     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), mask_value,
	                          state.pixel_valid_mask_variable);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, mask_value,
	                          ConstantU32(state, 0));
	EmitKillIfBoolFalse(state, active);
}

uint32_t SpillPointerType(ValueEmitContext& ctx, IR::Type type) {
	const auto value_type = TypeId(ctx.state, type);
	return value_type == 0 ? 0 : TypePointer(ctx.state, spv::StorageClassFunction, value_type);
}

struct DeferredPhiPatch {
	DeferredPhi     phi;
	const IR::Inst* instruction = nullptr;
	uint32_t        half        = 0;
};

struct StructuredFunctionState {
	std::unordered_map<const IR::Block*, uint32_t> block_exit_labels;
	std::vector<DeferredPhiPatch>                  deferred_phis;
};

struct DispatcherFunctionState {
	std::array<std::unordered_map<const IR::Inst*, uint32_t>, 2> spills;
	uint32_t                                      header_label       = 0;
	uint32_t                                      select_label       = 0;
	uint32_t                                      after_switch_label = 0;
	uint32_t                                      continue_label     = 0;
	uint32_t                                      merge_label        = 0;
};

void StoreDispatcherPhiEdge(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                            const IR::Block* from, const IR::Block* to) {
	if (to == nullptr) {
		return;
	}
	for (const auto& phi: *to) {
		if (phi.GetOpcode() != IR::ValueOpcode::Phi) {
			break;
		}
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			if (phi.PhiBlock(index) == from) {
				ctx.state.builder.AddFunction(spv::OpStore, dispatcher.spills[ctx.half].at(&phi),
				                              ctx.Def(phi.Arg(index)));
				break;
			}
		}
	}
}

void EmitReturn(ValueEmitContext& ctx) {
	EmitKillIfPixelValidMaskInactive(ctx.state);
	if (ctx.state.object_primitive_id_variable != 0) {
		auto& state = ctx.state;
		const auto primitive = EmitGetBuiltin(
		    ctx, IR::Value {static_cast<uint32_t>(IR::StageInputKind::PrimitiveId)}, IR::Value {0u});
		const auto object = EmitPushDataLoad(state, ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.object_primitive_id_variable,
		                          EmitAddU32(state, object, primitive));
	}
	ctx.state.builder.AddFunction(spv::OpReturn);
}

void EmitStructuredTerminator(ValueEmitContext& ctx, const IR::Block* block) {
	const auto& term       = block->terminator;
	const auto  emit_merge = [&]() {
		if (term.loop_header) {
			const auto* merge = term.merge_block;
			const auto* cont  = term.continue_block;
			if (merge != nullptr && cont != nullptr) {
				ctx.state.builder.AddFunction(spv::OpLoopMerge, merge->Definition(),
				                              cont->Definition(), spv::LoopControlMaskNone);
			}
		} else if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
		           term.merge_block != nullptr) {
			ctx.state.builder.AddFunction(spv::OpSelectionMerge, term.merge_block->Definition(),
			                              spv::SelectionControlMaskNone);
		}
	};

	switch (term.kind) {
		case CFG::TerminatorKind::Branch: {
			const auto* target = term.true_block;
			if (target == nullptr) {
				EmitReturn(ctx);
				return;
			}
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranch, target->Definition());
			return;
		}
		case CFG::TerminatorKind::ConditionalBranch: {
			const auto* true_block  = term.true_block;
			const auto* false_block = term.false_block;
			if (true_block == nullptr || false_block == nullptr || block->condition.IsEmpty()) {
				EmitReturn(ctx);
				return;
			}
			const auto condition = ctx.Def(block->condition);
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranchConditional, condition,
			                              true_block->Definition(), false_block->Definition());
			return;
		}
		default: EmitReturn(ctx); return;
	}
}

void EmitDispatcherTarget(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                          const IR::Block* from, const IR::Block* block) {
	if (block != nullptr) {
		StoreDispatcherPhiEdge(ctx, dispatcher, from, block);
		if (ctx.other_half != nullptr) {
			StoreDispatcherPhiEdge(*ctx.other_half, dispatcher, from, block);
		}
	}
}

uint32_t EmitDispatcherNextPc(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                              const IR::Block* block) {
	const auto& term = block->terminator;
	switch (term.kind) {
		case CFG::TerminatorKind::Branch:
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			return ConstantU32(ctx.state, term.true_block->id);
		case CFG::TerminatorKind::ConditionalBranch: {
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			EmitDispatcherTarget(ctx, dispatcher, block, term.false_block);
			const auto selected = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), selected,
			                              ctx.Def(block->condition),
			                              ConstantU32(ctx.state, term.true_block->id),
			                              ConstantU32(ctx.state, term.false_block->id));
			return selected;
		}
		case CFG::TerminatorKind::IndirectBranch: {
			for (const auto* target: block->ImmSuccessors()) {
				EmitDispatcherTarget(ctx, dispatcher, block, target);
			}
			uint32_t selected = ConstantU32(ctx.state, UINT32_MAX);
			if (!block->indirect_target.IsEmpty()) {
				const auto selector = ctx.Def(block->indirect_target);
				for (const auto& branch: term.cases) {
					const auto match = ctx.state.builder.AllocateId();
					const auto next  = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpIEqual, TypeBool(ctx.state), match,
					                              selector, ConstantU32(ctx.state, branch.value));
					ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), next, match,
					                              ConstantU32(ctx.state, branch.target->id),
					                              selected);
					selected = next;
				}
			}
			return selected;
		}
		default: return ConstantU32(ctx.state, UINT32_MAX);
	}
}

template <typename T>
decltype(auto) Arg(ValueEmitContext& ctx, const IR::Inst& inst, size_t index) {
	if constexpr (std::is_same_v<T, const IR::Inst&>) {
		return inst;
	} else if constexpr (std::is_same_v<T, IR::Value>) {
		return inst.Arg(index);
	} else if constexpr (std::is_same_v<T, IR::ScalarReg>) {
		return inst.Arg(index).ScalarRegister();
	} else {
		static_assert(std::is_same_v<T, uint32_t>);
		return ctx.Def(inst.Arg(index));
	}
}

template <typename Context, typename Return, typename... Args>
void Invoke(Return (*emit)(Context&, Args...), ValueEmitContext& ctx, const IR::Inst& inst) {
	// A full instruction keeps metadata and predicated/lane operand loads lazy.
	static_assert(std::is_same_v<Context, ValueEmitContext> ||
	              std::is_same_v<Context, EmitterState>);
	auto& context = [&]() -> Context& {
		if constexpr (std::is_same_v<Context, EmitterState>)
			return ctx.state;
		else
			return ctx;
	}();
	constexpr bool has_inst = (std::is_same_v<Args, const IR::Inst&> || ...);
	[&]<size_t... I>(std::index_sequence<I...>) {
		static_assert(((!std::is_same_v<Args, const IR::Inst&> || I == 0) && ...));
		const auto call = [&] {
			return emit(context, Arg<Args>(ctx, inst, I - (has_inst && I != 0))...);
		};
		if constexpr (std::is_void_v<Return>) {
			call();
		} else {
			static_assert(std::is_same_v<Return, uint32_t>);
			ctx.Define(inst, call());
		}
	}(std::index_sequence_for<Args...> {});
}

void EmitDirectInstruction(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (ctx.half != 0 && (inst.GetOpcode() == IR::ValueOpcode::Ballot ||
	                     inst.GetOpcode() == IR::ValueOpcode::ReadFirstLane ||
	                     inst.GetOpcode() == IR::ValueOpcode::SubgroupUMinU32)) {
		// These operations combine both emulated halves into one whole-wave result.
		ctx.Define(inst, ctx.other_half->Result(inst));
		return;
	}
	switch (inst.GetOpcode()) {
#define VALUE_OPCODE(name, ...)                                                                    \
	case IR::ValueOpcode::name: return Invoke(Emit##name, ctx, inst);
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.inc"
#undef VALUE_OPCODE
		default: ctx.Fail(inst, "has no direct SPIR-V emitter");
	}
}

void EmitStructuredInstruction(ValueEmitContext& ctx, StructuredFunctionState& structured,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0 || inst.NumArgs() == 0) {
			ctx.Fail(inst, "has no native SPIR-V representation");
		}
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			const auto* predecessor = inst.PhiBlock(index);
			if (predecessor == nullptr || predecessor->Definition() == 0) {
				ctx.Fail(inst, "has a predecessor outside the structured function");
			}
		}
		structured.deferred_phis.push_back(
		    {ctx.state.builder.AddDeferredPhi(type, ctx.Result(inst), inst.NumArgs()), &inst,
		     ctx.half});
		return;
	}
	EmitDirectInstruction(ctx, inst);
	if (TraceShaderEnabled(ctx.state)) EmitTraceMemory(ctx, inst);
}

void EmitDispatcherInstruction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0) {
			ctx.Fail(inst, "cannot be loaded by the dispatcher");
		}
		ctx.state.builder.AddFunction(spv::OpLoad, type, ctx.Result(inst),
		                              dispatcher.spills[ctx.half].at(&inst));
		return;
	}
	EmitDirectInstruction(ctx, inst);
	if (TraceShaderEnabled(ctx.state)) EmitTraceMemory(ctx, inst);
	if (const auto found = dispatcher.spills[ctx.half].find(&inst);
	    found != dispatcher.spills[ctx.half].end()) {
		ctx.state.builder.AddFunction(spv::OpStore, found->second,
		                              ctx.Def(IR::Value(const_cast<IR::Inst*>(&inst))));
	}
}

template <typename EmitInstruction>
void EmitBlock(ValueEmitContext& ctx, const IR::Block* block, EmitInstruction&& emit_instruction,
               const std::function<void()>& after_phis = {}) {
	ctx.state.current_block = block;
	EmitLabel(ctx.state, block->Definition());
	bool emitted_non_phi = false;
	for (const auto& inst: *block) {
		if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
			if (emitted_non_phi) {
				ctx.Fail(inst, "appears after a non-Phi instruction");
			}
		} else {
			if (!emitted_non_phi && after_phis) {
				after_phis();
			}
			emitted_non_phi = true;
		}
		for (uint32_t half = 0; half < ctx.state.lane_count; half++) {
			auto& lane          = half == 0 ? ctx : *ctx.other_half;
			ctx.state.lane_half = half;
			if (half == 0 || (inst.GetOpcode() != IR::ValueOpcode::Barrier &&
			                  inst.GetOpcode() != IR::ValueOpcode::StoreCompletion &&
			                  inst.GetOpcode() != IR::ValueOpcode::DataCacheInvalidate &&
			                  inst.GetOpcode() != IR::ValueOpcode::MeshAllocate)) {
				emit_instruction(lane, inst);
			}
		}
		ctx.state.lane_half = 0;
	}
	if (!emitted_non_phi && after_phis) {
		after_phis();
	}
}

// A BDA read of an unmapped page returns zero and records a fault, which is serviced only after
// the dispatch. A loop that waits on such memory (GTA V's BVH traversal walks a primitive list
// until it reads an end marker) would then spin until the driver resets the device. An invocation
// that has faulted and keeps looping returns instead; the page is mapped for the next dispatch.
constexpr uint32_t FaultedLoopIterationLimit = 4096;

} // namespace

// Debug: KYTY_LOOP_WATCHDOG=N stops every invocation after N loop iterations, faulted or not.
uint32_t LoopWatchdogLimit() {
	static const uint32_t limit = [] {
		const char* value = std::getenv("KYTY_LOOP_WATCHDOG");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return limit;
}

// Debug: KYTY_LOOP_PRINTF=<hash> prints, through SPIR-V debug printf, the loop header values of
// that shader's invocations the watchdog stops (first invocation of every 8th workgroup).
uint64_t LoopPrintfHash() {
	static const uint64_t hash = [] {
		const char* value = std::getenv("KYTY_LOOP_PRINTF");
		return value != nullptr ? static_cast<uint64_t>(std::strtoull(value, nullptr, 16)) : 0u;
	}();
	return hash;
}

bool LoopPrintfEnabled(const EmitterState& state) {
	return LoopWatchdogLimit() != 0 && state.loop_iteration_variable != 0 &&
	       LoopPrintfHash() == state.program.shader_hash;
}

uint32_t EmitDebugSample(EmitterState& state, uint32_t& wg_x, uint32_t& wg_y) {
	const auto u = TypeU32(state), b = TypeBool(state);
	auto sample = state.subgroup_local_invocation_id_variable == 0
	    ? ConstantBool(state, true)
	    : Binary(state, spv::OpIEqual, b, EmitSubgroupLocalInvocationId(state), ConstantU32(state, 0));
	wg_x = ConstantU32(state, 0xffffffffu);
	wg_y = wg_x;
	if (InputVariableForKind(state, IR::StageInputKind::WorkgroupId) != 0) {
		wg_x = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
		wg_y = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1);
		const auto grid = Binary(state, spv::OpBitwiseAnd, u,
		                         Binary(state, spv::OpBitwiseOr, u, wg_x, wg_y), ConstantU32(state, 7));
		sample = Binary(state, spv::OpLogicalAnd, b, sample,
		                Binary(state, spv::OpIEqual, b, grid, ConstantU32(state, 0)));
	}
	return sample;
}

void EmitDebugPrintf(EmitterState& state, uint32_t condition, const std::string& format,
                     const std::vector<uint32_t>& args) {
	const auto print_label = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, condition, print_label, merge_label);
	EmitLabel(state, print_label);
	state.builder.RequireExtension("SPV_KHR_non_semantic_info");
	const auto import = state.builder.Import("NonSemantic.DebugPrintf");
	const auto text   = state.builder.AddString(format.c_str());
	std::vector<uint32_t> operands {text};
	operands.insert(operands.end(), args.begin(), args.end());
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeVoid(state), result, import, 1u,
	                          std::span<const uint32_t>(operands));
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
}

// Debug: KYTY_TRACE_SHADERS=<hash>[,<hash>...] prints every buffer load, store and atomic those
// compute shaders run in single-workgroup dispatches (threads 0-15), through SPIR-V debug printf.
bool TraceShaderEnabled(const EmitterState& state) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_TRACE_SHADERS");
		while (value != nullptr && *value != '\0') {
			char*      end  = nullptr;
			const auto hash = std::strtoull(value, &end, 16);
			if (end == value) break;
			result.push_back(hash);
			value = *end == ',' ? end + 1 : end;
		}
		return result;
	}();
	return state.program.stage == ShaderType::Compute &&
	       std::ranges::find(hashes, state.program.shader_hash) != hashes.end();
}

void EmitTraceMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&            state = ctx.state;
	const std::string name {IR::ValueOpcodeName(inst.GetOpcode())};
	const bool        store = name.starts_with("StoreBuffer") || name.starts_with("StoreAddress");
	const bool        load  = name.starts_with("LoadBuffer") || name.starts_with("LoadAddress") ||
	                  name == "ReadConstBuffer" || name == "ReadConst";
	const bool atomic = name.starts_with("BufferAtomic");
	if (!store && !load && !atomic) return;
	const auto u = TypeU32(state), b = TypeBool(state);
	const auto args = inst.NumArgs();
	auto       cond = args != 0 && inst.Arg(args - 1).GetType() == IR::Type::U1
	                      ? ctx.Arg(inst, args - 1)
	                      : ConstantBool(state, true);
	const auto thread = InputVariableForKind(state, IR::StageInputKind::LocalInvocationIndex) != 0
	                        ? EmitLocalInvocationIndex(state)
	                    : state.subgroup_local_invocation_id_variable != 0
	                        ? EmitSubgroupLocalInvocationId(state)
	                        : ConstantU32(state, 0);
	cond = Binary(state, spv::OpLogicalAnd, b, cond,
	              Binary(state, spv::OpULessThan, b, thread, ConstantU32(state, 16)));
	for (uint32_t axis = 0; axis < 2; ++axis) {
		if (InputVariableForKind(state, IR::StageInputKind::WorkgroupId) != 0)
			cond = Binary(state, spv::OpLogicalAnd, b, cond,
			    Binary(state, spv::OpIEqual, b,
			           EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, axis),
			           ConstantU32(state, 0)));
		if (InputVariableForKind(state, IR::StageInputKind::NumWorkgroups) != 0)
			cond = Binary(state, spv::OpLogicalAnd, b, cond,
			    Binary(state, spv::OpIEqual, b,
			           EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, axis),
			           ConstantU32(state, 1)));
	}
	std::vector<uint32_t> values {ConstantU32(state, inst.Flags<IR::MemoryFlags>().pc), thread,
	                              ConstantU32(state, ctx.Memory(inst).resource)};
	for (size_t index = 1; index < 4; ++index)
		values.push_back(index < args && inst.Arg(index).GetType() == IR::Type::U32
		                     ? ctx.Arg(inst, index)
		                     : ConstantU32(state, 0xeeeeeeeeu));
	const auto add_components = [&](uint32_t id, IR::Type type) {
		uint32_t count = 0;
		switch (type) {
			case IR::Type::U32: values.push_back(id); count = 1; break;
			case IR::Type::U32x2: count = 2; break;
			case IR::Type::U32x3: count = 3; break;
			case IR::Type::U32x4: count = 4; break;
			default: break;
		}
		for (uint32_t i = 0; count > 1 && i < count; ++i) {
			const auto component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, u, component, id, i);
			values.push_back(component);
		}
		while (values.size() < 10) values.push_back(ConstantU32(state, 0xeeeeeeeeu));
	};
	if (store)
		add_components(ctx.Arg(inst, args - 2), inst.Arg(args - 2).GetType());
	else
		add_components(ctx.Def(IR::Value(const_cast<IR::Inst*>(&inst))), inst.GetType());
	if (atomic && args > 4) values[7] = ctx.Arg(inst, 4); // v[0] old value, v[1] operand
	// The guest V# words, when the resource is still the raw descriptor.
	const auto* handle = args != 0 ? inst.Arg(0).TryInstruction() : nullptr;
	for (size_t word = 0; word < 4; ++word) {
		values.push_back(handle != nullptr &&
		                         handle->GetOpcode() == IR::ValueOpcode::GetBufferResource &&
		                         word < handle->NumArgs() &&
		                         handle->Arg(word).GetType() == IR::Type::U32
		                     ? ctx.Arg(*handle, word)
		                     : ConstantU32(state, 0xeeeeeeeeu));
	}
	char hash[24];
	std::snprintf(hash, sizeof(hash), "%016llx",
	              static_cast<unsigned long long>(state.program.shader_hash));
	EmitDebugPrintf(state, cond,
	                std::string("TRACE h=") + hash + " " + name +
	                    " pc=0x%x t=%u res=%u idx=%x off=%x so=%x v=%x %x %x %x vs=%x %x %x %x\n",
	                values);
}

namespace {

struct LoopPrint {
	const IR::Block*                     header = nullptr;
	const IR::Block*                     body   = nullptr;
	const std::map<const IR::Inst*, size_t>* ids = nullptr;
};

void EmitLoopPrintf(ValueEmitContext& ctx, const LoopPrint& print, uint32_t count) {
	auto&      state = ctx.state;
	const auto u = TypeU32(state);
	uint32_t   wg_x = 0, wg_y = 0;
	const auto sample = EmitDebugSample(state, wg_x, wg_y);
	std::string format = "LOOPCAP pc=0x%x wg=%u,%u it=%u";
	std::vector<uint32_t> args;
	args.push_back(ConstantU32(state, print.body->start_pc));
	args.push_back(wg_x);
	args.push_back(wg_y);
	args.push_back(count);
	for (const auto& inst: *print.header) {
		if (inst.GetOpcode() != IR::ValueOpcode::Phi || args.size() >= 48u) continue;
		const auto value = ctx.Def(IR::Value(const_cast<IR::Inst*>(&inst)));
		uint32_t   word  = 0;
		switch (inst.GetType()) {
			case IR::Type::U32: word = value; break;
			case IR::Type::F32: word = Unary(state, spv::OpBitcast, u, value); break;
			case IR::Type::U1:
				word = Select(state, u, value, ConstantU32(state, 1), ConstantU32(state, 0));
				break;
			default: continue;
		}
		format += " v" + std::to_string(print.ids->at(&inst)) + "=%x";
		args.push_back(word);
	}
	format += "\n";
	EmitDebugPrintf(state, sample, format, args);
}

void EmitFaultedLoopWatchdog(ValueEmitContext& ctx, const LoopPrint* print) {
	auto&      state = ctx.state;
	const auto u     = TypeU32(state);
	const auto count = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, u, count, state.loop_iteration_variable);
	const auto next = EmitAddU32(state, count, ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpStore, state.loop_iteration_variable, next);
	const auto fault = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, u, fault, state.bda_fault_variable);
	const auto limit = LoopWatchdogLimit();
	const auto stop = limit != 0
	    ? Binary(state, spv::OpUGreaterThan, TypeBool(state), next, ConstantU32(state, limit))
	    : Binary(state, spv::OpLogicalAnd, TypeBool(state),
	             Binary(state, spv::OpINotEqual, TypeBool(state), fault, ConstantU32(state, 0)),
	             Binary(state, spv::OpUGreaterThan, TypeBool(state), next,
	                    ConstantU32(state, FaultedLoopIterationLimit)));
	const auto return_label   = state.builder.AllocateId();
	const auto continue_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, continue_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, stop, return_label, continue_label);
	EmitLabel(state, return_label);
	if (print != nullptr) EmitLoopPrintf(ctx, *print, next);
	state.builder.AddFunction(spv::OpReturn);
	EmitLabel(state, continue_label);
}

void PatchStructuredPhis(ValueEmitContext& ctx, StructuredFunctionState& structured) {
	for (const auto& deferred: structured.deferred_phis) {
		auto& lane = deferred.half == 0 ? ctx : *ctx.other_half;
		for (size_t index = 0; index < deferred.instruction->NumArgs(); index++) {
			const auto* predecessor = deferred.instruction->PhiBlock(index);
			const auto  found       = structured.block_exit_labels.find(predecessor);
			if (found == structured.block_exit_labels.end()) {
				ctx.Fail(*deferred.instruction, "has a predecessor that was not emitted");
			}
			ctx.state.builder.PatchDeferredPhi(
			    deferred.phi, index, lane.Def(deferred.instruction->Arg(index)), found->second);
		}
	}
}

void EmitStructuredFunction(ValueEmitContext& ctx) {
	const auto& program = ctx.state.program;
	StructuredFunctionState structured;
	// The first block of each loop body, entered once per iteration.
	std::map<const IR::Block*, const IR::Block*> loop_bodies;
	if (ctx.state.bda_fault_variable != 0) {
		for (const auto* block: program.blocks) {
			const auto& term = block->terminator;
			if (!term.loop_header) continue;
			const auto* body = term.kind == CFG::TerminatorKind::ConditionalBranch &&
			                           term.true_block == term.merge_block
			                       ? term.false_block
			                       : term.true_block;
			// Splitting a block that is itself a loop header or continue target would move its
			// OpLoopMerge or back edge out of the block the structured rules require.
			if (body != nullptr && body != term.merge_block && body != term.continue_block &&
			    !body->terminator.loop_header)
				loop_bodies.emplace(body, block);
		}
	}
	if (!loop_bodies.empty() && LoopWatchdogLimit() != 0) {
		std::fprintf(stderr, "LOOP-WATCHDOG hash=0x%016llx loops=%zu limit=%u\n",
		             static_cast<unsigned long long>(program.shader_hash), loop_bodies.size(),
		             LoopWatchdogLimit());
	}
	std::map<const IR::Inst*, size_t> ids;
	const bool print_loops = !loop_bodies.empty() && LoopWatchdogLimit() != 0 &&
	                         LoopPrintfHash() == program.shader_hash;
	if (print_loops) {
		// Same numbering as ProgramToString, so the printed values can be matched with this dump.
		size_t next_id = 1;
		for (const auto* block: program.blocks)
			for (const auto& inst: *block) ids.emplace(&inst, next_id++);
		std::fprintf(stderr, "LOOPCAP-IR-BEGIN hash=0x%016llx\n%s\nLOOPCAP-IR-END\n",
		             static_cast<unsigned long long>(program.shader_hash),
		             IR::ProgramToString(program).c_str());
		for (const auto& [body, header]: loop_bodies)
			std::fprintf(stderr, "LOOPCAP-LOOP body=0x%x header_block=0x%x\n", body->start_pc,
			             header->start_pc);
	}
	ctx.state.builder.AddFunction(spv::OpBranch, program.blocks.front()->Definition());
	for (size_t index = 0; index < program.blocks.size(); index++) {
		const auto* block = program.blocks[index];
		std::function<void()> watchdog;
		if (const auto found = loop_bodies.find(block); found != loop_bodies.end()) {
			const LoopPrint print {.header = found->second, .body = block, .ids = &ids};
			watchdog = [&ctx, print, print_loops] {
				EmitFaultedLoopWatchdog(ctx, print_loops ? &print : nullptr);
			};
		}
		EmitBlock(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			EmitStructuredInstruction(lane, structured, inst);
		}, watchdog);
		structured.block_exit_labels.emplace(block, ctx.state.current_label);
		EmitStructuredTerminator(ctx, block);
	}
	PatchStructuredPhis(ctx, structured);
}

void EmitDispatcherFunction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher) {
	auto&       state = ctx.state;
	const auto* entry = state.program.blocks.front();
	state.builder.AddFunction(spv::OpBranch, entry->Definition());
	EmitBlock(ctx, entry, [&](ValueEmitContext& lane, const IR::Inst& inst) {
		EmitDispatcherInstruction(lane, dispatcher, inst);
	});
	const auto initial_pc     = EmitDispatcherNextPc(ctx, dispatcher, entry);
	const auto initial_parent = state.current_label;
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);

	state.current_block = nullptr;
	EmitLabel(state, dispatcher.header_label);
	const auto pc      = state.builder.AllocateId();
	const auto next_pc = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), pc, initial_pc, initial_parent, next_pc,
	                          dispatcher.continue_label);
	const auto done = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), done, pc,
	                          ConstantU32(ctx.state, UINT32_MAX));
	state.builder.AddFunction(spv::OpLoopMerge, dispatcher.merge_label, dispatcher.continue_label,
	                          spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, done, dispatcher.merge_label,
	                          dispatcher.select_label);

	EmitLabel(state, dispatcher.select_label);
	state.builder.AddFunction(spv::OpSelectionMerge, dispatcher.after_switch_label,
	                          spv::SelectionControlMaskNone);
	std::vector<uint32_t> words {spv::OpSwitch, pc, dispatcher.after_switch_label};
	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		words.push_back(state.program.blocks[index]->id);
		words.push_back(state.program.blocks[index]->Definition());
	}
	state.builder.AddFunction(words);
	std::vector<uint32_t> next_pc_words {spv::OpPhi, TypeU32(state), next_pc,
	                                     ConstantU32(state, UINT32_MAX), dispatcher.select_label};

	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		EmitBlock(ctx, state.program.blocks[index],
		          [&](ValueEmitContext& lane, const IR::Inst& inst) {
			          EmitDispatcherInstruction(lane, dispatcher, inst);
		          });
		const auto selected = EmitDispatcherNextPc(ctx, dispatcher, state.program.blocks[index]);
		next_pc_words.push_back(selected);
		next_pc_words.push_back(state.current_label);
		state.builder.AddFunction(spv::OpBranch, dispatcher.after_switch_label);
	}
	state.current_block = nullptr;
	EmitLabel(state, dispatcher.after_switch_label);
	state.builder.AddFunction(next_pc_words);
	state.builder.AddFunction(spv::OpBranch, dispatcher.continue_label);
	EmitLabel(state, dispatcher.continue_label);
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);
	EmitLabel(state, dispatcher.merge_label);
	EmitReturn(ctx);
}

} // namespace

uint32_t TypeId(EmitterState& state, IR::Type type) {
	switch (type) {
		case IR::Type::U1: return TypeBool(state);
		case IR::Type::U8:
		case IR::Type::U16:
		case IR::Type::U32:
		case IR::Type::F16: return TypeU32(state);
		case IR::Type::U64: return TypeU64(state);
		case IR::Type::U32x2: return TypeU32Pair(state);
		case IR::Type::F32: return TypeF32(state);
		case IR::Type::F64: return TypeF64(state);
		case IR::Type::U32x3: return TypeU32Vector(state, 3);
		case IR::Type::U32x4: return TypeU32Vector(state, 4);
		case IR::Type::F32x2: return TypeF32Vector(state, 2);
		default: return 0;
	}
}

uint32_t ValueEmitContext::Def(IR::Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case IR::Type::U1: return ConstantBool(state, value.U1());
			case IR::Type::U8: return ConstantU32(state, value.U8());
			case IR::Type::U16: return ConstantU32(state, value.U16());
			case IR::Type::U32: return ConstantU32(state, value.U32());
			case IR::Type::U64: return ConstantU64(state, value.U64());
			case IR::Type::F16: return ConstantU32(state, value.F16Bits());
			case IR::Type::F32:
				return ConstantF32(state, std::bit_cast<uint32_t>(value.F32Value()));
			default: break;
		}
	}
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr) {
		Fail("direct SPIR-V emitter received a non-value argument");
	}
	if (dispatcher_spills != nullptr && state.current_block != nullptr &&
	    inst->Parent() != state.current_block) {
		if (const auto found = dispatcher_spills->find(inst); found != dispatcher_spills->end()) {
			if (const auto loaded = dispatcher_block_loads.find(inst);
			    loaded != dispatcher_block_loads.end() &&
			    loaded->second.first == state.current_label) {
				return loaded->second.second;
			}
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeId(state, inst->GetType()), id,
			                          found->second);
			dispatcher_block_loads.insert_or_assign(inst, std::pair {state.current_label, id});
			return id;
		}
	}
	return Result(*inst);
}

uint32_t ValueEmitContext::Arg(const IR::Inst& inst, size_t index) {
	return Def(inst.Arg(index));
}

uint32_t ValueEmitContext::HalfArg(const IR::Inst& inst, size_t index, uint32_t lane_half) {
	return lane_half == half ? Arg(inst, index) : other_half->Arg(inst, index);
}

uint32_t ValueEmitContext::Ballot(IR::Value predicate) {
	const auto ballot_type = TypeU32Vector(state, 4);
	const auto scope       = ConstantU32(state, spv::ScopeSubgroup);
	const auto low         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, low, scope,
	                          other_half == nullptr || half == 0 ? Def(predicate)
	                                                             : other_half->Def(predicate));
	if (other_half == nullptr) {
		return low;
	}
	const auto high      = state.builder.AllocateId();
	const auto low_word  = state.builder.AllocateId();
	const auto high_word = state.builder.AllocateId();
	const auto ballot    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, high, scope,
	                          half == 1 ? Def(predicate) : other_half->Def(predicate));
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low_word, low, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high_word, high, 0);
	state.builder.AddFunction(spv::OpCompositeConstruct, ballot_type, ballot, low_word, high_word,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	return ballot;
}

uint32_t ValueEmitContext::FirstLane(uint32_t ballot) {
	if (other_half == nullptr) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallotFindLSB, TypeU32(state), result,
		                          ConstantU32(state, spv::ScopeSubgroup), ballot);
		return result;
	}
	const auto low        = state.builder.AllocateId();
	const auto high       = state.builder.AllocateId();
	const auto low_first  = state.builder.AllocateId();
	const auto high_first = state.builder.AllocateId();
	const auto low_active = state.builder.AllocateId();
	const auto result     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), low_first, GlslStd450(state),
	                          GLSLstd450FindILsb, low);
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), high_first, GlslStd450(state),
	                          GLSLstd450FindILsb, high);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), low_active, low,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, low_active, low_first,
	                          EmitAddU32(state, high_first, ConstantU32(state, 32)));
	return result;
}

uint32_t ValueEmitContext::Shuffle(const IR::Inst& inst, size_t index, uint32_t lane) {
	const auto type  = TypeId(state, inst.Arg(index).GetType());
	const auto scope = ConstantU32(state, spv::ScopeSubgroup);
	const auto low   = state.builder.AllocateId();
	if (other_half == nullptr) {
		state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope, Arg(inst, index),
		                          lane);
		return low;
	}
	const auto physical_lane =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
	const auto high          = state.builder.AllocateId();
	const auto in_high       = state.builder.AllocateId();
	const auto value         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope,
	                          HalfArg(inst, index, 0), physical_lane);
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, high, scope,
	                          HalfArg(inst, index, 1), physical_lane);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), in_high,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 32)),
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, type, value, in_high, high, low);
	return value;
}

uint32_t ValueEmitContext::Result(const IR::Inst& inst) {
	if (const auto id = inst.Definition(half); id != 0) {
		return id;
	}
	const auto id = state.builder.AllocateId();
	inst.SetDefinition(half, id);
	return id;
}

uint32_t ValueEmitContext::Define(const IR::Inst& inst, uint32_t value) {
	if (const auto id = inst.Definition(half); id != 0) {
		if (id != value) {
			state.builder.AddFunction(spv::OpCopyObject, TypeId(state, inst.GetType()),
			                          id, value);
		}
		return id;
	}
	inst.SetDefinition(half, value);
	return value;
}

uint32_t ValueEmitContext::ResourceIndex(IR::Value value, IR::ValueOpcode opcode) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != opcode) {
		Fail("typed resource handle has the wrong producer");
	}
	return inst->Flags<uint32_t>();
}

const IR::Inst* ValueEmitContext::ImageAddress(IR::Value value) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::MakeImageAddress) {
		Fail("typed image address was not constructed by MakeImageAddress");
	}
	return inst;
}

const IR::MemoryInfo& ValueEmitContext::Memory(const IR::Inst& inst) const {
	return state.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
}

const IR::ExportInfo& ValueEmitContext::Export(const IR::Inst& inst) const {
	return state.program.export_info.at(inst.Flags<IR::ExportFlags>().index);
}

[[noreturn]] void ValueEmitContext::Fail(const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage), reason);
	std::abort();
}

[[noreturn]] void ValueEmitContext::Fail(const IR::Inst& inst, const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u opcode=%s reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage),
	     IR::ValueOpcodeName(inst.GetOpcode()), reason);
	std::abort();
}

void EmitProgram(EmitterState& state) {
	const auto&      program = state.program;
	ValueEmitContext ctx(state);
	ValueEmitContext high(state);
	if (state.lane_count == 2) {
		ctx.other_half  = &high;
		high.other_half = &ctx;
		high.half       = 1;
	}
	std::optional<DispatcherFunctionState> dispatcher;
	if (state.program.stage == ShaderType::Pixel && state.program.info.pixel_valid_mask) {
		state.pixel_valid_mask_variable = state.builder.AllocateId();
		state.builder.AddName(state.pixel_valid_mask_variable, "pixel_valid_mask_active");
	}
	for (const auto* block: program.blocks) {
		// Re-emission may use a different host subgroup width and ID allocation order.
		for (const auto& inst: *block) {
			inst.ResetDefinitions();
		}
		block->SetDefinition(state.builder.AllocateId());
	}
	if (state.program.dispatcher_fallback) {
		auto& dispatch = dispatcher.emplace();
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != IR::ValueOpcode::Phi) {
					continue;
				}
				if (SpillPointerType(ctx, inst.GetType()) == 0) {
					ctx.Fail(inst, "cannot be stored by the dispatcher");
					break;
				}
				dispatch.spills[0].emplace(&inst, state.builder.AllocateId());
			}
		}
		const auto mark_cross_block = [&](IR::Value value, const IR::Block* consumer) {
			value                  = value.Resolve();
			const auto* definition = value.TryInstruction();
			if (definition == nullptr || definition->Parent() == consumer ||
			    definition->Parent() == program.blocks.front()) {
				return;
			}
			if (SpillPointerType(ctx, definition->GetType()) == 0) {
				ctx.Fail(*definition, "cannot be stored by the dispatcher");
				return;
			}
			if (!dispatch.spills[0].contains(definition)) {
				dispatch.spills[0].emplace(definition, state.builder.AllocateId());
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					const auto* consumer =
					    inst.GetOpcode() == IR::ValueOpcode::Phi ? inst.PhiBlock(index) : block;
					mark_cross_block(inst.Arg(index), consumer);
				}
			}
		}
		for (size_t index = 0; index < program.blocks.size(); index++) {
			mark_cross_block(program.blocks[index]->condition, program.blocks[index]);
			mark_cross_block(program.blocks[index]->indirect_target, program.blocks[index]);
		}
		dispatch.header_label       = state.builder.AllocateId();
		dispatch.select_label       = state.builder.AllocateId();
		dispatch.after_switch_label = state.builder.AllocateId();
		dispatch.continue_label     = state.builder.AllocateId();
		dispatch.merge_label        = state.builder.AllocateId();
		ctx.dispatcher_spills       = &dispatch.spills[0];
		if (state.lane_count == 2) {
			for (const auto& [inst, id]: dispatch.spills[0]) {
				dispatch.spills[1].emplace(inst, state.builder.AllocateId());
			}
			high.dispatcher_spills = &dispatch.spills[1];
		}
	}
	if (LoopWatchdogLimit() != 0) DefineLoopWatchdogVariables(state);
	DefineGetBdaPointer(state);
	DefineIndirectBufferLoads(ctx);
	DefineIndirectBufferFormats(ctx);
	DefineBvhIntersect(state);
	if (program.info.uses_swizzle) {
		ctx.scratch_u32_variable = state.builder.AllocateId();
		if (state.lane_count == 2) {
			high.scratch_u32_variable = state.builder.AllocateId();
		}
	}
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state),
	                          state.mesh_guest_func != 0 ? state.mesh_guest_func : state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.entry_label);
	if (state.program.info.function_lds) {
		state.builder.AddFunction(
		    spv::OpVariable,
		    TypeU32ArrayPointer(state, spv::StorageClassFunction, LdsDwordCount(state)),
		    state.lds_variable, spv::StorageClassFunction);
	}
	if (state.program.info.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddFunction(
			    spv::OpVariable,
			    TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.scratch_dwords),
			    state.scratch_variable[half], spv::StorageClassFunction);
		}
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.pixel_valid_mask_variable, spv::StorageClassFunction);
	}
	for (uint32_t half = 0; half < state.lane_count; half++) {
		auto& lane = half == 0 ? ctx : high;
		if (state.program.dispatcher_fallback) {
			for (const auto* block: program.blocks) {
				for (const auto& inst: *block) {
					if (const auto found = dispatcher->spills[half].find(&inst);
					    found != dispatcher->spills[half].end()) {
						state.builder.AddFunction(spv::OpVariable,
						                          SpillPointerType(lane, inst.GetType()),
						                          found->second, spv::StorageClassFunction);
					}
				}
			}
		}
		if (lane.scratch_u32_variable != 0) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          lane.scratch_u32_variable, spv::StorageClassFunction);
		}
	}
	if (state.gds_variable != 0) {
		state.gds_length = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), state.gds_length,
		                          state.gds_variable, 0);
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable,
		                          ConstantU32(state, 1));
	}
	if (state.bda_fault_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.bda_fault_variable, ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.loop_iteration_variable, ConstantU32(state, 0));
	}
	if (state.lds_storage_class == spv::StorageClassStorageBuffer && state.lds_variable != 0) {
		const auto group_x = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
		const auto group_y = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1);
		const auto group_z = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 2);
		const auto count_x = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 0);
		const auto count_y = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 1);
		const auto row = EmitAddU32(state, group_y, EmitBinaryU32(state, spv::OpIMul, group_z, count_y));
		const auto index = EmitAddU32(state, group_x, EmitBinaryU32(state, spv::OpIMul, row, count_x));
		state.lds_base_dwords = EmitBinaryU32(state, spv::OpIMul, index,
		                                     ConstantU32(state, LdsDwordCount(state)));
	}
	EmitMemoryOffsets(state);
	if (program.blocks.empty()) {
		EmitReturn(ctx);
	} else if (state.program.dispatcher_fallback) {
		EmitDispatcherFunction(ctx, *dispatcher);
	} else {
		EmitStructuredFunction(ctx);
	}
	state.builder.AddFunction(spv::OpFunctionEnd);
	state.current_block = nullptr;
	if (state.program.stage == ShaderType::Mesh) {
		EmitMeshEntryPoint(state);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
