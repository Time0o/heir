#include "lib/Dialect/TensorExt/Transforms/ShiftScheme.h"

#include <cassert>
#include <cstdint>

#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

#define DEBUG_TYPE "shift-scheme"

namespace mlir {
namespace heir {
namespace tensor_ext {

SmallVector<int64_t> ShiftStrategy::defaultShiftOrder(int64_t n, ShiftKind shiftKind) {
  SmallVector<int64_t> result;

  int64_t maxLog2 = APInt(64, n).getActiveBits();
  if (isPowerOfTwo(n)) maxLog2 -= 1;

  for (int64_t i = 0; i < maxLog2; i++) {
    int64_t shift;
    switch (shiftKind) {
    case ShiftKind::LEFT:
      shift = 1 << i;
      break;
    case ShiftKind::RIGHT:
      shift = -(1 << i);
      break;
    }
    result.push_back(shift);
  }

  return result;
}

int64_t ShiftStrategy::getVirtualShift(const CtSlot& source,
                                       const CtSlot& target) const {
  int64_t sourceIndex = source.ct * ciphertextSize + source.slot;
  int64_t targetIndex = target.ct * ciphertextSize + target.slot;

  // Convert a source->target index mapping to a canonical left-shift (or
  // right-shift) amount for a given ciphertext size.
  // Example: 1 -> 13 with a 64-size ciphertext should produce a rotation of 52
  // Example: 13 -> 1 with a 64-size ciphertext should produce a rotation of 12
  int64_t shift = (targetIndex - sourceIndex) % virtualCiphertextSize;

  if (shiftKind == ShiftKind::LEFT) {
    shift = -shift;  // Account for leftward rotations
  }
  if (shift < 0) {
    shift += virtualCiphertextSize;
  }
  return shift;
}

void ShiftStrategy::evaluate(const Mapping& mapping) {
  // First compute the virtual shifts needed for each source slot
  SmallVector<SourceShift> sourceShifts;
  sourceShifts.reserve(mapping.size());
  for (const MappingEntry& entry : mapping) {
    int64_t shift = getVirtualShift(entry.source, entry.target);
    sourceShifts.push_back({entry.source, shift});
  }

  // Compute the corresponding table of positions after each rotation,
  // akin to the table in Figure 3 of the Vos-Vos-Erkin paper, including the
  // first column of values that have not yet been rotated.
  rounds.reserve(shiftOrder.size() + 1);
  ShiftRound initialRound;
  for (const SourceShift& ss : sourceShifts) {
    initialRound.positions[ss] = ss.source;
    initialRound.rotationAmount = 0;
  }
  rounds.push_back(initialRound);

  for (auto rotationAmount : shiftOrder) {
    auto lastRoundPositions = rounds.back().positions;
    DenseMap<SourceShift, CtSlot> currentRoundPosns;

    for (const SourceShift& key : sourceShifts) {
      assert(lastRoundPositions.contains(key) &&
             "Expected to find source in last round positions");
      CtSlot currentPos = lastRoundPositions[key];
      int64_t currentVirtualSlot =
          currentPos.ct * ciphertextSize + currentPos.slot;

      int64_t shift = key.shift;
      CtSlot nextPosition = currentPos;
      if (std::abs(rotationAmount) & std::abs(shift)) {
        currentVirtualSlot =
            (currentVirtualSlot - rotationAmount + virtualCiphertextSize) %
            virtualCiphertextSize;
        nextPosition = CtSlot{currentVirtualSlot / ciphertextSize,
                              currentVirtualSlot % ciphertextSize};
      }
      currentRoundPosns[key] = nextPosition;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After rotation " << rotationAmount << ":\n";
      for (const auto& [ss, pos] : currentRoundPosns) {
        llvm::dbgs() << "  (" << ss.source.ct << "," << ss.source.slot << ")["
                     << ss.shift << "] -> (" << pos.ct << "," << pos.slot << ")"
                     << "\n";
      }
    });

    rounds.push_back({currentRoundPosns, rotationAmount});
  }
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
