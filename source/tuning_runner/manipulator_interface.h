#pragma once

#include <cstddef>
#include <vector>

#include "ktt_type_aliases.h"
#include "enum/argument_data_type.h"
#include "enum/argument_location.h"
#include "enum/argument_memory_type.h"

namespace ktt
{

class ManipulatorInterface
{
public:
    // Destructor
    virtual ~ManipulatorInterface() = default;

    // Kernel run methods
    virtual void runKernel(const size_t kernelId) = 0;
    virtual void runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize) = 0;

    // Configuration retrieval methods
    virtual DimensionVector getCurrentGlobalSize(const size_t kernelId) const = 0;
    virtual DimensionVector getCurrentLocalSize(const size_t kernelId) const = 0;
    virtual std::vector<ParameterValue> getCurrentConfiguration() const = 0;

    // Argument update and synchronization methods
    virtual void updateArgumentScalar(const size_t argumentId, const void* argumentData) = 0;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData, const ArgumentLocation& argumentLocation) = 0;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData, const ArgumentLocation& argumentLocation,
        const size_t numberOfElements) = 0;
    virtual void synchronizeArgumentVector(const size_t argumentId, const bool downloadToHost) = 0;

    // Kernel argument handling methods
    virtual void changeKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds) = 0;
    virtual void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond) = 0;
};

} // namespace ktt
