#pragma once

#include <map>

#include "manipulator_interface.h"
#include "compute_api_driver/compute_api_driver.h"
#include "dto/kernel_runtime_data.h"
#include "kernel/kernel_configuration.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

class ManipulatorInterfaceImplementation : public ManipulatorInterface
{
public:
    // Constructor
    explicit ManipulatorInterfaceImplementation(ComputeApiDriver* computeApiDriver);

    // Inherited methods
    void runKernel(const size_t kernelId) override;
    void runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize) override;
    DimensionVector getCurrentGlobalSize(const size_t kernelId) const override;
    DimensionVector getCurrentLocalSize(const size_t kernelId) const override;
    std::vector<ParameterValue> getCurrentConfiguration() const override;
    void updateArgumentScalar(const size_t argumentId, const void* argumentData) override;
    void updateArgumentVector(const size_t argumentId, const void* argumentData) override;
    void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements) override;
    void setAutomaticArgumentUpdate(const bool flag) override;
    void setArgumentSynchronization(const bool flag, const ArgumentMemoryType& argumentMemoryType) override;
    void updateKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds) override;
    void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond) override;

    // Core methods
    void addKernel(const size_t id, const KernelRuntimeData& kernelRuntimeData);
    void setConfiguration(const KernelConfiguration& kernelConfiguration);
    void setKernelArguments(const std::vector<KernelArgument>& kernelArguments);
    KernelRunResult getCurrentResult() const;
    void clearData();

private:
    // Attributes
    ComputeApiDriver* computeApiDriver;
    KernelRunResult currentResult;
    KernelConfiguration currentConfiguration;
    std::map<size_t, KernelRuntimeData> kernelDataMap;
    std::vector<KernelArgument> kernelArguments;
    bool automaticArgumentUpdate;
    bool synchronizeWriteArguments;
    bool synchronizeReadWriteArguments;

    // Helper methods
    void updateArgument(const size_t argumentId, const void* argumentData, const size_t numberOfElements,
        const ArgumentUploadType& argumentUploadType, const bool overrideNumberOfElements);
    std::vector<const KernelArgument*> getArgumentPointers(const std::vector<size_t>& argumentIndices);
};

} // namespace ktt
