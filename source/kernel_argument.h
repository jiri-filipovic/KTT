#pragma once

#include <vector>
#include "kernel_argument_type.h"

namespace ktt
{

template <typename T> class KernelArgument
{
public:
    explicit KernelArgument(const size_t index, const std::vector<T>& data, const KernelArgumentType& kernelArgumentType):
        index(index),
        data(data),
        kernelArgumentType(kernelArgumentType)
    {}
    
    size_t getIndex()
    {
        return index;
    }

    std::vector<T> getData()
    {
        return data;
    }

    KernelArgumentType getKernelArgumentType()
    {
        return kernelArgumentType;
    }

private:
    size_t index;
    std::vector<T> data;
    KernelArgumentType kernelArgumentType;
};

} // namespace ktt
