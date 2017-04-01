#pragma once

#include <memory>
#include <vector>

#include "kernel_argument.h"

namespace ktt
{

class ArgumentManager
{
public:
    // Constructor
    ArgumentManager():
        argumentCount(0)
    {}

    // Core methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType,
        const ArgumentQuantity& argumentQuantity)
    {
        arguments.emplace_back(KernelArgument(data, argumentMemoryType, argumentQuantity));
        return argumentCount++;
    }

    template <typename T> void updateArgument(const size_t id, const std::vector<T>& data, const ArgumentQuantity& argumentQuantity)
    {
        arguments.at(id).updateData(data, argumentQuantity);
    }

    // Getters
    size_t getArgumentCount() const
    {
        return argumentCount;
    }

    const KernelArgument getArgument(const size_t id) const
    {
        if (id >= argumentCount)
        {
            throw std::runtime_error(std::string("Invalid argument id: " + std::to_string(id)));
        }
        return arguments.at(id);
    }

private:
    // Attributes
    size_t argumentCount;
    std::vector<KernelArgument> arguments;
};

} // namespace ktt
