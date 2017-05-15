#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <vector>

// Type aliases and enums relevant to usage of API methods
#include "ktt_type_aliases.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_type.h"
#include "enum/argument_print_condition.h"
#include "enum/compute_api.h"
#include "enum/dimension.h"
#include "enum/print_format.h"
#include "enum/time_unit.h"
#include "enum/search_method.h"
#include "enum/thread_modifier_action.h"
#include "enum/thread_modifier_type.h"
#include "enum/validation_method.h"

// Information about platforms and devices
#include "dto/device_info.h"
#include "dto/platform_info.h"

// Reference class interface
#include "customization/reference_class.h"

// Tuning manipulator interface
#include "customization/tuning_manipulator.h"

namespace ktt
{

class TunerCore; // Forward declaration of TunerCore class

class Tuner
{
public:
    // Constructor and destructor
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex);
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi);
    ~Tuner();

    // Basic kernel handling methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds);
    void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values);
    void addParameter(const std::vector<size_t>& kernelIds, const std::string& name, const std::vector<size_t>& values);

    // Advanced kernel handling methods
    void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values,
        const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addParameter(const std::vector<size_t>& kernelIds, const std::string& name, const std::vector<size_t>& values,
        const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void addConstraint(const std::vector<size_t>& kernelIds, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);
    void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator);

    // Argument handling methods
    size_t addArgument(const void* vectorData, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
        const ArgumentMemoryType& argumentMemoryType);
    size_t addArgument(const void* scalarData, const ArgumentDataType& argumentDataType);
    void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition);

    // Kernel tuning methods
    void tuneKernel(const size_t kernelId);

    // Result printing methods
    void setPrintingTimeUnit(const TimeUnit& timeUnit);
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const;

    // Result validation methods
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);

    // Compute API methods
    void setCompilerOptions(const std::string& options);
    void printComputeApiInfo(std::ostream& outputTarget) const;
    std::vector<PlatformInfo> getPlatformInfo() const;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const;

    // Utility methods
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);

    // Convenience argument addition methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(data.data(), data.size(), dataType, argumentMemoryType);
    }

    template <typename T> size_t addArgument(const T& value)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(&value, dataType);
    }

private:
    // Attributes
    std::unique_ptr<TunerCore> tunerCore;

    // Helper methods
    template <typename T> ArgumentDataType getMatchingArgumentDataType() const
    {
        if (sizeof(T) == 1 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedChar;
        }
        else if (sizeof(T) == 1)
        {
            return ArgumentDataType::Char;
        }
        else if (sizeof(T) == 2 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedShort;
        }
        else if (sizeof(T) == 2)
        {
            return ArgumentDataType::Short;
        }
        else if (typeid(T) == typeid(float))
        {
            return ArgumentDataType::Float;
        }
        else if (sizeof(T) == 4 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedInt;
        }
        else if (sizeof(T) == 4)
        {
            return ArgumentDataType::Int;
        }
        else if (typeid(T) == typeid(double))
        {
            return ArgumentDataType::Double;
        }
        else if (sizeof(T) == 8 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedLong;
        }
        else if (sizeof(T) == 8)
        {
            return ArgumentDataType::Long;
        }
        else
        {
            std::cerr << "Unsupported argument data type" << std::endl;
            throw std::runtime_error("Unsupported argument data type");
        }
    }
};

} // namespace ktt
