#pragma once

#if defined(_MSC_VER) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251)
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _MSC_VER

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
#include "api/device_info.h"
#include "api/platform_info.h"

// Reference class interface
#include "api/reference_class.h"

// Tuning manipulator interface
#include "api/tuning_manipulator.h"

// Support for 16-bit floating point data type
#include "half.hpp"
using half_float::half;

namespace ktt
{

class TunerCore; // Forward declaration of TunerCore class

class KTT_API Tuner
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
    template <typename T> size_t addArgument(const size_t elementsCount)
    {
        ArgumentDataType dataType = getMatchingArgumentDataType<T>();
        return addArgument(elementsCount, dataType);
    }
    void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition);

    // Kernel tuning methods
    void tuneKernel(const size_t kernelId);

    // Result printing methods
    void setPrintingTimeUnit(const TimeUnit& timeUnit);
    void setInvalidResultPrinting(const bool flag);
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const;

    // Result validation methods
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);
    void setValidationRange(const size_t argumentId, const size_t validationRange);

    // Compute API methods
    void setCompilerOptions(const std::string& options);
    void printComputeApiInfo(std::ostream& outputTarget) const;
    std::vector<PlatformInfo> getPlatformInfo() const;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const;
    DeviceInfo getCurrentDeviceInfo() const;

    // Utility methods
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);

private:
    // Attributes
    std::unique_ptr<TunerCore> tunerCore;

    // Helper methods
    size_t addArgument(const void* vectorData, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
        const ArgumentMemoryType& argumentMemoryType);
    size_t addArgument(const void* scalarData, const ArgumentDataType& argumentDataType);
    size_t addArgument(const size_t elementsCount, const ArgumentDataType& argumentDataType);

    template <typename T> ArgumentDataType getMatchingArgumentDataType() const
    {
        if (!std::is_trivially_copyable<T>())
        {
            std::cerr << "Unsupported argument data type" << std::endl;
            throw std::runtime_error("Unsupported argument data type");
        }

        if (sizeof(T) == 1 && std::is_unsigned<T>())
        {
            return ArgumentDataType::UnsignedChar;
        }
        else if (sizeof(T) == 1)
        {
            return ArgumentDataType::Char;
        }
        else if (typeid(T) == typeid(half))
        {
            return ArgumentDataType::Half;
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
