#include "kernel.h"

namespace ktt
{

Kernel::Kernel(const size_t id, const std::string& source, const std::string& name, const DimensionVector& globalSize,
    const DimensionVector& localSize):
    id(id),
    source(source),
    name(name),
    globalSize(globalSize),
    localSize(localSize),
    searchMethod(SearchMethod::FullSearch),
    referenceKernelValid(false),
    referenceClassValid(false)
{}

void Kernel::addParameter(const KernelParameter& parameter)
{
    if (parameterExists(parameter.getName()))
    {
        throw std::runtime_error(std::string("Parameter with given name already exists: " + parameter.getName()));
    }
    parameters.push_back(parameter);
}

void Kernel::addConstraint(const KernelConstraint& constraint)
{
    auto parameterNames = constraint.getParameterNames();

    for (const auto& parameterName : parameterNames)
    {
        if (!parameterExists(parameterName))
        {
            throw std::runtime_error(std::string("Constraint parameter with given name does not exist: " + parameterName));
        }
    }
    constraints.push_back(constraint);
}

void Kernel::setArguments(const std::vector<size_t>& argumentIndices)
{
    this->argumentIndices = argumentIndices;
}

void Kernel::setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    if (searchMethod == SearchMethod::RandomSearch && searchArguments.size() < 1
        || searchMethod == SearchMethod::Annealing && searchArguments.size() < 2
        || searchMethod == SearchMethod::PSO && searchArguments.size() < 5)
    {
        throw std::runtime_error(std::string("Insufficient number of arguments given for specified search method: "
            + getSearchMethodName(searchMethod)));
    }
    
    this->searchArguments = searchArguments;
    this->searchMethod = searchMethod;
}

void Kernel::setReferenceKernel(const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
    const std::vector<size_t>& resultArgumentIds)
{
    for (const auto argumentId : resultArgumentIds)
    {
        if (!argumentIndexExists(argumentId))
        {
            throw std::runtime_error(std::string("Following argument id is not associated with this kernel: " + std::to_string(argumentId)));
        }
    }
    this->referenceKernelId = referenceKernelId;
    this->referenceKernelConfiguration = referenceKernelConfiguration;
    this->resultArgumentIds = resultArgumentIds;
    referenceKernelValid = true;
}

void Kernel::setReferenceClass(std::unique_ptr<ReferenceClass> referenceClass, const size_t resultArgumentId)
{
    if (!argumentIndexExists(resultArgumentId))
    {
        throw std::runtime_error(std::string("Following argument id is not associated with this kernel: " + std::to_string(resultArgumentId)));
    }
    this->referenceClass = std::move(referenceClass);
    this->resultArgumentId = resultArgumentId;
    referenceClassValid = true;
}

size_t Kernel::getId() const
{
    return id;
}

std::string Kernel::getSource() const
{
    return source;
}

std::string Kernel::getName() const
{
    return name;
}

DimensionVector Kernel::getGlobalSize() const
{
    return globalSize;
}

DimensionVector Kernel::getLocalSize() const
{
    return localSize;
}

std::vector<KernelParameter> Kernel::getParameters() const
{
    return parameters;
}

std::vector<KernelConstraint> Kernel::getConstraints() const
{
    return constraints;
}

size_t Kernel::getArgumentCount() const
{
    return argumentIndices.size();
}

std::vector<size_t> Kernel::getArgumentIndices() const
{
    return argumentIndices;
}

SearchMethod Kernel::getSearchMethod() const
{
    return searchMethod;
}

std::vector<double> Kernel::getSearchArguments() const
{
    return searchArguments;
}

bool Kernel::hasReferenceKernel() const
{
    return referenceKernelValid;
}

size_t Kernel::getReferenceKernelId() const
{
    return referenceKernelId;
}

std::vector<ParameterValue> Kernel::getReferenceKernelConfiguration() const
{
    return referenceKernelConfiguration;
}

std::vector<size_t> Kernel::getResultArgumentIds() const
{
    return resultArgumentIds;
}

bool Kernel::hasReferenceClass() const
{
    return referenceClassValid;
}

const ReferenceClass* Kernel::getReferenceClass() const
{
    return referenceClass.get();
}

size_t Kernel::getResultArgumentIdForClass() const
{
    return resultArgumentId;
}

bool Kernel::argumentIndexExists(const size_t argumentIndex) const
{
    for (const auto index : argumentIndices)
    {
        if (index == argumentIndex)
        {
            return true;
        }
    }
    return false;
}

bool Kernel::parameterExists(const std::string& parameterName) const
{
    for (const auto& currentParameter : parameters)
    {
        if (currentParameter.getName() == parameterName)
        {
            return true;
        }
    }
    return false;
}

std::string Kernel::getSearchMethodName(const SearchMethod& searchMethod) const
{
    switch (searchMethod)
    {
    case SearchMethod::FullSearch:
        return std::string("FullSearch");
    case SearchMethod::RandomSearch:
        return std::string("RandomSearch");
    case SearchMethod::PSO:
        return std::string("PSO");
    case SearchMethod::Annealing:
        return std::string("Annealing");
    default:
        return std::string("Unknown search method");
    }
}

} // namespace ktt
