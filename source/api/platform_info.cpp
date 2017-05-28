#include "platform_info.h"

namespace ktt
{

PlatformInfo::PlatformInfo(const size_t id, const std::string& name) :
    id(id),
    name(name)
{}

size_t PlatformInfo::getId() const
{
    return id;
}

std::string PlatformInfo::getName() const
{
    return name;
}

std::string PlatformInfo::getVendor() const
{
    return vendor;
}

std::string PlatformInfo::getVersion() const
{
    return version;
}

std::string PlatformInfo::getExtensions() const
{
    return extensions;
}

void PlatformInfo::setVendor(const std::string& vendor)
{
    this->vendor = vendor;
}

void PlatformInfo::setVersion(const std::string& version)
{
    this->version = version;
}

void PlatformInfo::setExtensions(const std::string& extensions)
{
    this->extensions = extensions;
}

std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo)
{
    outputTarget << "Printing detailed info for platform with id: " << platformInfo.id << std::endl;
    outputTarget << "Name: " << platformInfo.name << std::endl;
    outputTarget << "Vendor: " << platformInfo.vendor << std::endl;
    outputTarget << "Compute API version: " << platformInfo.version << std::endl;
    outputTarget << "Extensions: " << platformInfo.extensions << std::endl;
    return outputTarget;
}

} // namespace ktt
