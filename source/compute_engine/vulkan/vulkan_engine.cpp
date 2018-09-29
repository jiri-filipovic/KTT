#include "vulkan_engine.h"
#include "utility/logger.h"
#include "utility/timer.h"

namespace ktt
{

#ifdef PLATFORM_VULKAN

VulkanEngine::VulkanEngine(const DeviceIndex deviceIndex, const uint32_t queueCount) :
    deviceIndex(deviceIndex),
    queueCount(queueCount),
    compilerOptions(std::string("")),
    globalSizeType(GlobalSizeType::Vulkan),
    globalSizeCorrection(false),
    kernelCacheFlag(true),
    kernelCacheCapacity(10),
    persistentBufferFlag(true),
    nextEventId(0)
{
    Logger::getLogger().log(LoggingLevel::Debug, "Initializing Vulkan instance");
    instance = std::make_unique<VulkanInstance>("KTT");
}

KernelResult VulkanEngine::runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
    const std::vector<OutputDescriptor>& outputDescriptors)
{
    EventId eventId = runKernelAsync(kernelData, argumentPointers, getDefaultQueue());
    KernelResult result = getKernelResult(eventId, outputDescriptors);
    return result;
}

EventId VulkanEngine::runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

KernelResult VulkanEngine::getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::getKernelOverhead(const EventId id) const
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void VulkanEngine::setGlobalSizeType(const GlobalSizeType type)
{
    globalSizeType = type;
}

void VulkanEngine::setAutomaticGlobalSizeCorrection(const bool flag)
{
    globalSizeCorrection = flag;
}

void VulkanEngine::setKernelCacheUsage(const bool flag)
{
    if (!flag)
    {
        clearKernelCache();
    }
    kernelCacheFlag = flag;
}

void VulkanEngine::setKernelCacheCapacity(const size_t capacity)
{
    kernelCacheCapacity = capacity;
}

void VulkanEngine::clearKernelCache()
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

QueueId VulkanEngine::getDefaultQueue() const
{
    return 0;
}

std::vector<QueueId> VulkanEngine::getAllQueues() const
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::synchronizeQueue(const QueueId queue)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::synchronizeDevice()
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::clearEvents()
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::uploadArgument(KernelArgument& kernelArgument)
{
    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return 0;
    }

    EventId eventId = uploadArgumentAsync(kernelArgument, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId VulkanEngine::uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes)
{
    EventId eventId = updateArgumentAsync(id, data, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId VulkanEngine::updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const
{
    EventId eventId = downloadArgumentAsync(id, destination, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId VulkanEngine::downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

KernelArgument VulkanEngine::downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes)
{
    EventId eventId = copyArgumentAsync(destination, source, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId VulkanEngine::copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::persistArgument(KernelArgument& kernelArgument, const bool flag)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::getArgumentOperationDuration(const EventId id) const
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::resizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::setPersistentBufferUsage(const bool flag)
{
    persistentBufferFlag = flag;
}

void VulkanEngine::clearBuffer(const ArgumentId id)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::clearBuffers()
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::clearBuffers(const ArgumentAccessType accessType)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::printComputeAPIInfo(std::ostream& outputTarget) const
{
    throw std::runtime_error("Vulkan API is not yet supported");

    /* outputTarget << "Platform 0: " << "Vulkan" << std::endl;
    auto devices = getVulkanDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        outputTarget << "Device " << i << ": " << devices.at(i).getName() << std::endl;
    }
    outputTarget << std::endl; */
}

std::vector<PlatformInfo> VulkanEngine::getPlatformInfo() const
{
    throw std::runtime_error("Vulkan API is not yet supported");

    /* PlatformInfo info(0, "Vulkan");
    info.setVendor("N/A");
    info.setVersion("1.0.0");
    info.setExtensions("N/A");
    return std::vector<PlatformInfo>{info}; */
}

std::vector<DeviceInfo> VulkanEngine::getDeviceInfo(const PlatformIndex platform) const
{
    throw std::runtime_error("Vulkan API is not yet supported");

    /* std::vector<DeviceInfo> result;
    auto devices = getVulkanDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        result.push_back(getVulkanDeviceInfo(i));
    }

    return result; */
}

DeviceInfo VulkanEngine::getCurrentDeviceInfo() const
{
    throw std::runtime_error("Vulkan API is not yet supported");

    /* return getVulkanDeviceInfo(deviceIndex); */
}

/* DeviceInfo VulkanEngine::getVulkanDeviceInfo(const size_t deviceIndex) const
{
    auto devices = getVulkanDevices();
    DeviceInfo result(deviceIndex, devices.at(deviceIndex).getName());

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(devices.at(deviceIndex).getPhysicalDevice(), &deviceProperties);
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(devices.at(deviceIndex).getPhysicalDevice(), &deviceMemoryProperties);

    result.setExtensions("N/A");
    result.setVendor(std::to_string(deviceProperties.vendorID));
    result.setDeviceType(getDeviceType(deviceProperties.deviceType));

    result.setGlobalMemorySize(deviceMemoryProperties.memoryHeaps[0].size);
    result.setLocalMemorySize(deviceProperties.limits.maxComputeSharedMemorySize);
    result.setMaxWorkGroupSize(deviceProperties.limits.maxComputeWorkGroupSize[0] * deviceProperties.limits.maxComputeWorkGroupSize[1]
        * deviceProperties.limits.maxComputeWorkGroupSize[2]);
    result.setMaxConstantBufferSize(0); // to do: find this information for Vulkan API
    result.setMaxComputeUnits(1); // to do: find this information for Vulkan API

    return result;
}

std::vector<VulkanPhysicalDevice> VulkanEngine::getVulkanDevices() const
{
    uint32_t deviceCount;
    checkVulkanError(vkEnumeratePhysicalDevices(instance.getInstance(), &deviceCount, nullptr), "vkEnumeratePhysicalDevices");

    std::vector<VkPhysicalDevice> vulkanDevices(deviceCount);
    checkVulkanError(vkEnumeratePhysicalDevices(instance.getInstance(), &deviceCount, vulkanDevices.data()), "vkEnumeratePhysicalDevices");

    std::vector<VulkanPhysicalDevice> devices;
    for (const auto vulkanDevice : vulkanDevices)
    {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(vulkanDevice, &deviceProperties);
        devices.push_back(VulkanPhysicalDevice(vulkanDevice, deviceProperties.deviceName));
    }

    return devices;
} */

DeviceType VulkanEngine::getDeviceType(const VkPhysicalDeviceType deviceType)
{
    switch (deviceType)
    {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        return DeviceType::Custom;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return DeviceType::GPU;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return DeviceType::GPU;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return DeviceType::GPU;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return DeviceType::CPU;
    default:
        return DeviceType::Custom;
    }
}

#else

VulkanEngine::VulkanEngine(const DeviceIndex, const uint32_t)
{
    throw std::runtime_error("Support for Vulkan API is not included in this version of KTT framework");
}

KernelResult VulkanEngine::runKernel(const KernelRuntimeData&, const std::vector<KernelArgument*>&, const std::vector<OutputDescriptor>&)
{
    throw std::runtime_error("");
}

EventId VulkanEngine::runKernelAsync(const KernelRuntimeData&, const std::vector<KernelArgument*>&, const QueueId)
{
    throw std::runtime_error("");
}

KernelResult VulkanEngine::getKernelResult(const EventId, const std::vector<OutputDescriptor>&) const
{
    throw std::runtime_error("");
}

uint64_t VulkanEngine::getKernelOverhead(const EventId) const
{
    throw std::runtime_error("");
}

void VulkanEngine::setCompilerOptions(const std::string&)
{
    throw std::runtime_error("");
}

void VulkanEngine::setGlobalSizeType(const GlobalSizeType)
{
    throw std::runtime_error("");
}

void VulkanEngine::setAutomaticGlobalSizeCorrection(const bool)
{
    throw std::runtime_error("");
}

void VulkanEngine::setKernelCacheUsage(const bool)
{
    throw std::runtime_error("");
}

void VulkanEngine::setKernelCacheCapacity(const size_t)
{
    throw std::runtime_error("");
}

void VulkanEngine::clearKernelCache()
{
    throw std::runtime_error("");
}

QueueId VulkanEngine::getDefaultQueue() const
{
    throw std::runtime_error("");
}

std::vector<QueueId> VulkanEngine::getAllQueues() const
{
    throw std::runtime_error("");
}

void VulkanEngine::synchronizeQueue(const QueueId)
{
    throw std::runtime_error("");
}

void VulkanEngine::synchronizeDevice()
{
    throw std::runtime_error("");
}

void VulkanEngine::clearEvents()
{
    throw std::runtime_error("");
}

uint64_t VulkanEngine::uploadArgument(KernelArgument&)
{
    throw std::runtime_error("");
}

EventId VulkanEngine::uploadArgumentAsync(KernelArgument&, const QueueId)
{
    throw std::runtime_error("");
}

uint64_t VulkanEngine::updateArgument(const ArgumentId, const void*, const size_t)
{
    throw std::runtime_error("");
}

EventId VulkanEngine::updateArgumentAsync(const ArgumentId, const void*, const size_t, const QueueId)
{
    throw std::runtime_error("");
}

uint64_t VulkanEngine::downloadArgument(const ArgumentId, void*, const size_t) const
{
    throw std::runtime_error("");
}

EventId VulkanEngine::downloadArgumentAsync(const ArgumentId, void*, const size_t, const QueueId) const
{
    throw std::runtime_error("");
}

KernelArgument VulkanEngine::downloadArgumentObject(const ArgumentId, uint64_t*) const
{
    throw std::runtime_error("");
}

uint64_t VulkanEngine::copyArgument(const ArgumentId, const ArgumentId, const size_t)
{
    throw std::runtime_error("");
}

EventId VulkanEngine::copyArgumentAsync(const ArgumentId, const ArgumentId, const size_t, const QueueId)
{
    throw std::runtime_error("");
}

uint64_t VulkanEngine::persistArgument(KernelArgument&, const bool)
{
    throw std::runtime_error("");
}

uint64_t VulkanEngine::getArgumentOperationDuration(const EventId) const
{
    throw std::runtime_error("");
}

void VulkanEngine::resizeArgument(const ArgumentId, const size_t, const bool)
{
    throw std::runtime_error("");
}

void VulkanEngine::setPersistentBufferUsage(const bool)
{
    throw std::runtime_error("");
}

void VulkanEngine::clearBuffer(const ArgumentId)
{
    throw std::runtime_error("");
}

void VulkanEngine::clearBuffers()
{
    throw std::runtime_error("");
}

void VulkanEngine::clearBuffers(const ArgumentAccessType)
{
    throw std::runtime_error("");
}

void VulkanEngine::printComputeAPIInfo(std::ostream&) const
{
    throw std::runtime_error("");
}

std::vector<PlatformInfo> VulkanEngine::getPlatformInfo() const
{
    throw std::runtime_error("");
}

std::vector<DeviceInfo> VulkanEngine::getDeviceInfo(const PlatformIndex) const
{
    throw std::runtime_error("");
}

DeviceInfo VulkanEngine::getCurrentDeviceInfo() const
{
    throw std::runtime_error("");
}

#endif // PLATFORM_VULKAN

} // namespace ktt
