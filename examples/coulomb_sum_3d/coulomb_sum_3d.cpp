#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#define USE_CUDA 0
#define USE_PROFILING 0

#if USE_CUDA == 0
    #if defined(_MSC_VER)
        #define KTT_KERNEL_FILE "../examples/coulomb_sum_3d/coulomb_sum_3d_kernel.cl"
        #define KTT_REFERENCE_KERNEL_FILE "../examples/coulomb_sum_3d/coulomb_sum_3d_reference_kernel.cl"
    #else
        #define KTT_KERNEL_FILE "../../examples/coulomb_sum_3d/coulomb_sum_3d_kernel.cl"
        #define KTT_REFERENCE_KERNEL_FILE "../../examples/coulomb_sum_3d/coulomb_sum_3d_reference_kernel.cl"
    #endif
#else
    #if defined(_MSC_VER)
        #define KTT_KERNEL_FILE "../examples/coulomb_sum_3d/coulomb_sum_3d_kernel.cu"
        #define KTT_REFERENCE_KERNEL_FILE "../examples/coulomb_sum_3d/coulomb_sum_3d_reference_kernel.cu"
    #else
        #define KTT_KERNEL_FILE "../../examples/coulomb_sum_3d/coulomb_sum_3d_kernel.cu"
        #define KTT_REFERENCE_KERNEL_FILE "../../examples/coulomb_sum_3d/coulomb_sum_3d_reference_kernel.cu"
    #endif
#endif

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = KTT_KERNEL_FILE;
    std::string referenceKernelFile = KTT_REFERENCE_KERNEL_FILE;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));
            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);
                if (argc >= 5)
                {
                    referenceKernelFile = std::string(argv[4]);
                }
            }
        }
    }

    // Declare kernel parameters
    const int gridSize = 256;
#if USE_PROFILING == 0
    const int atoms = 4000;
#else
    const int atoms = 400; /* faster execution of slowly profiled kernel */
#endif
    const ktt::DimensionVector ndRangeDimensions(gridSize, gridSize, gridSize);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

    // Declare data variables
    float gridSpacing;
    std::vector<float> atomInfoX(atoms);
    std::vector<float> atomInfoY(atoms);
    std::vector<float> atomInfoZ(atoms);
    std::vector<float> atomInfoW(atoms);
    std::vector<float> atomInfo(4 * atoms);
    std::vector<float> energyGrid(gridSize * gridSize * gridSize, 0.0f);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 20.0f);
    gridSpacing = 0.5f; // in Angstroms

    for (int i = 0; i < atoms; i++)
    {
        atomInfoX.at(i) = distribution(engine);
        atomInfoY.at(i) = distribution(engine);
        atomInfoZ.at(i) = distribution(engine);
        atomInfoW.at(i) = distribution(engine) / 40.0f;
        atomInfo.at(i*4) = atomInfoX.at(i);
        atomInfo.at(i*4 + 1) = atomInfoY.at(i);
        atomInfo.at(i*4 + 2) = atomInfoZ.at(i);
        atomInfo.at(i*4 + 3) = atomInfoW.at(i);
    }

#if USE_CUDA == 0
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeAPI::OpenCL);
    tuner.setCompilerOptions("-cl-fast-relaxed-math");
#else
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);
    tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.setCompilerOptions("-use_fast_math");
  #if USE_PROFILING == 1
    printf("Executing with profiling switched ON.\n");
    tuner.setKernelProfiling(true);
  #endif
#endif
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);
    //tuner.setLoggingLevel(ktt::LoggingLevel::Debug);

    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "directCoulombSum", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "directCoulombSumReference", ndRangeDimensions,
        referenceWorkGroupDimensions);

    ktt::ArgumentId aiId = tuner.addArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aixId = tuner.addArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aiyId = tuner.addArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aizId = tuner.addArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aiwId = tuner.addArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId aId = tuner.addArgumentScalar(atoms);
    ktt::ArgumentId gsId = tuner.addArgumentScalar(gridSpacing);
    ktt::ArgumentId gridId = tuner.addArgumentVector(energyGrid, ktt::ArgumentAccessType::WriteOnly);

    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {16, 32});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", {1, 2, 4, 8});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Z", {1});
    tuner.addParameter(kernelId, "Z_ITERATIONS", {1, 2, 4, 8, 16, 32});
    tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Z, "Z_ITERATIONS", ktt::ModifierAction::Divide);
    tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", {0, 1, 2, 4, 8, 16, 32});
#if USE_CUDA == 0
    tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {0, 1});
#else
    tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {0}); // not implemented in CUDA
#endif
    tuner.addParameter(kernelId, "USE_SOA", {0, 1});
#if USE_CUDA == 0
    tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2 , 4, 8, 16});
#else
    tuner.addParameter(kernelId, "VECTOR_SIZE", {1}); // not implemented in CUDA
#endif

    auto lt = [](const std::vector<size_t>& vector) {return vector.at(0) < vector.at(1);};
    tuner.addConstraint(kernelId, {"INNER_UNROLL_FACTOR", "Z_ITERATIONS"}, lt);
    auto vec = [](const std::vector<size_t>& vector) {return vector.at(0) || vector.at(1) == 1;};
    tuner.addConstraint(kernelId, {"USE_SOA", "VECTOR_SIZE"}, vec);
    auto par = [](const std::vector<size_t>& vector) {return vector.at(0) * vector.at(1) >= 64;};
    tuner.addConstraint(kernelId, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, par);

    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aiId, aixId, aiyId, aizId, aiwId, aId, gsId, gridId});
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{aiId, aId, gsId, gridId});

#if USE_PROFILING == 0
    //TODO: this is temporal hack, there should be composition of zeroizing and coulomb kernel, otherwise, multiple profiling runs corrupt results
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{gridId});
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);
#endif

    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "coulomb_sum_3d_output.csv", ktt::PrintFormat::CSV);

    return 0;
}