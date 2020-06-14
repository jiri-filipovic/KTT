#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"
//#include "stdafx.h"

#define FAKE_KERNEL_FILE "fake_kernel.cl"

#define REDUCED_SET_1070 0
#define ROCM_WORKAROUND 0

enum eProblem{
    BICGK,
    CONV,
    COLUMB_3D,
    GEMM,
    HOTSPOT,
    NBODY,
    REDUCTION,
    SORT,
    MTRAN,
    GEMM_REDUCED,
    GEMM_BATCHED,
    FOURIER_REC_32,
    FOURIER_REC_64,
    FOURIER_REC_128,

    PROBLEMS_ENUM_END
};

enum eSearchMethod{
    RANDOM,
    ANNEALING,
    MCMC_SIMPLE,
    MCMC_SENSIT,
    MCMC_CONVEX,
    MCMC_FULL,

    SEARCHMETHODS_ENUM_END
};

void printProblemName(eProblem problem);
std::string getPerfFileName(eProblem problem);
void printSearcherName(eSearchMethod searchMethod);
void createTuningSpace(ktt::Tuner &tuner, ktt::KernelId &kernelId, eProblem problem);
void configureSearcher(ktt::Tuner &tuner, eSearchMethod searcher, eProblem problem);

unsigned long cus = std::numeric_limits<unsigned long>::max();

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
    eProblem problem;
    eSearchMethod searcher;
    size_t steps;
    size_t experiments=10;
    std::string kernelFile = FAKE_KERNEL_FILE;
    std::string dir = "";
	
	// Nezarat :for testing the configuration possible values
	//$problem 0 1500 1000 .. / data - selection / $dir 90
	argv[1] = "0";
	argv[2] = "0";
	argv[3] = "1500";
	argv[4] = "1000";
	argv[5] = "GPU-1070";


    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " problem_num searcher_num tuning_steps experiments_num [data_dir]" << std::endl;
        std::cout << std::endl;
        std::cout << "Problems implemented: " << std::endl;
        for (int i = 0; i < PROBLEMS_ENUM_END; i++) {
            std::cout << i << " ";
            printProblemName((eProblem)i);
        }
        std::cout << std::endl;
        std::cout << "Searchers implemented: " << std::endl;
        for (int i = 0; i < SEARCHMETHODS_ENUM_END; i++) {
            std::cout << i << " ";
            printSearcherName((eSearchMethod)i);
        }
        exit(0);
    }
    else {
        problem = (eProblem)std::stoul(std::string(argv[1]));
        searcher = (eSearchMethod)std::stoul(std::string(argv[2]));
        steps = std::stoul(std::string(argv[3]));
        experiments = std::stoul(std::string(argv[4]));
        if (argc == 6) dir=std::string(argv[5]);
        std::cout << "Selected problem: ";
        printProblemName(problem);
        std::cout << "Selected searcher: ";
        printSearcherName(searcher);
        std::cout << "Data directory: " << dir << "\n";
        std::cout << "Performing " << experiments << " experiment(s), "
            << "each with " << steps << " tuning steps" << std::endl;
    }

    // Declare kernel fake parameters
    const int gridSize = 1024;
    const ktt::DimensionVector ndRangeDimensions(gridSize, gridSize, gridSize);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

    // Declare fake data variables
    std::vector<float> fakeInput(1);
    std::vector<float> fakeOutput(1);

    // Perform experiments
    double statsTimeSum = 0.0;
    double statsTimeMax = 0;
    double statsTimeMin = std::numeric_limits<double>::max();
    double statsTimes[10];

    // Create tuner
    ktt::Tuner tuner(0, 0);
    tuner.setPrintingTimeUnit(ktt::TimeUnit::Nanoseconds);

    // Create and configure fake kernel
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "fake_kernel", ndRangeDimensions, workGroupDimensions);
    ktt::ArgumentId inId = tuner.addArgumentVector(fakeInput, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId outId = tuner.addArgumentVector(fakeOutput, ktt::ArgumentAccessType::WriteOnly);
    tuner.persistArgument(inId, true);
    tuner.persistArgument(outId, true);
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{inId, outId});

    // Create tuning space according to selected problem
    createTuningSpace(tuner, kernelId, problem);

    // Set search method
    configureSearcher(tuner, searcher, problem);

    // Execute experiments
    for (size_t ex = 0; ex < experiments; ex++)
    {
        std::cout << "Executing experiment " << ex+1 << "/" << experiments << std::endl;

        // Dry-tune kernel
        tuner.dryTuneKernel(kernelId, dir+"/"+getPerfFileName(problem), steps);
        tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);

        // Compute statistics
        ktt::ComputationResult bestConf = tuner.getBestComputationResult(kernelId);
        double duration = (double)bestConf.getDuration()/1000.0;
        std::cout << "Best time: " << duration << "us " << std::endl;
        statsTimes[ex] = duration;
        statsTimeSum += duration;
        statsTimeMax = std::max(statsTimeMax, duration);
        statsTimeMin = std::min(statsTimeMin, duration);

        // Clear search results
        tuner.clearKernelData(kernelId, false);
    }

    // Compute standard deviation and print statistics
    double statsStdDev = 0.0;
    double mean = statsTimeSum / (double)experiments;
    for (size_t ex = 0; ex < experiments; ex++) 
        statsStdDev += (mean-statsTimes[ex])*(mean-statsTimes[ex]);
    statsStdDev = sqrt(statsStdDev / (double)experiments);
    std::cout << "Experiment statistics:" << std::endl
        << "average: " << mean << std::endl
        << "stdDev:  " << statsStdDev << std::endl
        << "minimum: " << statsTimeMin << std::endl
        << "maximum: " << statsTimeMax << std::endl
        << std::endl;

    return 0;
}

void printProblemName(eProblem problem) {
    switch(problem) {
    case BICGK:
        std::cout << "BiCGK stab (PolyBench)";
        break;
    case CONV:
        std::cout << "2D convolution (CLTune)";
        break;
    case COLUMB_3D:
        std::cout << "3D Coulomb Sum";
        break;
    case GEMM:
        std::cout << "GEMM (CLTune)";
        break;
    case HOTSPOT:
        std::cout << "Hotspot (Rodinia)";
        break;
    case NBODY:
        std::cout << "n-body (NVIDIA)";
        break;
    case REDUCTION:
        std::cout << "reduction";
        break;
    case SORT:
        std::cout << "sort (Rodinia)";
        break;
    case MTRAN:
        std::cout << "matrix transposition";
        break;
    case GEMM_REDUCED:
        std::cout << "GEMM (reduced space from CLBlast)";
        break;
    case GEMM_BATCHED:
        std::cout << "GEMM batched (16x16)";
        break;
    case FOURIER_REC_32:
        std::cout << "3D Fourier Reconstruction (32x32)";
        break;
    case FOURIER_REC_64:
        std::cout << "3D Fourier Reconstruction (64x64)";
        break;
    case FOURIER_REC_128:
        std::cout << "3D Fourier Reconstruction (128x128)";
        break;
    }
    std::cout << std::endl;
}

std::string getPerfFileName(eProblem problem) {
    switch(problem) {
    case BICGK:
        return "bicg_output.csv";
        break;
    case CONV:
        return "conv_output.csv";
        break;
    case COLUMB_3D:
        return "coulomb_sum_3d_output.csv";
        break;
    case GEMM:
    case GEMM_REDUCED:
        return "gemm_output.csv";
        break;
    case HOTSPOT:
        return "hotspot_output.csv";
        break;
    case NBODY:
        return "nbody_output.csv";
        break;
    case REDUCTION:
        return "reduction_output.csv";
        break;
    case SORT:
        return "sort_output.csv";
        break;
    case MTRAN:
        return "mtran_output.csv";
        break;
    case GEMM_BATCHED:
        return "gemm_batch_output.csv";
        break;
    case FOURIER_REC_32:
        return "fourier_32_results.csv";
        break;
    case FOURIER_REC_64:
        return "fourier_64_results.csv";
        break;
    case FOURIER_REC_128:
        return "fourier_128_results.csv";
        break;
    }
}

void printSearcherName(eSearchMethod searchMethod) {
    switch(searchMethod){
    case RANDOM:
        std::cout << "Random search";
        break;
    case ANNEALING:
        std::cout << "Simmulated annealing";
        break;
    case MCMC_SIMPLE:
        std::cout << "Markov-chain Monte Carlo (not parametrized)";
        break;
    case MCMC_SENSIT:
        std::cout << "Markov-chain Monte Carlo (parameters sensitivity)";
        break;
    case MCMC_CONVEX:
        std::cout << "Markov-chain Monte Carlo (parameters convexness)";
        break;
    case MCMC_FULL:
        std::cout << "Markov-chain Monte Carlo (fully parametrized)";
        break;
    }
    std::cout << std::endl;
}

// Helper function to perform an integer division + ceiling (round-up)
size_t CeilDiv(const size_t a, const size_t b) {
    return (a + b - 1) / b;
}

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b) {
    return ((a / b) * b == a) ? true : false;
}

void createTuningSpace(ktt::Tuner &tuner, ktt::KernelId &kernelId, eProblem problem){
    switch(problem){
    case BICGK:{
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "FUSED", std::vector<size_t>{ /*0, 1,*/ 2 });
        tuner.addParameter(kernelId, "BICG_BATCH", std::vector<size_t>{ 1, 2, 4, 8 });
        tuner.addParameter(kernelId, "USE_SHARED_MATRIX", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "USE_SHARED_VECTOR_1", std::vector<size_t>{ /*0,*/ 1 });
        tuner.addParameter(kernelId, "USE_SHARED_VECTOR_2", std::vector<size_t>{ /*0,*/ 1 });
        tuner.addParameter(kernelId, "USE_SHARED_REDUCTION_1", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "USE_SHARED_REDUCTION_2", std::vector<size_t>{ /*0,*/ 1 });
        tuner.addParameter(kernelId, "ATOMICS", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "UNROLL_BICG_STEP", std::vector<size_t>{ /*0,*/ 1 });
        tuner.addParameter(kernelId, "ROWS_PROCESSED", std::vector<size_t>{ 128, 256, 512, 1024 });
        tuner.addParameter(kernelId, "TILE", std::vector<size_t>{ 16, 32, 64 });
#else
        tuner.addParameter(kernelId, "FUSED", std::vector<size_t>{ 0, 1, 2 });
        tuner.addParameter(kernelId, "BICG_BATCH", std::vector<size_t>{ 1, 2, 4, 8 });
        tuner.addParameter(kernelId, "USE_SHARED_MATRIX", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "USE_SHARED_VECTOR_1", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "USE_SHARED_VECTOR_2", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "USE_SHARED_REDUCTION_1", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "USE_SHARED_REDUCTION_2", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "ATOMICS", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "UNROLL_BICG_STEP", std::vector<size_t>{ 0, 1 });
        tuner.addParameter(kernelId, "ROWS_PROCESSED", std::vector<size_t>{ 128, 256, 512, 1024 });
        tuner.addParameter(kernelId, "TILE", std::vector<size_t>{ 16, 32, 64 });
#endif
        auto fused = [](std::vector<size_t> vector) {return vector.at(0) == 2 || ((vector.at(0) == 0 || vector.at(0) == 1) && vector.at(1) == 4 && vector.at(2) == 1 && vector.at(3) == 1 && vector.at(4) == 1 && vector.at(5) == 1 && vector.at(6) == 1 && vector.at(7) == 1 && vector.at(8) == 1 && vector.at(9) == 512 && vector.at(10) == 32); };
        tuner.addConstraint(kernelId, std::vector<std::string>{"FUSED", "BICG_BATCH", "USE_SHARED_MATRIX", "USE_SHARED_VECTOR_1", "USE_SHARED_VECTOR_2", "USE_SHARED_REDUCTION_1", "USE_SHARED_REDUCTION_2", "ATOMICS", "UNROLL_BICG_STEP", "ROWS_PROCESSED", "TILE"}, fused);
        auto maxWgSize = [](std::vector<size_t> vector) {return vector.at(0) * vector.at(0) / vector.at(1) <= 1024; };
        tuner.addConstraint(kernelId, std::vector<std::string>{"TILE", "BICG_BATCH"}, maxWgSize);}
        break;
    case CONV:{
        #define HFS (3)        // Half filter size
        #define FS (HFS+HFS+1) // Filter size
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "TBX", {/*8, 16,*/ 32/*, 64*/});
        tuner.addParameter(kernelId, "TBY", {/*8, 16,*/ 32/*, 64*/});
        tuner.addParameter(kernelId, "LOCAL", {0, 1, 2});
        tuner.addParameter(kernelId, "WPTX", {1, 2, 4, 8});
        tuner.addParameter(kernelId, "WPTY", {1, 2, 4, 8});
        tuner.addParameter(kernelId, "VECTOR", {/*1,*/ 2/*, 4*/});
        tuner.addParameter(kernelId, "UNROLL_FACTOR", {1, FS});
        tuner.addParameter(kernelId, "PADDING", {/*0,*/ 1});
        std::vector<size_t> integers{8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74};
        tuner.addParameter(kernelId, "TBX_XL", integers);
        tuner.addParameter(kernelId, "TBY_XL", integers);
#else
        tuner.addParameter(kernelId, "TBX", {8, 16, 32, 64});
        tuner.addParameter(kernelId, "TBY", {8, 16, 32, 64});
        tuner.addParameter(kernelId, "LOCAL", {0, 1, 2});
        tuner.addParameter(kernelId, "WPTX", {1, 2, 4, 8});
        tuner.addParameter(kernelId, "WPTY", {1, 2, 4, 8});
        tuner.addParameter(kernelId, "VECTOR", {1, 2, 4});
        tuner.addParameter(kernelId, "UNROLL_FACTOR", {1, FS});
        tuner.addParameter(kernelId, "PADDING", {0, 1});
        std::vector<size_t> integers{8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74};
        tuner.addParameter(kernelId, "TBX_XL", integers);
        tuner.addParameter(kernelId, "TBY_XL", integers);
#endif
        auto HaloThreads = [](const std::vector<size_t>& v) {
            if (v[0] == 2) {return (v[1] == v[2] + CeilDiv(2 * HFS, v[3]));}
            else           {return (v[1] == v[2]);}
        };
        tuner.addConstraint(kernelId, {"LOCAL", "TBX_XL", "TBX", "WPTX"}, HaloThreads);
        tuner.addConstraint(kernelId, {"LOCAL", "TBY_XL", "TBY", "WPTY"}, HaloThreads);
        auto VectorConstraint = [](const std::vector<size_t>& v) {
            if (v[0] == 2) {return IsMultiple(v[2], v[1]) && IsMultiple(2 * HFS, v[1]);}
            else           {return IsMultiple(v[2], v[1]);}
        };
        tuner.addConstraint(kernelId, {"LOCAL", "VECTOR", "WPTX"}, VectorConstraint);
        //auto WorkPerThreadConstraint = [] (std::vector<size_t> v) { return (v[0]*v[1] < 32); };
        //tuner.AddConstraint(id, WorkPerThreadConstraint, {"WPTX", "WPTY"});
        auto PaddingConstraint = [](const std::vector<size_t>& v) {return (v[1] == 0 || v[0] != 0);};
        tuner.addConstraint(kernelId, {"LOCAL", "PADDING"}, PaddingConstraint);}
        break;
    case COLUMB_3D:{
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {/*16,*/ 32});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", {/*1, 2,*/ 4/*, 8*/});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Z", {1});
        tuner.addParameter(kernelId, "Z_ITERATIONS", {1, 2, 4, 8, 16, 32});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Z, "Z_ITERATIONS", ktt::ModifierAction::Divide);
    #if ROCM_WORKAROUND > 0
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", {0});
    #else
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", {0, 1, 2, 4, 8, 16, 32});
    #endif
        tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {/*0,*/ 1});
        tuner.addParameter(kernelId, "USE_SOA", {0, 1});
    #if ROCM_WORKAROUND > 0
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2 , 4, 8});
    #else
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2 , 4, 8, 16});
    #endif
#else
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {16, 32});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", {1, 2, 4, 8});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Z", {1});
        tuner.addParameter(kernelId, "Z_ITERATIONS", {1, 2, 4, 8, 16, 32});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Z, "Z_ITERATIONS", ktt::ModifierAction::Divide);
    #if ROCM_WORKAROUND > 0
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", {0});
    #else
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", {0, 1, 2, 4, 8, 16, 32});
    #endif
        tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {0, 1});
        tuner.addParameter(kernelId, "USE_SOA", {0, 1});
    #if ROCM_WORKAROUND > 0
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2 , 4, 8});
    #else
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2 , 4, 8, 16});
    #endif
#endif

        auto lt = [](const std::vector<size_t>& vector) {return vector.at(0) < vector.at(1);};
        tuner.addConstraint(kernelId, {"INNER_UNROLL_FACTOR", "Z_ITERATIONS"}, lt);
        auto vec = [](const std::vector<size_t>& vector) {return vector.at(0) || vector.at(1) == 1;};
        tuner.addConstraint(kernelId, {"USE_SOA", "VECTOR_SIZE"}, vec);
        auto par = [](const std::vector<size_t>& vector) {return vector.at(0) * vector.at(1) >= 64;};
        tuner.addConstraint(kernelId, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, par);}
        break;
    case GEMM:{
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "MWG", {16, 32, 64, 128});
        tuner.addParameter(kernelId, "NWG", {16, 32, 64, 128});
        tuner.addParameter(kernelId, "KWG", {/*16,*/ 32});
        tuner.addParameter(kernelId, "MDIMC", {8, 16, 32});
        tuner.addParameter(kernelId, "NDIMC", {8, 16, 32});
        tuner.addParameter(kernelId, "MDIMA", {/*8,*/ 16/*, 32*/});
        tuner.addParameter(kernelId, "NDIMB", {/*8,*/ 16/*, 32*/});
        tuner.addParameter(kernelId, "KWI", {/*2,*/ 8});
        tuner.addParameter(kernelId, "VWM", {1, 2, 4, 8});
        tuner.addParameter(kernelId, "VWN", {1, 2, 4, 8});
        tuner.addParameter(kernelId, "STRM", {0, 1});
        tuner.addParameter(kernelId, "STRN", {/*0, */1});
        tuner.addParameter(kernelId, "SA", {0, 1});
        tuner.addParameter(kernelId, "SB", {0, 1});
        tuner.addParameter(kernelId, "PRECISION", {32});
#else
        tuner.addParameter(kernelId, "MWG", {16, 32, 64, 128});
        tuner.addParameter(kernelId, "NWG", {16, 32, 64, 128});
        tuner.addParameter(kernelId, "KWG", {16, 32});
        tuner.addParameter(kernelId, "MDIMC", {8, 16, 32});
        tuner.addParameter(kernelId, "NDIMC", {8, 16, 32});
        tuner.addParameter(kernelId, "MDIMA", {8, 16, 32});
        tuner.addParameter(kernelId, "NDIMB", {8, 16, 32});
        tuner.addParameter(kernelId, "KWI", {2, 8});
        tuner.addParameter(kernelId, "VWM", {1, 2, 4, 8});
        tuner.addParameter(kernelId, "VWN", {1, 2, 4, 8});
        tuner.addParameter(kernelId, "STRM", {0, 1});
        tuner.addParameter(kernelId, "STRN", {0, 1});
        tuner.addParameter(kernelId, "SA", {0, 1});
        tuner.addParameter(kernelId, "SB", {0, 1});
        tuner.addParameter(kernelId, "PRECISION", {32});
#endif
        auto MultipleOfX = [](const std::vector<size_t>& v) {return IsMultiple(v[0], v[1]);};
        auto MultipleOfXMulY = [](const std::vector<size_t>& v) {return IsMultiple(v[0], v[1] * v[2]);};
        auto MultipleOfXMulYDivZ = [](const std::vector<size_t>& v) {return IsMultiple(v[0], (v[1] * v[2]) / v[3]);};
        tuner.addConstraint(kernelId, {"KWG", "KWI"}, MultipleOfX);
        tuner.addConstraint(kernelId, {"MWG", "MDIMC", "VWM"}, MultipleOfXMulY);
        tuner.addConstraint(kernelId, {"NWG", "NDIMC", "VWN"}, MultipleOfXMulY);
        tuner.addConstraint(kernelId, {"MWG", "MDIMA", "VWM"}, MultipleOfXMulY);
        tuner.addConstraint(kernelId, {"NWG", "NDIMB", "VWN"}, MultipleOfXMulY);
        tuner.addConstraint(kernelId, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, MultipleOfXMulYDivZ);
        tuner.addConstraint(kernelId, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, MultipleOfXMulYDivZ);}
        break;
    case HOTSPOT:{
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "BLOCK_SIZE_ROWS", {8, 16, 32, 64});
        tuner.addParameter(kernelId, "BLOCK_SIZE_COLS", {/*8, 16,*/ 32/*, 64*/});
        tuner.addParameter(kernelId, "PYRAMID_HEIGHT", {1, 2, 4, 8}); //, 2, 4});
        tuner.addParameter(kernelId, "WORK_GROUP_Y", {4, 8, 16, 32, 64});
        tuner.addParameter(kernelId, "LOCAL_MEMORY", {0, 1});
        tuner.addParameter(kernelId, "LOOP_UNROLL", {/*0,*/ 1});
#else
        tuner.addParameter(kernelId, "BLOCK_SIZE_ROWS", {8, 16, 32, 64});
        tuner.addParameter(kernelId, "BLOCK_SIZE_COLS", {8, 16, 32, 64});
        tuner.addParameter(kernelId, "PYRAMID_HEIGHT", {1, 2, 4, 8}); //, 2, 4});
        tuner.addParameter(kernelId, "WORK_GROUP_Y", {4, 8, 16, 32, 64});
        tuner.addParameter(kernelId, "LOCAL_MEMORY", {0, 1});
        tuner.addParameter(kernelId, "LOOP_UNROLL", {0,1});
#endif
        auto enoughToCompute = [](const std::vector<size_t>& vector) {return vector.at(0)/(vector.at(2)*2) > 1 && vector.at(1)/(vector.at(2)*2) > 1;};
        tuner.addConstraint(kernelId, {"BLOCK_SIZE_COLS", "WORK_GROUP_Y", "PYRAMID_HEIGHT"}, enoughToCompute);
        auto workGroupSmaller = [](const std::vector<size_t>& vector) {return vector.at(0)<=vector.at(1);};
        auto workGroupDividable = [](const std::vector<size_t>& vector) {return vector.at(1)%vector.at(0) == 0;};
        tuner.addConstraint(kernelId, {"WORK_GROUP_Y", "BLOCK_SIZE_ROWS"}, workGroupSmaller);
        tuner.addConstraint(kernelId, {"WORK_GROUP_Y", "BLOCK_SIZE_ROWS"}, workGroupDividable);}
        break;
    case NBODY:{
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {64, 128, 256, 512});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "OUTER_UNROLL_FACTOR", {1, 2, 4, 8});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR", ktt::ModifierAction::Divide);
    #if ROCM_WORKAROUND > 0
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR1", {0/*, 1, 2, 4, 8, 16, 32*/});
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR2", {0/*, 1, 2, 4, 8, 16, 32*/});
    #else
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR1", {0, 1, 2, 4, 8, 16, 32});
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR2", {0, 1, 2, 4, 8, 16, 32});
    #endif
        tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {/*0,*/ 1});
        tuner.addParameter(kernelId, "USE_SOA", {/*0,*/ 1});
        tuner.addParameter(kernelId, "LOCAL_MEM", {0, 1});
    #if ROCM_WORKAROUND > 0
        tuner.addParameter(kernelId, "VECTOR_TYPE", {1, 2, 4, 8});
    #else
        tuner.addParameter(kernelId, "VECTOR_TYPE", {1, 2, 4, 8, 16});
    #endif
#else
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {64, 128, 256, 512});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "OUTER_UNROLL_FACTOR", {1, 2, 4, 8});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "OUTER_UNROLL_FACTOR", ktt::ModifierAction::Divide);
    #if ROCM_WORKAROUND > 0
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR1", {0});
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR2", {0});
    #else
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR1", {0, 1, 2, 4, 8, 16, 32});
        tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR2", {0, 1, 2, 4, 8, 16, 32});
    #endif
        tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", {0, 1});
        tuner.addParameter(kernelId, "USE_SOA", {0, 1});
        tuner.addParameter(kernelId, "LOCAL_MEM", {0, 1});
    #if ROCM_WORKAROUND > 0
        tuner.addParameter(kernelId, "VECTOR_TYPE", {1, 2, 4, 8});
    #else
        tuner.addParameter(kernelId, "VECTOR_TYPE", {1, 2, 4, 8, 16});
    #endif
#endif
        auto lteq = [](const std::vector<size_t>& vector) {return vector.at(0) <= vector.at(1);};
        tuner.addConstraint(kernelId, {"INNER_UNROLL_FACTOR2", "OUTER_UNROLL_FACTOR"}, lteq);
        auto lteq256 = [](const std::vector<size_t>& vector) {return vector.at(0) * vector.at(1) <= 256;};
        tuner.addConstraint(kernelId, {"INNER_UNROLL_FACTOR1", "INNER_UNROLL_FACTOR2"}, lteq256);
        auto vectorizedSoA = [](const std::vector<size_t>& vector) {return (vector.at(0) == 1 && vector.at(1) == 0) || (vector.at(1) == 1);};
        tuner.addConstraint(kernelId, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"}, vectorizedSoA);}
        break;
    case REDUCTION:{
        if (cus == std::numeric_limits<unsigned long>::max()) {
            std::cout << "Enter a number of compute units of the device ";
            std::cin >> cus;
        }
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {32, 64, 128, 256, 512});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "UNBOUNDED_WG", {0, 1});
        tuner.addParameter(kernelId, "WG_NUM", {0, cus, cus * 2, cus * 4, cus * 8, cus * 16});
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2, 4, 8, 16});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "VECTOR_SIZE", ktt::ModifierAction::Divide);
        tuner.addParameter(kernelId, "USE_ATOMICS", {0, 1});
#else
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", {32, 64, 128, 256, 512});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "UNBOUNDED_WG", {0, 1});
        tuner.addParameter(kernelId, "WG_NUM", {0, cus, cus * 2, cus * 4, cus * 8, cus * 16});
        tuner.addParameter(kernelId, "VECTOR_SIZE", {1, 2, 4, 8, 16});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "VECTOR_SIZE", ktt::ModifierAction::Divide);
        tuner.addParameter(kernelId, "USE_ATOMICS", {0, 1});
#endif
        auto persistConstraint = [](const std::vector<size_t>& v) {return (v[0] && v[1] == 0) || (!v[0] && v[1] > 0);};
        tuner.addConstraint(kernelId, {"UNBOUNDED_WG", "WG_NUM"}, persistConstraint);
        auto persistentAtomic = [](const std::vector<size_t>& v) {return (v[0] == 1) || (v[0] == 0 && v[1] == 1);};
        tuner.addConstraint(kernelId, {"UNBOUNDED_WG", "USE_ATOMICS"}, persistentAtomic);
        auto unboundedWG = [](const std::vector<size_t>& v) {return (!v[0] || v[1] >= 32);};
        tuner.addConstraint(kernelId, {"UNBOUNDED_WG", "WORK_GROUP_SIZE_X"}, unboundedWG);}
        break;
    case SORT:{
        tuner.addParameter(kernelId, "FPVECTNUM", {4, 8, 16});
        tuner.addParameter(kernelId, "LOCAL_SIZE", {128, 256, 512});
        tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "LOCAL_SIZE", ktt::ModifierAction::Multiply);
        tuner.addParameter(kernelId, "GLOBAL_SIZE", {512, 1024, 2048, 4096, 8192, 16384, 32768});
        auto workGroupConstraint = [](const std::vector<size_t>& vector) {return vector.at(0) != 128 || vector.at(1) != 32768;};
        tuner.addConstraint(kernelId, {"LOCAL_SIZE", "GLOBAL_SIZE"}, workGroupConstraint);}
        break;
    case MTRAN:{
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "LOCAL_MEM", { 0, 1 });
        tuner.addParameter(kernelId, "VECTOR_TYPE", { 1, 2, 4, 8 });
        tuner.addParameter(kernelId, "CR", { /*0,*/ 1 });
        tuner.addParameter(kernelId, "PREFETCH", { /*0,*/ 1/*, 2*/ });
        tuner.addParameter(kernelId, "PADD_LOCAL", { 0, 1 });
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", { 1, 2, 4, 8, 16, 32, 64 });
        tuner.addParameter(kernelId, "TILE_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
        tuner.addParameter(kernelId, "TILE_SIZE_Y", { 1, 2, 4, 8, 16, 32, 64 });
        tuner.addParameter(kernelId, "DIAGONAL_MAP", {0, 1});
#else
        tuner.addParameter(kernelId, "LOCAL_MEM", { 0, 1 });
        tuner.addParameter(kernelId, "VECTOR_TYPE", { 1, 2, 4, 8 });
        tuner.addParameter(kernelId, "CR", { 0, 1 });
        tuner.addParameter(kernelId, "PREFETCH", { 0, 1, 2 });
        tuner.addParameter(kernelId, "PADD_LOCAL", { 0, 1 });
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
        tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", { 1, 2, 4, 8, 16, 32, 64 });
        tuner.addParameter(kernelId, "TILE_SIZE_X", { 1, 2, 4, 8, 16, 32, 64 });
        tuner.addParameter(kernelId, "TILE_SIZE_Y", { 1, 2, 4, 8, 16, 32, 64 });
        tuner.addParameter(kernelId, "DIAGONAL_MAP", {0, 1});
#endif
        auto xConstraint = [] (std::vector<size_t> v) { return (v[0] == v[1]); };
        auto yConstraint = [] (std::vector<size_t> v) { return (v[1] <= v[0]); };
        auto tConstraint = [] (std::vector<size_t> v) { return (!v[0] || (v[1] <= v[2]*v[3])); };
        auto pConstraint = [] (std::vector<size_t> v) { return (v[0] || !v[1]); };
        auto vConstraint = [] (std::vector<size_t> v) { return (v[0]*v[1] <= 64);  };
        auto vlConstraint = [] (std::vector<size_t> v) { return (!v[0] || v[1] == 1);  };
        auto minparConstraint = [] (std::vector<size_t> v) {return (v[0] * v[1] >= 32);}; tuner.addConstraint(kernelId, { "TILE_SIZE_X", "WORK_GROUP_SIZE_X" }, xConstraint);
        tuner.addConstraint(kernelId, { "TILE_SIZE_Y", "WORK_GROUP_SIZE_Y" }, yConstraint);
        tuner.addConstraint(kernelId, { "LOCAL_MEM", "TILE_SIZE_Y", "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y" }, tConstraint);
        tuner.addConstraint(kernelId, { "LOCAL_MEM", "PADD_LOCAL" }, pConstraint);
        tuner.addConstraint(kernelId, { "TILE_SIZE_X", "VECTOR_TYPE" }, vConstraint);
        tuner.addConstraint(kernelId, { "LOCAL_MEM", "VECTOR_TYPE" }, vlConstraint);}
        break;
    case GEMM_REDUCED:{
        tuner.addParameter(kernelId, "MWG", {16, 32, 64});
        tuner.addParameter(kernelId, "NWG", {16, 32, 64});
        tuner.addParameter(kernelId, "KWG", {32});
        tuner.addParameter(kernelId, "MDIMC", {8, 16, 32});
        tuner.addParameter(kernelId, "NDIMC", {8, 16, 32});
        tuner.addParameter(kernelId, "MDIMA", {8, 16, 32});
        tuner.addParameter(kernelId, "NDIMB", {8, 16, 32});
        tuner.addParameter(kernelId, "KWI", {2});
        tuner.addParameter(kernelId, "VWM", {1, 2, 4});
        tuner.addParameter(kernelId, "VWN", {1, 2, 4});
        tuner.addParameter(kernelId, "STRM", {0});
        tuner.addParameter(kernelId, "STRN", {0});
        tuner.addParameter(kernelId, "SA", {0, 1});
        tuner.addParameter(kernelId, "SB", {0, 1});
        tuner.addParameter(kernelId, "PRECISION", {32});
        auto MultipleOfX = [](const std::vector<size_t>& v) {return IsMultiple(v[0], v[1]);};
        auto MultipleOfXMulY = [](const std::vector<size_t>& v) {return IsMultiple(v[0], v[1] * v[2]);};
        auto MultipleOfXMulYDivZ = [](const std::vector<size_t>& v) {return IsMultiple(v[0], (v[1] * v[2]) / v[3]);};
        tuner.addConstraint(kernelId, {"KWG", "KWI"}, MultipleOfX);
        tuner.addConstraint(kernelId, {"MWG", "MDIMC", "VWM"}, MultipleOfXMulY);
        tuner.addConstraint(kernelId, {"NWG", "NDIMC", "VWN"}, MultipleOfXMulY);
        tuner.addConstraint(kernelId, {"MWG", "MDIMA", "VWM"}, MultipleOfXMulY);
        tuner.addConstraint(kernelId, {"NWG", "NDIMB", "VWN"}, MultipleOfXMulY);
        tuner.addConstraint(kernelId, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, MultipleOfXMulYDivZ);
        tuner.addConstraint(kernelId, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, MultipleOfXMulYDivZ);}
        break;
    case GEMM_BATCHED:{
        int a, b, c;
        a = b = c = 16;
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "SIZE_A", {(size_t)a});
        tuner.addParameter(kernelId, "SIZE_B", {(size_t)b});
        tuner.addParameter(kernelId, "SIZE_C", {(size_t)c});
        tuner.addParameter(kernelId, "GROUP_SIZE_Y", {1, 2, 4, 8, 16, 32});
        tuner.addParameter(kernelId, "GROUP_SIZE_Z", {1, 2, 4, 8, 16, 32, 64});
        tuner.addParameter(kernelId, "CACHING_STRATEGY", {/*0, 1, */2}); /* 0 = implicit caching, 1 = local memory, 2 = private memory */
        tuner.addParameter(kernelId, "PADD_AA", {/*0,*/ 1});
        tuner.addParameter(kernelId, "PADD_AB", {/*0,*/ 1});
        if (c % 4 == 0)
            tuner.addParameter(kernelId, "PADD_C", {0});
        else
            tuner.addParameter(kernelId, "PADD_C", {0, c % 4});
        tuner.addParameter(kernelId, "DIRECT_WRITE", {/*0,*/ 1});
        tuner.addParameter(kernelId, "UNROLL_K", {/*0,*/ 1});
#else
        tuner.addParameter(kernelId, "SIZE_A", {(size_t)a});
        tuner.addParameter(kernelId, "SIZE_B", {(size_t)b});
        tuner.addParameter(kernelId, "SIZE_C", {(size_t)c});
        tuner.addParameter(kernelId, "GROUP_SIZE_Y", {1, 2, 4, 8, 16, 32});
        tuner.addParameter(kernelId, "GROUP_SIZE_Z", {1, 2, 4, 8, 16, 32, 64});
        tuner.addParameter(kernelId, "CACHING_STRATEGY", {0, 1, 2}); /* 0 = implicit caching, 1 = local memory, 2 = private memory */
        tuner.addParameter(kernelId, "PADD_AA", {0, 1});
        tuner.addParameter(kernelId, "PADD_AB", {0, 1});
        if (c % 4 == 0)
            tuner.addParameter(kernelId, "PADD_C", {0});
        else
            tuner.addParameter(kernelId, "PADD_C", {0, unsigned(c) % 4});
        tuner.addParameter(kernelId, "DIRECT_WRITE", {0, 1});
        tuner.addParameter(kernelId, "UNROLL_K", {0, 1});
#endif
        auto parallelismConstraint = [](const std::vector<size_t>& v) {return v[0] <= v[1];};
        tuner.addConstraint(kernelId, {"GROUP_SIZE_Y", "SIZE_B"}, parallelismConstraint);
        auto paddConstraint = [](const std::vector<size_t>& v) {return (v[0] == 0 && v[1] == 0 && v[2] == 0) || (v[3] > 0);};
        tuner.addConstraint(kernelId, {"PADD_AA", "PADD_AB", "PADD_C", "CACHING_STRATEGY"}, paddConstraint);
        auto dwConstraint = [](const std::vector<size_t>& v) {return (v[0] == 1) || (v[1] > 0);};
        tuner.addConstraint(kernelId, {"DIRECT_WRITE", "CACHING_STRATEGY"}, dwConstraint);
        auto unrollkConstraint = [](const std::vector<size_t>& v) {return (v[0] == 0) || (v[1] == 2);};
        tuner.addConstraint(kernelId, {"UNROLL_K", "CACHING_STRATEGY"}, unrollkConstraint);
    #define SHARED_PER_BLOCK (49152/4)
        auto memConstraint = [](const std::vector<size_t>& v) {size_t a = v[1]; size_t b = v[2]; size_t c = v[3]; return (v[0] == 1 && ((a+v[7])*(b+v[8])+c*a+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK) || (v[0] == 2 && v[5] == 1 && ((a+v[7])*(b+v[8])+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK) || (v[0] == 2 && ((a+v[7])*(b+v[8])+c*a+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK);};
        tuner.addConstraint(kernelId, {"CACHING_STRATEGY", "SIZE_A", "SIZE_B", "SIZE_C", "DIRECT_WRITE", "GROUP_SIZE_Y", "GROUP_SIZE_Z", "PADD_AA", "PADD_AB"}, memConstraint);
    #define MAX_BLOCK_SIZE 1024
        auto blockConstraint = [](const std::vector<size_t>&v) {return ((v[0]+v[2])*v[1]*v[3] < MAX_BLOCK_SIZE) && ((v[0]+v[2])*v[1]*v[3] >= 32);};
        tuner.addConstraint(kernelId, {"SIZE_C", "GROUP_SIZE_Y", "PADD_C", "GROUP_SIZE_Z"}, blockConstraint);}
        break;
    case FOURIER_REC_32:
    case FOURIER_REC_64:
    case FOURIER_REC_128:{
        tuner.addParameter(kernelId, "BLOCK_DIM_X", {8,12,16,20,24,28,32});
        tuner.addParameter(kernelId, "BLOCK_DIM_Y", {8,12,16,20,24,28,32});
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "TILE", {/*1,2,*/4/*,8*/});
#else
        tuner.addParameter(kernelId, "TILE", {1,2,4,8});
#endif

        tuner.addParameter(kernelId, "GRID_DIM_Z", {1,4,8,16});

        tuner.addParameter(kernelId, "SHARED_BLOB_TABLE", {0,1});
#if REDUCED_SET_1070 > 0
        tuner.addParameter(kernelId, "SHARED_IMG", {0/*, 1*/});
#else
        tuner.addParameter(kernelId, "SHARED_IMG", {0, 1});
#endif
        tuner.addParameter(kernelId, "USE_ATOMICS", {0, 1});
        tuner.addParameter(kernelId, "BLOB_TABLE_SIZE_SQRT", {10000});
        tuner.addParameter(kernelId, "PRECOMPUTE_BLOB_VAL", {0,1});
        if (problem == FOURIER_REC_32) {
            tuner.addParameter(kernelId, "cMaxVolumeIndexX", {64});
            tuner.addParameter(kernelId, "cMaxVolumeIndexYZ", {64});
        }
        else if (problem == FOURIER_REC_64) {
            tuner.addParameter(kernelId, "cMaxVolumeIndexX", {128});
            tuner.addParameter(kernelId, "cMaxVolumeIndexYZ", {128});
        }
        else if (problem == FOURIER_REC_128) {
            tuner.addParameter(kernelId, "cMaxVolumeIndexX", {256});
            tuner.addParameter(kernelId, "cMaxVolumeIndexYZ", {256});
        }
        tuner.addParameter(kernelId, "blobOrder", {0});
        auto blocksDimEqConstr = [](std::vector<size_t> vector) {return vector.at(0)== vector.at(1);};
        tuner.addConstraint(kernelId, std::vector<std::string>{"BLOCK_DIM_X", "BLOCK_DIM_Y"}, blocksDimEqConstr);
        auto tileMultXConstr = [](std::vector<size_t> vector) {return vector.at(1) == 1 || (vector.at(0) % vector.at(1) == 0);};
        tuner.addConstraint(kernelId, std::vector<std::string>{"BLOCK_DIM_X", "TILE"}, tileMultXConstr);
        auto tileMultYConstr = [](std::vector<size_t> vector) {return vector.at(1) == 1 || (vector.at(0) % vector.at(1) == 0);};
        tuner.addConstraint(kernelId, std::vector<std::string>{"BLOCK_DIM_Y", "TILE"}, tileMultYConstr);
        auto tileSharedImgConstr = [](std::vector<size_t> vector) {return vector.at(0) == 0 || vector.at(1) == 1;};
        tuner.addConstraint(kernelId, std::vector<std::string>{"SHARED_IMG", "TILE"}, tileSharedImgConstr);
        auto useAtomicsZDimConstr = [](std::vector<size_t> vector) {return !(vector.at(0) == 0 && vector.at(1) != 1);};
        tuner.addConstraint(kernelId, std::vector<std::string>{"USE_ATOMICS", "GRID_DIM_Z"}, useAtomicsZDimConstr);
        auto tileSmallerThanBlockConstr = [](std::vector<size_t> vector) {return vector.at(0) > vector.at(2) && vector.at(1) > vector.at(2);};
        tuner.addConstraint(kernelId, std::vector<std::string>{"BLOCK_DIM_X", "BLOCK_DIM_Y", "TILE"}, tileSmallerThanBlockConstr);
        auto tooMuchSharedMemConstr = [](std::vector<size_t> vector) {return !(vector.at(0)==1 && vector.at(1)==1);};
        tuner.addConstraint(kernelId, std::vector<std::string>{"SHARED_BLOB_TABLE", "SHARED_IMG"}, tooMuchSharedMemConstr);
        auto blobTableConstr = [](std::vector<size_t> vector) {return vector.at(0)==0 || (vector.at(0)==1 && vector.at(1)==1);};
        tuner.addConstraint(kernelId, std::vector<std::string>{"SHARED_BLOB_TABLE", "PRECOMPUTE_BLOB_VAL"}, blobTableConstr);}
        break;
    default:
        std::cerr << "Selected problem not implemented" << std::endl;
        exit(1);
    }
}

void configureSearcher(ktt::Tuner &tuner, eSearchMethod searcher, eProblem problem){
    switch(searcher) {
    case RANDOM:
        tuner.setSearchMethod(ktt::SearchMethod::RandomSearch, {});
        break;
    case ANNEALING:
        tuner.setSearchMethod(ktt::SearchMethod::Annealing, {4.0});
        break;
    case MCMC_SIMPLE:
        tuner.setSearchMethod(ktt::SearchMethod::MCMC, {});
        break;
    case MCMC_SENSIT:
        switch(problem) {
            case COLUMB_3D:
                tuner.setSearchMethod(ktt::SearchMethod::MCMC, std::vector<double>{
                0.1, 1, 0.1, 5, 5, 0.1, 1, 4,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1}); /* optimum for GTX 1070, 128x128, 4000 atoms*/
                break;
            default:
                std::cerr << "Selected search method is not implemented for given problem" << std::endl;
                exit(1);
        }
        break;
    case MCMC_CONVEX:
        switch(problem) {
            case COLUMB_3D:
                tuner.setSearchMethod(ktt::SearchMethod::MCMC, std::vector<double>{
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                16, 4, 1, 8, 0.1, 1, 0.1, 1}); /* optimum for GTX 1070, 128x128, 4000 atoms*/
                break;
            default:
                std::cerr << "Selected search method is not implemented for given problem" << std::endl;
                exit(1);
        }
        break;
    case MCMC_FULL:
        switch(problem) {
            case COLUMB_3D:
                tuner.setSearchMethod(ktt::SearchMethod::MCMC, std::vector<double>{
                0.1, 1, 0.1, 5, 5, 0.1, 1, 4,
                1, 1, 1, 1, 1, 1, 1, 1,
                16, 4, 1, 8, 0.1, 1, 0.1, 1}); /* optimum for GTX 1070, 128x128, 4000 atoms*/
                break;
            default:
                std::cerr << "Selected search method is not implemented for given problem" << std::endl;
                exit(1);
        }
        break;
    default:
        std::cerr << "Selected search method is not implemented" << std::endl;
        exit(1);
    }
}

