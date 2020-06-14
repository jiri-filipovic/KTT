#pragma once


#include <tuning_runner/searcher/searcher.h>
#include <dto/kernel_result.h>

#include <enum/Hardware.h>
#include <enum/Architecture.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <random>
#include <algorithm>  // copy()
#include <iterator>   // istream_iterator, back_inserter
#include <sstream>    // istringstream
#include <vector>
#include <numeric>





std::string exec(const char* cmd) {
	std::array<char, sizeof(cmd)> buffer;
	std::string result;
	#ifdef _WIN32
		std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
	#else
		std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
	#endif
	if (!pipe) {
		throw std::runtime_error("popen() failed!");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
		result += buffer.data();
	}
	return result;
}


std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
	str.erase(0, str.find_first_not_of(chars));
	return str;
}

std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
	str.erase(str.find_last_not_of(chars) + 1);
	return str;
}

std::string& trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
	return ltrim(rtrim(str, chars), chars);
}

std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
	}
	return str;
}




namespace ktt
{

	class DLNezaratSearcher : public Searcher
	{
	public:
		DLNezaratSearcher(const std::vector<KernelConfiguration>& configurations, const double problem, float hw, float arch) :
			configurations(configurations),
			configurationIndices(configurations.size()),
			index(0)
		{

			if (configurations.size() == 0)
			{
				throw std::runtime_error("Configurations vector provided for searcher is empty");
			}

			std::random_device device;
			std::default_random_engine engine(device());

			// Retriving the values of configuraion space for each row and predict the Computation time then we sort it.
			int hardware = hw;
			int Architecture = arch;
			std::string str_val = "";
			std::string str_CSV = "";
			char* c_val = "";
			std::string res = "";
			size_t start = 0;
			size_t finish = 100;
			size_t step = 100;
			size_t partition = ceil(configurations.size() / step);
			size_t remained = configurations.size() % step;
			for (size_t p = 0; p < partition; p++) {
				#ifdef _WIN32
					str_val = "C:\\ProgramData\\Anaconda3\\Scripts\\activate.bat C:\\ProgramData\\Anaconda3 & python DLNezaratpredictor.py ..\\trainedData\\Proposed_" + getPerfTrainedFileName(problem) + ".sav [";
				#else
				    str_val = "python DLNezaratpredictor.py trainedData/Proposed_" + getPerfTrainedFileName(problem) + ".sav [";
				#endif
				
				
				for (size_t i = start ; i < finish; i++)
				{
					str_val = str_val + "[";
					str_CSV = str_CSV + getPerfTrainedFileName(problem) + ",";
					std::size_t GlobalSize = configurations.at(i).getGlobalSize().getTotalSize();
					std::size_t LocalSize = configurations.at(i).getLocalSize().getTotalSize();
					str_val = str_val + std::to_string(GlobalSize);
					str_CSV = str_CSV + std::to_string(GlobalSize) + ",";
					str_val = str_val + "," + std::to_string(LocalSize);
					str_CSV = str_CSV + std::to_string(LocalSize) + ",";
					std::size_t val = 0;
					for (size_t j = 0; j < configurations.at(0).getParameterPairs().capacity(); j++)
					{
						val = configurations.at(i).getParameterPairs().at(j).getValue();
						str_val = str_val + "," + std::to_string(val);
						str_CSV = str_CSV + std::to_string(val) + ",";
					}
					str_val = str_val + "," + std::to_string(hardware);
					str_CSV = str_CSV + std::to_string(hardware) + ",";
					str_val = str_val + "," + std::to_string(Architecture);
					str_CSV = str_CSV + std::to_string(Architecture) + "\n";
					if (i == finish - 1)
					{
						str_val = str_val + "]";
					}
					else {
						str_val = str_val + "],";
					}

				}
				str_val = str_val + "]";
				c_val = const_cast<char*>(str_val.c_str());
				res = res + exec(c_val);


				start = start + step ;
				if ((p == (partition -1)) && (remained != 0))
				{
					partition = partition + 1;
					finish = start + remained;
					remained = 0;
				}
				else
				{
					finish = start + step;
				}
				

			}
			//res = trim(ReplaceAll(res, "]", ""));
			res = ReplaceAll(res, "]", "");
			res = ReplaceAll(res, "[", "");
			res = ReplaceAll(res, "\n"," ");
			res = ReplaceAll(res, "    ", " ");
			res = ReplaceAll(res, "   ", " ");
			res = ReplaceAll(res, "  ", " ");
			std::vector <float> predicted = using_vector(res);  // Converting string of float to vector of float

			std::ofstream outputFile(getPerfTrainedFileName(problem) + "_predicted.csv");
			outputFile << str_CSV;
			outputFile.close();

			std::ifstream file(getPerfTrainedFileName(problem) + "_predicted.csv");
			std::string read_str="";
			std::string final_str = "";
			int pos = 0;
			while (std::getline(file, read_str)) {
				final_str = final_str + read_str + "," + std::to_string(predicted.at(pos)) + "\n";
				pos++;
			}
			file.close();

			std::ofstream outputFileFinal(getPerfTrainedFileName(problem) + "_predicted.csv");
			outputFileFinal << final_str;
			outputFileFinal.close();

			
			// Sorting the predicted vector and creating another vector for its corresponding index
			std::vector<size_t> y(predicted.size());
			std::iota(y.begin(), y.end(), 0);
			auto comparator = [&predicted](int a, int b) { return predicted[a] < predicted[b]; };
			std::sort(y.begin(), y.end(), comparator);
			//std::ofstream of1("temp.txt");
			//std::string pr = "";
			//pr = "dsfsdf";
			//	//std::to_string(predicted.at(0)) + "hey";
			//of1 << pr;
			//of1.close();
			std::cout << "Computation time for best configuration for Deep Learning searcher is :" << std::to_string(predicted.at(y.at(0))/1000) << "us" << std::endl;
			configurationIndices = y;

			


		}


		
		
		void calculateNextConfiguration(const bool, const KernelConfiguration&, const double, const KernelProfilingData&,
			const std::map<KernelId, KernelProfilingData>&) override
		{
			index++;
		}

		KernelConfiguration getNextConfiguration() const override
		{
			size_t currentIndex = configurationIndices.at(index);
			return configurations.at(currentIndex);
		}

		size_t getUnexploredConfigurationCount() const override
		{
			size_t topN = ceil(configurations.size() * 0.10);
			if (index >= topN)
			{
				return 0;
			}

			return topN - index;
		}

		// Converting string of floats to vector of floats
		std::vector <float> using_vector(std::string InputVertices)
		{
			//std::cout << "\nUsing vector\n";

			std::vector <float> OutputVertices;
			//std::cout << "input  = \"" << InputVertices << "\";\n";

			// conversion here:
			std::istringstream ss(InputVertices);
			std::copy(
				std::istream_iterator <float>(ss),
				std::istream_iterator <float>(),
				back_inserter(OutputVertices)
			);

			//std::cout << "output = {";
			//for (size_t n = 0; n < OutputVertices.size(); n++)
			//	std::cout << OutputVertices[n] << ",";
			//std::cout << "\b};\n";
			return OutputVertices;
		}

		
		
	private:

		std::string getPerfTrainedFileName(int problem) {
			switch (problem) {
			case 0:
				return "bicg";
				break;
			case 1:
				return "conv";
				break;
			case 2:
				return "coulomb_sum_3d";
				break;
			case 3:
				return "gemm";
				break;
			case 4:
				return "hotspot";
				break;
			case 5:
				return "nbody";
				break;
			case 6:
				return "reduction";
				break;
			case 7:
				return "sort";
				break;
			case 8:
				return "mtran";
				break;
			case 10:
				return "gemm_batch";
				break;
			case 11:
				return "fourier_32";
				break;
			case 12:
				return "fourier_64";
				break;
			case 13:
				return "fourier_128";
				break;
			}
		}

		const std::vector<KernelConfiguration>& configurations;
		std::vector<size_t> configurationIndices;
		size_t index;
	};


} // namespace ktt
