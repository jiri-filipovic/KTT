/** @file Hardware_Architecture.h
* Definition of enum for Hardware and architecture type of kernel arguments.
*/
#pragma once

namespace ktt
{

	/** @enum ArgumentDataType
	* Enum for hardware type and architecture model for DLNezarat searcher
	*/
	//enum class Hardware_Architecture
	//{
	enum ARCHITECTURE {
		A_Vega56 = 7, //Radeoon
		A_TitanV = 5, //Volta
		A_P100 = 2, //Pascal
		A_Mic5110P = 6, //XeonPhi
		A_K20 = 4, //Kepler
		A_Gtx2080Ti = 3, //Turing
		A_Gtx1070 = 2, //Pascal
		A_Gtx750 = 1, //Maxwell
		A_Gtx680 = 4 //Kepler
	};

} // namespace ktt
