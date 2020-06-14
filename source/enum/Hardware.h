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
	enum HARDWARE {
		Vega56 = 1,
		TitanV,
		P100,
		Mic5110P,
		K20,
		Gtx2080Ti,
		Gtx1070,
		Gtx750,
		Gtx680
	};
	
} // namespace ktt
