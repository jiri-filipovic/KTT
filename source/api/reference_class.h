/** @file reference_class.h
  * @brief File containing functionality related to validating kernel output with reference class.
  */
#pragma once

#include "ktt_types.h"

namespace ktt
{

/** @class ReferenceClass
  * @brief Class which can be used to compute reference output for selected kernel arguments inside regular C++ method. In order to use this
  * functionality, new class which publicly inherits from reference class has to be defined.
  */
class ReferenceClass
{
public:
    /** @fn ~ReferenceClass()
      * @brief Reference class destructor. Inheriting class can override destructor with custom implementation. Default implementation is
      * provided by KTT library.
      */
    virtual ~ReferenceClass() = default;

    /** @fn computeResult()
      * @brief Computes reference output for all kernel arguments validated by the class and stores it for later retrieval by tuner. Inheriting class
      * must provide implementation for this method.
      */
    virtual void computeResult() = 0;

    /** @fn getData(const ArgumentId id)
      * @brief Returns pointer to buffer containing reference output for specified kernel argument. This method will be called only after running
      * computeResult() method. It can be called multiple times for same kernel argument. Inheriting class must provide implementation for this
      * method.
      * @param id Id of kernel argument for which reference output is retrieved. This can be used by inheriting class to support validation of
      * multiple kernel arguments.
      * @return Pointer to buffer containing reference output for specified kernel argument.
      */
    virtual void* getData(const ArgumentId id) = 0;

    /** @fn getNumberOfElements(const ArgumentId id) const
      * @brief Returns number of validated elements returned by getData() method for specified kernel argument. This method will be called only after
      * running computeResult() method. It can be called multiple times for same kernel argument. Inheriting class can override this method, which is
      * useful in conjuction with Tuner::setValidationRange() method. If number of validated elements equals zero, all elements in corresponding
      * kernel argument will be validated.
      * @param id Id of kernel argument for which number of validated elements is retrieved. This can be used by inheriting class to support
      * validation of multiple kernel arguments.
      */
    virtual size_t getNumberOfElements(const ArgumentId id) const
    {
        return 0;
    }
};

} // namespace ktt
