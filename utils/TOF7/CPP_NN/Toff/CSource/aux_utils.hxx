#ifndef AUX_UTILS_HXX
#define AUX_UTILS_HXX

#include "./aux_utils.h"

template<class X> X		sq( X x );

template <class X>
X sq( X x )
{
	return(x*x);
}

#endif