
#ifndef TOFF_DEFS_H
#define TOFF_DEFS_H

#define TOFF_APPROX 7 // allowed values: [2,3,...,7]


#if(TOFF_APPROX > 7) 
#pragma message("WARNING: ToF order larger than maximum of 7!")
#endif


#endif