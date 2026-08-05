#ifndef GLOBALS_H
#define GLOBALS_H

//*** classes used here ********
class CString;

extern CString	g_absDir; 
extern const char *g_fname_tab_Sn, *g_fname_tab_Snp, *g_fname_tab_fn, *g_fname_tab_fnp, *g_fname_tab_m;

void initialize_global_variables(void); 
void delete_global_variables(void); 

#endif
