#include "./aux_utils.h"
#include "./toff_globals.h"
#include "./figureFkts20.h"

CString			g_absDir;
CToffCoff2020	*g_pToffCoff2020;



void initialize_global_variables(void) 
{	
	g_absDir = "C:/Home/Toff/";
	//g_absDir = "/your_directory/Toff";

	g_pToffCoff2020 = new CToffCoff2020;
}

void delete_global_variables(void) 
{	
	delete g_pToffCoff2020;
}


//*** data file names ***
const char *g_fname_tab_Sn = "InputData/tab_Sn7.dat", 
			*g_fname_tab_Snp = "InputData/tab_Snp7.dat",
			*g_fname_tab_fn = "InputData/tab_fn7.dat", 
			*g_fname_tab_fnp = "InputData/tab_fnp7.dat", 
			*g_fname_tab_m = "InputData/tab_m7.dat";
