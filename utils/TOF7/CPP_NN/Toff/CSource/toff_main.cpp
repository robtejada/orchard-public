
#include "./aux_utils.h"
#include "./toff_globals.h"
#include "./figureFkts20.h"
#include "./toff_defines.h"

//---------------------------------------
//*** global variables ******************
//---------------------------------------


//*** functions defined here ***
int main(void);
void example_use_coeffs_Sn_Snp_m_fn_fnp(void);




//*** entry point of program ***
int main(void)
{
	printf("-Begin of Program-\n");
	initialize_global_variables();

	//*** enable for testing, default: disabled ***
	example_use_coeffs_Sn_Snp_m_fn_fnp();

	delete_global_variables(); 
	printf("-End of Program-\n");
	warte_getchar();
	return 0;
}


//----------------------------------------------------------------------------------
//*** EXAMPLE  *********************************************************************
//----------------------------------------------------------------------------------
// The code below is meant to be an example of how the five functions
// 'coeff_Sn()', 'coeff_Snp()', 'coeff_m()', 'inner_f()', 'inner_fprime()' ,
// can be used. They provide access to the ToF-coefficients in the tables. 
// WARNING: the code below is not meant work stand-alone. 
//----------------------------------------------------------------------------------


//***  external variables ***
const int nl=3; // number of radial grid points
double g_sn[nl][TOFF_APPROX]; // array of all figure functions from previous iteration, here per default all zero


void example_use_coeffs_Sn_Snp_m_fn_fnp(void)
{
	const int il=nl-1; // l at surface or center 
	const int n=2; // calculate updated figure function value s2(l) at l
	double snl[TOFF_APPROX+1]; // [s2 at l, s4 at l, s6 at l, etc]
	const double mrot=0.083;
	double sn_new, fn, fnp, intS[TOFF_APPROX+1], intSp[TOFF_APPROX+1];
	int k;
	
	
	printf("---example of using the five functions ---\n");
	if(n<2 || n%2 || n>2*TOFF_APPROX)
	{
		printf("invalid index n=%d \n",n); exit_abfrage();
	}
	for(k=0; k<=TOFF_APPROX; k++)
	{
		snl[k] = g_sn[il][k]; // values of s2, s4, s6 etc at l

		//*** default values, insert your own integral values ***
		intS[k] = 1.; // is integral S_2k from center to l based on the figure functions g_sn
		intSp[k] = 0.; // is integral S_2k_prime from l to surface based on the figure functions g_sn
	}
	//*** function coeff_m() is used here: ***
	sn_new = mrot * coeff_m(n,snl);
	
	for(k=0; k<=TOFF_APPROX; k++)
	{
		//*** functions coeff_Sn(), coeff_Snp() are used here: ***
		sn_new += intS[k] * coeff_S(n,2*k,snl) + intSp[k] * coeff_Sprime(n,2*k,snl); 
	}
	
	//*** updated value of figure function sn at l ***
	sn_new /= intS[0];


	//*** function values of the fn, fnp at l ***
	fn = inner_f(n, snl);
	fnp = inner_fprime(n, snl);
}


