
#ifndef FIGFKTS20_H
#define FIGFKTS20_H

//*** classes defined here ***
class CToffCoff2020;


//*** classes used here ***
class CToffCoffFileData;

double coeff_S(int idxA_k, int idxS_n, double *sn);
double coeff_Sprime(int idxA_k, int idxS_n, double *sn);
double coeff_m(int k, double *sn);
double inner_f(int n, double* sn);
double inner_fprime(int n, double* sn);



class CToffCoff2020
{
	CToffCoffFileData	*tab_sn, *tab_snp, *tab_fn, *tab_fnp, *tab_m; 


public:
	~CToffCoff2020(void);
	CToffCoff2020(void);
	double coeff_Ak_Sn(int idxA_k, int idxS_n, double *sn);
	double coeff_Ak_Snprime(int idxA_k, int idxS_n, double *sn);
	double coeff_m(int k, double *sn);
	double coeff_inner_fn(int n, double* sn);
	double coeff_inner_fnprime(int n, double* sn);
	double summup( CToffCoffFileData* tab, int k, int n, double *sn);
};

#endif
