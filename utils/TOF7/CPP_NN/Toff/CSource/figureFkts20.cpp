// code by Nadine Nettelmann, 2020

#include "./toff_defines.h"
#include "./figureFkts20.h"
#include "./toff_cData.h"
#include "./toff_globals.h"
#include "./aux_utils.h"

extern CToffCoff2020 *g_pToffCoff2020;



//*** user: call only these five functions, do not change anything ***

double coeff_S(int idxA_k, int idxS_n, double *sn)
{
	double coeffAk = g_pToffCoff2020->coeff_Ak_Sn(idxA_k, idxS_n, sn);
	return coeffAk;
}	

double coeff_Sprime(int idxA_k, int idxS_n, double *sn)
{
	return g_pToffCoff2020->coeff_Ak_Snprime(idxA_k, idxS_n, sn);
}

double coeff_m(int k, double *sn)
{
	double coeffm = (1./3.)*g_pToffCoff2020->coeff_m(k, sn);
	return coeffm;
}

double inner_f( int n, double* sn )
{
	double fn = g_pToffCoff2020->coeff_inner_fn(n, sn);
	return fn;
}

double inner_fprime( int n, double* sn ) // returning the full fnp with ana-term
{
	double fnp = g_pToffCoff2020->coeff_inner_fnprime(n, sn);
	return fnp;
}



//----------------------------------------------------------------
//*** class CToffCoff2020 ***-------------------------------------
//----------------------------------------------------------------


CToffCoff2020::~CToffCoff2020(void)
{
	delete tab_sn; delete tab_snp; delete tab_fn; delete tab_fnp; delete tab_m;
}

CToffCoff2020::CToffCoff2020(void)
{
	tab_sn=NULL; tab_snp=NULL; tab_fn=NULL; tab_fnp=NULL; tab_m=NULL;

	tab_sn = new CToffCoffFileData;
	tab_sn->readFile( (g_absDir + CString(g_fname_tab_Sn)).get() );
	tab_snp = new CToffCoffFileData;
	tab_snp->readFile( (g_absDir + CString(g_fname_tab_Snp)).get() );
	tab_fn = new CToffCoffFileData;
	tab_fn->readFile( (g_absDir + CString(g_fname_tab_fn)).get() );
	tab_fnp = new CToffCoffFileData;
	tab_fnp->readFile( (g_absDir + CString(g_fname_tab_fnp)).get() );
	tab_m = new CToffCoffFileData;
	tab_m->readFile( (g_absDir + CString(g_fname_tab_m)).get() );
}

double CToffCoff2020::summup( CToffCoffFileData* tab, int k, int n, double *sn)
{
	CToffCoffFileData::SingleSummand *pss = tab->getSummands(n,k);
	double xsum = 0.;
	int i, nofcoff = tab->getNofCoff(n,k), io;

	for(i=0; i<nofcoff; i++)
	{
		io = pss[i].getOrder() ;
		if(io > TOFF_APPROX) break;
		xsum += pss[i].getValue(sn);
	}
	return xsum;
}

double CToffCoff2020::coeff_Ak_Sn(int idxA_k, int idxS_n, double *sn)
{
	return summup(tab_sn, idxA_k, idxS_n, sn);
}

double CToffCoff2020::coeff_Ak_Snprime(int idxA_k, int idxS_n, double *sn)
{
	return summup(tab_snp, idxA_k, idxS_n, sn);
}

double CToffCoff2020::coeff_m(int idxA_k, double *sn)
{
	return summup(tab_m, idxA_k, 0, sn);
}

double CToffCoff2020::coeff_inner_fn(int n, double* sn)
{
	return summup(tab_fn, 0, n, sn);
}

double CToffCoff2020::coeff_inner_fnprime(int n, double* sn)
{
	return 	summup(tab_fnp, 0, n, sn);
}


