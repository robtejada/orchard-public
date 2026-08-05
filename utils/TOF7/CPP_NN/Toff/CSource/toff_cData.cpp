
#include "./aux_utils.h"
#include "./aux_utils.hxx"
#include "./aux_numParse.h"
#include "./toff_cData.h"



//----------------------------------------------------------------------------------
//*** class CToffCoffFileData::SingleSummand *** -----------------------------------
//----------------------------------------------------------------------------------


CToffCoffFileData::SingleSummand::~SingleSummand(void)
{
	delete [] m_snPotArr;
}

CToffCoffFileData::SingleSummand::SingleSummand(void)
{
	m_binit = false;
	m_ptffff = NULL;
}

CToffCoffFileData::SingleSummand::SingleSummand(CToffCoffFileData *p)
{
	initialize(p);
}

void CToffCoffFileData::SingleSummand::initialize(CToffCoffFileData *p)
{
	if(p->m_maxorder<0 || p->m_maxorder>7)
	{
		printf("Error in CToffCoffFileData::SingleSummand::initialize(): invalid value maxorder=%d!", p->m_maxorder); exit_abfrage(); 
	}
	m_ptffff = p;
	m_snPotArr = new int [m_ptffff->m_maxorder+1];
	m_snPotArr[0] = -1; // unsorted
	m_binit = true;
}

void CToffCoffFileData::SingleSummand::setRow(int *iarr, double x)
{
	int i;

	if(!m_binit)
	{
		printf("Error in SingleSummand::setRow(): must first initialize! "); exit_abfrage();
	}
	for(i=0; i<= m_ptffff->m_maxorder; i++)
	{
		m_snPotArr[i] = iarr[i]; 
	}	
	m_coff = x;
}


double CToffCoffFileData::SingleSummand::getValue(double *sn)
{
	double x=1., snhoch;
	int i,j;

	if(!m_binit)
	{
		printf("Error in SingleSummand::getValue(): must first initialize! "); exit_abfrage();
	}
	for(i=1; i<=m_ptffff->m_maxorder; i++)
	{
		snhoch = 1.;
		for(j=0; j<m_snPotArr[i]; j++) snhoch *= sn[i];
		x *= snhoch;
	}
	return x*m_coff;
}

int CToffCoffFileData::SingleSummand::getExpo(int n)
{
	if(n<2 || n>m_ptffff->m_maxorder2 || n%2)
	{
		printf("in SingleSummand::getExpo(): invalid Index n=%d ", n); exit_abfrage();
		return 0;
	}
	return m_snPotArr[n/2];	
}	

int CToffCoffFileData::SingleSummand::getOrder(void)
{ 
	if(!m_binit)
	{
		printf("Error in SingleSummand::getOrder(): must first initialize! "); exit_abfrage();
		return 0;
	}
	return m_snPotArr[0]; 
}

		
int CToffCoffFileData::SingleSummand::calcOrder_from_sn(void)
{
	int i;
	if(!m_binit)
	{
		printf("Fehler in SingleSummand::calcOrder_from_sn(): not initialized! "); exit_abfrage();
		return 0;
	}
	m_snPotArr[0] = 0;
	for(i=1; i<= m_ptffff->m_maxorder; i++) m_snPotArr[0] += i*m_snPotArr[i]; 
	return m_snPotArr[0];
}	

//----------------------------------------------------------------------------------
//*** class CToffCoffFileData *** -----------------------------------------------------
//----------------------------------------------------------------------------------

CToffCoffFileData::CToffCoffFileData(void)
{
	m_maxorder=0; m_nofcoffArr_XnAk=NULL;
	m_ssArr = NULL;
}

CToffCoffFileData::~CToffCoffFileData(void)
{
	int i, imax = sq(m_maxorder+1);

	if(m_ssArr)
	{ 
		for(i=0; i<imax; i++){ if(m_ssArr[i]) delete [] m_ssArr[i]; }
		delete [] m_ssArr; 
	}
	if( m_nofcoffArr_XnAk ) delete [] m_nofcoffArr_XnAk;
}

int CToffCoffFileData::getNofCoff(int n, int k)
{
	if(n<0 || k<0 || n>m_maxorder2 || k>m_maxorder2 || !m_maxorder || n%2 || k%2)
	{
		printf("Error in getNofCoff(%d,%d): index! ",n,k); exit_abfrage();
	}
	return m_nofcoffArr_XnAk[n/2*(m_maxorder+1) + k/2];
}


void CToffCoffFileData::setNofCoff(int n, int k, int mnof)
{
	
	if(n<0 || k<0 || n>2*m_maxorder || k>2*m_maxorder || !m_maxorder || n%2 || k%2)
	{
		printf("Fehler in setNofCoff(%d,%d,%d): index! ",n,k,mnof); exit_abfrage();
	}
	m_nofcoffArr_XnAk[n/2*(m_maxorder+1) + k/2] = mnof;
}

CToffCoffFileData::SingleSummand* CToffCoffFileData::getSummands(int n, int k)
{
	if(n<0 || k<0 || n>m_maxorder2 || k>m_maxorder2 || !m_maxorder || n%2 || k%2)
	{
		printf("Error in CToffCoffFileData::getSummands_SnAk(): index! "); exit_abfrage();
	}
	return m_ssArr[n/2*(m_maxorder+1) + k/2];
}

void CToffCoffFileData::readFile(const char *fullfname)
{
	FILE *pFi=NULL;
	int k,n,m,i,j,nk, ko, no, nprev=-13,kprev=-13,mprev=-13;
	int *hochz=NULL, maxorder;	
	double x;
		
	openFile(pFi,fullfname,"r");
	if(!pFi){ printf("Table has not been read.\n"); return; }

int ixyz = m_maxorder;
	readNextNumber_d(pFi,maxorder);
	set_max_order(maxorder);
	m_nofcoffArr_XnAk = new int [sq(m_maxorder+1)];
	m_ssArr = new SingleSummand* [sq(m_maxorder+1)];
	hochz = new int [m_maxorder+1];

	//*** default Werte / Initialization ***
	for(i=0; i<sq(m_maxorder+1); i++)
	{	
		m_nofcoffArr_XnAk[i] = -1;
		m_ssArr[i] = NULL;
	}
	while(!feof(pFi))
	{
		if(!readNextNumber_d(pFi,n)) break;
		if(!readNextNumber_d(pFi,k)) break;
		if(!readNextNumber_d(pFi,m)) break;

		if(n<0 || k<0 || m<0)
		{
			printf("Error in CToffCoffFileData::readFile(): n=%d  k=%d  m=%d ", n,k,m); exit_abfrage();
		}

		setNofCoff(n,k,m);
		ko = k/2; no=n/2;
		nk = no*(m_maxorder+1) + ko;

		m_ssArr[nk] = new SingleSummand [m] ;
		for(i=0; i<m; i++)
		{
			m_ssArr[nk][i].initialize(this);
			for(j=0; j<=m_maxorder; j++) readNextNumber_d(pFi,hochz[j]);
			readNextNumber_e(pFi,x);
			m_ssArr[nk][i].setRow(hochz, x);
		}
		nprev=n; kprev=k, mprev=m;
	}
	closeFile(pFi);
	delete [] hochz;	
}


