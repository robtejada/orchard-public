// C++ code by Nadine Nettelmann
#include<stdio.h>
#include<string.h> 
#include<stdlib.h>

#include "./aux_utils.h"


void openFile( FILE* &pFile, const char* fname, const char* mod )
{
	if(pFile != NULL)
	{
		printf("error in openFile: initial value of pFile != NULL !"); exit_abfrage();
	}
	if(fname == NULL)
	{
		printf("error in openFile: fname = NULL !"); exit_abfrage();
	}
	else pFile = fopen( fname, mod );
	if( pFile == NULL )
	{
 		printf("error: could not open file '%s' in modus '%s'\n", fname, mod ); 
		perror("perror: ");
		exit_abfrage();
	}
}


void closeFile( FILE* &pFile )
{
	if( pFile != NULL )
	{
		fclose(pFile);
		pFile = NULL;
	}
}


void warte_getchar(void)
{
	char c='\0';
	
	printf("press 'O'K to continue  "); fflush(stdout);
	while( (c=(char)getchar()) != 'O' );
}

void exit_abfrage()
{
	char	c='\0';
	
	printf("Wanna abort program? 'y/n' :  "); fflush(stdout);
	while( ! ((c=getchar()) == 'y' || (c=='n')) );
	if( c == 'y' )	exit(1);
}

//---------------------------------------------------------------------------------
//******* class CString  **********************************************************
//---------------------------------------------------------------------------------

int CString::sizeofchar = sizeof(char);
CString::CString(void)
{
	m_buf = NULL;
}

CString::CString(const CString &str)
{
	int k = str.strlength();
	if(k)
	{
		m_buf = new char[k+1];
		memcpy(m_buf, str.m_buf, k*sizeofchar);
		m_buf[k] = '\0';
	}
	else m_buf = NULL;
}
CString::CString(const char* pc)
{
	int k= strlength(pc);
	if(k)
	{ 
		m_buf = new char[k+1];
		memcpy(m_buf, pc, k*sizeofchar);
		m_buf[k] = '\0';
	}
	else m_buf = NULL;
}
CString::~CString(void)
{
	if(m_buf != NULL) delete [] m_buf;
}

void CString::operator=(const CString &str)
{
	int k = str.strlength();

	delete [] m_buf; m_buf = NULL;
	if(k)
	{
		m_buf = new char[k+1];
		memcpy(m_buf, str.m_buf, k*sizeofchar);
		m_buf[k] = '\0';
	}
}

void CString::operator=(const char* pc)
{
	int k = strlen(pc);

	delete [] m_buf; m_buf = NULL;
	if(k)
	{
		m_buf = new char[k+1];
		memcpy(m_buf, pc, k*sizeofchar);
		m_buf[k] = '\0';
	}
}

CString CString::operator+(const CString &str2)
{
	int		n,m;
	char	*buf=NULL;

	n = strlength();
	m = str2.strlength();
	buf = new char[n+m+1];
	memcpy(buf, m_buf, n*sizeofchar);
	memcpy(buf+n, str2.m_buf, m*sizeofchar);
	buf[m+n] = '\0';

	CString strSum(buf);
	delete [] buf;
	return strSum;
}

int CString::strlength(void) const 
{ 
	return strlength(m_buf); 
} 

int CString::strlength (const char* buff) const 
{
	if(buff == NULL) return 0;
	else return strlen(buff); // returns number of chars before '\0'
}

/*
void CString::append(const CString &str)
{
	int		n,m;
	char	*buf=NULL;

	n = this->strlength();
	m = str.strlength();
	if(m)
	{	buf = new char[n+1];
		memcpy(buf,m_buf,n*sizeofchar); 
		buf[n]='\0';
		delete [] m_buf;
		m_buf = new char[n+m+1];
		memcpy(m_buf,buf,n*sizeofchar);
		memcpy(m_buf+n,str.m_buf,m*sizeofchar);
		m_buf[n+m]='\0';
		delete [] buf;
	}
}
*/
const char* CString::get(void)
{
	return m_buf;
}


//-----------------------------------------------------------------------
//*** class TSortedList<int>**** ****************************************
//-----------------------------------------------------------------------
/*
//template<>
int TCSortedList<int>::addElement(int &e) // for integer
{
	TCSortedList<int>::iterator	it;
	int eval=e, it_eval=0;
	int idx=-1;

	if(this->size() == 0) // first
	{
		this->push_back(e); return 1;
	}
	it = this->begin(); 
	if(eval < (*it)) // smaller than all
	{
		this->push_front(e); return 1;
	}
	it=this->end(); it--;
	if(eval > (*it)) // larger than all
	{
		this->push_back(e); (int)this->size();
	}
	for(it=this->begin(), idx=0; it != this->end(); it++, idx++)
	{
		it_eval = (*it);
		if(eval == it_eval) break;
		if(eval < it_eval)
		{
			this->insert(it,e); break;
		}	
	}
	if(it == this->end())
	{
		printf("Fehler in TSortedList::addElement!\n"); exit_abfrage();
	}
	return idx;
}


//-----------------------------------------------------------------------
//*** class TSortedList<double> *****************************************
//-----------------------------------------------------------------------


//template<> // g++ - Compilerehler
int TCSortedList<double>::addElement(double &e) // for double
{
	TCSortedList<double>::iterator	it;
	double eval=e, it_eval=0, err=1e-12;
	int idx=-1;

	if(this->size() == 0) // first
	{
		this->push_back(e); return 1;
	}
	it = this->begin(); 
	if(eval*(1.+err) < (*it)) // smaller than all
	{
		this->push_front(e); return 1;
	}
	it=this->end(); it--;
	if(eval*(1.-err) > (*it)) // larger than all
	{
		this->push_back(e); (int)this->size();
	}
	for(it=this->begin(), idx=0; it != this->end(); it++, idx++)
	{
		it_eval = (*it);
		if(fabs(eval/it_eval -1.) < err) break;
		if(eval*(1.+err) < it_eval)
		{
			this->insert(it,e); break;
		}	
	}
	if(it == this->end())
	{
		printf("Fehler in TSortedList::addElement!\n"); exit_abfrage();
	}
	return idx;
}
*/



