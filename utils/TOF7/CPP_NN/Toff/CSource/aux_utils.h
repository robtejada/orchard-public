
#ifndef NNUTIL_H
#define NNUTIL_H

#pragma warning (disable : 4244 )
#pragma warning (disable : 4101 ) // unreferenced local variable
#pragma warning (disable : 4018) // signed/unsigned mismatch
#pragma warning (disable : 4250)
#pragma warning (disable : 4715)
#pragma warning (disable : 4996) // sprintf may be unsafe


#include<stdio.h>
#include<math.h>



//** classes defined here *******
class CString;

void 	openFile( FILE* &pFile, const char* fname, const char* mod );	
void 	openFile( FILE* &pFile, const char* absPath, const char* fname, const char* mod );	
void	closeFile( FILE* &pFile );
void	warte_getchar(void);
void	exit_abfrage(void);




//---------------------------------------------------------------------------------
//*** classes *********************************************************************
//---------------------------------------------------------------------------------

class CString
{
	char *m_buf;
	static int sizeofchar;

 public:
	CString(void);
	CString(const CString &s);
	CString(const char* pc);
	~CString(void);
	int	strlength (void) const;
	int	strlength(const char*) const;
	const char*	get(void);
	void operator=(const CString &);
	void operator=(const char* pc);
	CString operator+(const CString &);
};

#endif
