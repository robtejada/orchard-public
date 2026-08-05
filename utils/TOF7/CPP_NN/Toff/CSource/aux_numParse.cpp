// C++ code by Nadine Nettelmann, 2004+
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "./aux_numParse.h"
#include "./aux_utils.h"

bool isBlankChar( char c )
{
	if( c==' ' || c=='\t' || c=='\n' || c=='\r' || c==',')	return true;
	return false;
}

bool isPureBlankChar( char c )
{
	if( c==' ' || c=='\t' )	return true;
	return false;
}

bool isNewLineChar( char c )
{
	if( c=='\n' || c=='\r' )	return true;
	return false;
}

bool isDigitChar( char c )
{
	if( c>='0' && c<='9' )	return true;
	return false;
}

int getNofCols_thisRow( FILE* pF )
{
	char	c = 0;
	int		ncols = 0;
	bool	bFirstNumberFound = false;
	long	fposition = ftell(pF);

	while(!feof(pF) )
	{
		c = fgetc(pF);

		if( c == '#' )
		{
			gotoNextLine(pF);
			continue;
		}
		if( isDigitChar(c) )
		{
			ncols++;
			bFirstNumberFound = true;
			for(;;)
			{
				if( feof(pF) )	break;
				c = fgetc(pF);
				if( isBlankChar(c) )	break;
			}
		}
		if( isNewLineChar(c) && bFirstNumberFound )	break;
	}
	fseek(pF,fposition,SEEK_SET);
	return ncols;
}

void gotoNextLine(FILE* &pF)
{
	char c;
	long int position = 0;

	while(!feof(pF))
	{
		c = fgetc(pF);
		if( isNewLineChar(c) )
		{
			break;
		}
	}
	while(!feof(pF))
	{
		position = ftell(pF);
		c = fgetc(pF);
		if( !isNewLineChar(c) || !isBlankChar(c) )
		{            
			//fseek(pF,-1L,SEEK_CUR);
			position = ftell(pF);

			if(fseek(pF,position-1,SEEK_SET))
			{
				printf("Fehler bei fseek()! \n"); exit_abfrage();
			}
			position = ftell(pF);
			//c = fgetc(pF);
			//position = ftell(pF);
			break;
		}
	}
}

bool readNextNumber_e( FILE* pFile, double &result )
{
	double	sum = 0.0;
	char	c='?', sign;
	static int	nc=0;

	result = 0.0;
	while( !feof(pFile) )
	{
		c = fgetc(pFile); nc++; 
		if( isBlankChar(c) )	continue;
		if(c == '#')
		{
			gotoNextLine(pFile);
			continue;
		}
		sign = 1;
		if( c=='+' || c=='-' )
		{
			if( c=='-' )	sign=-1;
			c = fgetc(pFile); nc++;
		}
		if( isDigitChar(c) )
		{
			sum = c-'0';
			for(;;)
			{
				if( feof(pFile) )	break;
				c = fgetc(pFile); nc++;
				if( isDigitChar(c) )	sum = sum*10.0 + (c-'0');
				else					break;
			}
			if( c=='.' )
			{
				for( double place = 0.1;;place*=0.1 )
				{
					if( feof(pFile) )	break;
					c = fgetc(pFile); nc++;
					if( isDigitChar(c) )	sum += (c-'0')*place;
					else					break;
				}
			}
			if( c=='e' || c=='E' )
			{
				double	expSign = 1, 
						expo	= 0.0;

				for(;;)
				{
					if( feof(pFile)	)	break;
					c = fgetc(pFile); nc++;
					if( c=='-' || c=='+' )
					{
						if( c=='-' )	expSign = -1;
						continue;
					}
					if( isDigitChar(c) )	expo = expo*10 + (c-'0');
					else
					{
						sum *= pow(1e1, expSign*expo);
						break;
					}
				}
			}
			result = (sign==-1) ? -sum : sum;
			return(true);
		}
	}
	return false;
}


bool readNextNumber_d(FILE *pFi, int &result)
{
	int		sum = 0;
	char	c, sign;

	while( !feof(pFi) )
	{
		c = fgetc(pFi); 
		if( isBlankChar(c) )	continue;

		if(c == '#')
		{
			gotoNextLine(pFi);
			continue;
		}
		sign = 1;
		if( c=='+' || c=='-' )
		{
			if( c=='-' )	sign=-1;
			c = fgetc(pFi);
		}
		if( isDigitChar(c) )
		{
			sum = c-'0';
			for(;;)
			{
				if( feof(pFi) )	break;
				c = fgetc(pFi);
				if( isDigitChar(c) )	sum = sum*10 + (c-'0');
				else					break;
			}
			result = (sign==-1) ? -sum : sum;
			return true;
		}
	}
	return false;
}


void copyFile(const char* fnameIn, const char* fnameOut)
{
	FILE	*pFiIn=NULL, *pFiOut=NULL;
	char	zeichen;

	if( !strcmp(fnameIn,fnameOut) )	return;
	openFile(pFiIn, fnameIn, "rb");
	openFile(pFiOut, fnameOut, "w+b");
	while(!feof(pFiIn))
	{
		fread(&zeichen,1,1,pFiIn);		
		fwrite(&zeichen,1,1,pFiOut);		
	};
	closeFile(pFiIn);
	closeFile(pFiOut);
}


