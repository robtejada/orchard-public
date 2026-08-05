
#ifndef NUMPARSE_H
#define NUMPARSE_H

bool readNextNumber_e( FILE* pFile, double &result );
bool readNextNumber_d( FILE* pFile, int &result);
bool isBlankChar( char c );
bool isPureBlankChar( char c );
bool isNewLineChar( char c );
bool isDigitChar( char c );
void gotoNextLine(FILE* &pFi);
void copyFile(const char* fnameIn, const char* fnameOut);

#endif
