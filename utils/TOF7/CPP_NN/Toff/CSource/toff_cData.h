#ifndef TOFF_CDATA_H
#define TOFF_CDATA_H


class CToffCoffFileData
{
public:
	struct SingleSummand;

private:
	int *m_nofcoffArr_XnAk;
	SingleSummand **m_ssArr;

protected:
	int m_maxorder, m_maxorder2; // do not make static , winds can have different order
	void set_max_order(int mo)
	{ 
		m_maxorder = mo; m_maxorder2 = 2*mo; 
	}

public:	
	CToffCoffFileData(void);
	~CToffCoffFileData(void);
	int getNofCoff(int n, int k);
	void setNofCoff(int n, int k, int mnof);
	SingleSummand* getSummands(int n, int k);
	void readFile(const char *fullfname);

public:
	struct SingleSummand
	{
		bool m_binit;
		int *m_snPotArr; 
		CToffCoffFileData *m_ptffff;
		double m_coff;

		
		SingleSummand(void);
		SingleSummand(CToffCoffFileData *p);
		~SingleSummand(void);
		void initialize(CToffCoffFileData *p);
		void setRow(int *iarr, double x);
		double getValue(double *sn_at_l);
		inline double getCoff(void){ return m_coff; }
		int calcOrder_from_sn(void);
		int getOrder(void);
		int getExpo(int n);
	};
	friend struct SingleSummand;
};

#endif
