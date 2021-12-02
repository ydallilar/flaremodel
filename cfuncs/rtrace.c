// Licensed under a 3-clause BSD style license - see LICENSE

#include "math.h"

void rtrace_s(double* res, int sz, double* j_nu, double* a_nu, double dx){
    
    double S=0;
    double S_step;
	int i;

    for(i=0; i<sz; i++){
        if (dx*a_nu[i] < 1e-4)
            S_step = j_nu[i]*dx;
        else
            S_step = j_nu[i]/a_nu[i]*(1-exp(-dx*a_nu[i]));
        S = S*exp(-dx*a_nu[i])+S_step;
        res[i] = S;
    }

}

void rtrace(double* res, int sz1, int sz2, double* j_nu, double* a_nu, double dx){
 
	int i;

    for(i=0; i<sz1; i++){
        rtrace_s(&res[i*sz2], sz2, &j_nu[i*sz2], &a_nu[i*sz2], dx);
    }

}
